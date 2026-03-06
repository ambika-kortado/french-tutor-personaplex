#!/usr/bin/env python3
"""
PersonaPlex Inference Server with Per-Request Text Prompts

Loads the model ONCE at startup, then accepts requests with custom text_prompts.
Much faster than offline mode which reloads the model each time.

Usage:
    python personaplex_server.py --port 8999

API:
    POST /generate
    {
        "audio_base64": "...",  # Input audio as base64-encoded WAV
        "text_prompt": "You are a history tutor...",
        "voice_prompt": "NATF2.pt"
    }

    Returns:
    {
        "audio_base64": "...",  # Output audio as base64-encoded WAV
        "transcript": "..."
    }
"""

import os
import sys
import json
import base64
import tempfile
import argparse
import asyncio
from typing import Optional

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Will be imported after checking availability
moshi_available = False
lm_gen = None
mimi = None
other_mimi = None
text_tokenizer = None
device = None
voice_prompts = {}


class GenerateRequest(BaseModel):
    audio_base64: str
    text_prompt: str = ""
    voice_prompt: str = "NATF2.pt"


class GenerateResponse(BaseModel):
    audio_base64: str
    transcript: str


app = FastAPI(title="PersonaPlex Inference Server")


def wrap_with_system_tags(text: str) -> str:
    """Wrap text with system tags for PersonaPlex"""
    return f"<system>{text}</system>"


def load_voice_prompt(voice_name: str):
    """Load a voice prompt from the model cache"""
    global voice_prompts

    if voice_name in voice_prompts:
        return voice_prompts[voice_name]

    # Find voice prompt file
    from huggingface_hub import hf_hub_download

    voice_path = hf_hub_download(
        repo_id="nvidia/personaplex-7b-v1",
        filename=f"voices/{voice_name}",
        token=os.environ.get("HF_TOKEN")
    )

    voice_data = torch.load(voice_path, map_location=device)
    voice_prompts[voice_name] = voice_data
    return voice_data


def load_model():
    """Load PersonaPlex model once at startup"""
    global moshi_available, lm_gen, mimi, other_mimi, text_tokenizer, device

    try:
        from moshi.models import loaders
        from moshi.models.lm_gen import LMGen
        import sentencepiece
        from huggingface_hub import hf_hub_download

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PersonaPlex] Using device: {device}")

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("[PersonaPlex] Warning: HF_TOKEN not set, model download may fail")

        # Download model weights
        print("[PersonaPlex] Downloading model weights...")
        mimi_weight = hf_hub_download(
            repo_id="nvidia/personaplex-7b-v1",
            filename="tokenizer-e351c8d8-checkpoint125.safetensors",
            token=hf_token
        )
        moshi_weight = hf_hub_download(
            repo_id="nvidia/personaplex-7b-v1",
            filename="personaplex-e0eb9f6d-checkpoint1000.safetensors",
            token=hf_token
        )
        tokenizer_path = hf_hub_download(
            repo_id="nvidia/personaplex-7b-v1",
            filename="tokenizer_spm_32k_3.model",
            token=hf_token
        )

        # Load models
        print("[PersonaPlex] Loading Mimi encoder/decoder...")
        mimi = loaders.get_mimi(mimi_weight, device)
        other_mimi = loaders.get_mimi(mimi_weight, device)

        print("[PersonaPlex] Loading tokenizer...")
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        print("[PersonaPlex] Loading Moshi LM (this takes a while)...")
        lm = loaders.get_moshi_lm(moshi_weight, device=device)
        lm.eval()

        print("[PersonaPlex] Creating LMGen...")
        lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
            sample_rate=mimi.sample_rate,
            device=device,
            frame_rate=mimi.frame_rate,
            use_sampling=True,
            temp=0.6,
            temp_text=0.7,
            top_k=200,
            top_k_text=25,
        )

        moshi_available = True
        print("[PersonaPlex] Model loaded successfully!")
        return True

    except Exception as e:
        print(f"[PersonaPlex] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


async def generate_response(
    input_audio: np.ndarray,
    sample_rate: int,
    text_prompt: str,
    voice_prompt: str
) -> tuple[np.ndarray, str]:
    """Generate a response with custom text_prompt"""
    global lm_gen, mimi, other_mimi, text_tokenizer, device

    if not moshi_available:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Run in thread pool to not block
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _generate_sync,
        input_audio,
        sample_rate,
        text_prompt,
        voice_prompt
    )


def _generate_sync(
    input_audio: np.ndarray,
    sample_rate: int,
    text_prompt: str,
    voice_prompt: str
) -> tuple[np.ndarray, str]:
    """Synchronous generation (runs in thread pool)"""
    global lm_gen, mimi, other_mimi, text_tokenizer, device

    # Reset the generator state
    lm_gen.reset()

    # Set text prompt (the fast part!)
    if text_prompt:
        lm_gen.text_prompt_tokens = text_tokenizer.encode(
            wrap_with_system_tags(text_prompt)
        )
    else:
        lm_gen.text_prompt_tokens = None

    # Load voice prompt
    voice_data = load_voice_prompt(voice_prompt)
    lm_gen.voice_prompt_tokens = voice_data.get("codes")
    lm_gen.voice_prompt_embeddings = voice_data.get("embeddings")

    # Resample input to 24kHz if needed
    if sample_rate != 24000:
        from scipy import signal
        num_samples = int(len(input_audio) * 24000 / sample_rate)
        input_audio = signal.resample(input_audio, num_samples).astype(np.float32)

    # Encode input audio
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_audio).unsqueeze(0).unsqueeze(0).to(device)
        input_codes = mimi.encode(input_tensor)

    # Step through system prompts (voice + text)
    lm_gen.step_system_prompts(mimi)

    # Generate response
    output_codes = []
    output_text_tokens = []

    # Feed input and generate
    for i in range(input_codes.shape[-1]):
        frame = input_codes[:, :, i:i+1]
        out = lm_gen.step(frame)
        if out is not None:
            audio_tokens, text_token = out
            output_codes.append(audio_tokens)
            if text_token is not None:
                output_text_tokens.append(text_token)

    # Continue generating until done or max length
    max_gen_frames = 500  # ~10 seconds
    for _ in range(max_gen_frames):
        out = lm_gen.step(None)  # No more input
        if out is None:
            break
        audio_tokens, text_token = out
        output_codes.append(audio_tokens)
        if text_token is not None:
            output_text_tokens.append(text_token)

    # Decode audio
    if output_codes:
        output_tensor = torch.cat(output_codes, dim=-1)
        with torch.no_grad():
            output_audio = other_mimi.decode(output_tensor)
        output_audio = output_audio.squeeze().cpu().numpy()
    else:
        output_audio = np.zeros(24000, dtype=np.float32)

    # Decode text
    if output_text_tokens:
        transcript = text_tokenizer.decode(output_text_tokens)
    else:
        transcript = ""

    return output_audio, transcript


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    """Generate a response with custom text_prompt"""

    # Decode input audio
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        input_audio, sample_rate = sf.read(temp_path)
        os.unlink(temp_path)

        if len(input_audio.shape) > 1:
            input_audio = input_audio[:, 0]  # Take first channel
        input_audio = input_audio.astype(np.float32)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    # Generate
    try:
        output_audio, transcript = await generate_response(
            input_audio,
            sample_rate,
            request.text_prompt,
            request.voice_prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Encode output audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, output_audio, 24000)
        with open(f.name, "rb") as audio_file:
            output_base64 = base64.b64encode(audio_file.read()).decode()
        os.unlink(f.name)

    return GenerateResponse(
        audio_base64=output_base64,
        transcript=transcript
    )


@app.get("/health")
async def health():
    return {
        "status": "ok" if moshi_available else "model_not_loaded",
        "model_loaded": moshi_available
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print("[PersonaPlex] Starting inference server...")
    print("[PersonaPlex] Loading model (this takes 30-60 seconds)...")

    if not load_model():
        print("[PersonaPlex] Failed to load model, exiting")
        sys.exit(1)

    print(f"[PersonaPlex] Server ready at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
