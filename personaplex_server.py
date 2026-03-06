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
    global voice_prompts, device

    if voice_name in voice_prompts:
        return voice_prompts[voice_name]

    # Find voice prompt in HF cache (extracted from voices.tgz)
    import glob
    cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices"
    )
    voice_dirs = glob.glob(cache_pattern)

    voice_path = None
    for voice_dir in voice_dirs:
        candidate = os.path.join(voice_dir, voice_name)
        if os.path.exists(candidate):
            voice_path = candidate
            break

    if not voice_path:
        raise FileNotFoundError(f"Voice prompt {voice_name} not found in cache")

    print(f"[PersonaPlex] Loading voice prompt: {voice_path}")
    voice_data = torch.load(voice_path, map_location=device, weights_only=True)
    voice_prompts[voice_name] = voice_data
    return voice_data


def load_model():
    """Load PersonaPlex model once at startup"""
    global moshi_available, lm_gen, mimi, other_mimi, text_tokenizer, device

    try:
        from moshi.models import loaders
        from moshi.models.lm import LMGen
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
            filename="model.safetensors",
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

        # Enter streaming mode (required before reset_streaming)
        print("[PersonaPlex] Entering streaming mode...")
        mimi.streaming_forever(1)
        other_mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        # Warmup
        print("[PersonaPlex] Warming up...")
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
            codes = mimi.encode(chunk)
            _ = lm_gen.step(codes)

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

    # Find voice prompt path
    import glob
    cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices"
    )
    voice_dirs = glob.glob(cache_pattern)
    voice_path = None
    for voice_dir in voice_dirs:
        candidate = os.path.join(voice_dir, voice_prompt)
        if os.path.exists(candidate):
            voice_path = candidate
            break

    if not voice_path:
        raise FileNotFoundError(f"Voice prompt {voice_prompt} not found")

    # 1) Load voice prompt (for .pt files use load_voice_prompt_embeddings)
    if voice_prompt.endswith('.pt'):
        lm_gen.load_voice_prompt_embeddings(voice_path)
    else:
        lm_gen.load_voice_prompt(voice_path)

    # 2) Set text prompt
    if text_prompt:
        lm_gen.text_prompt_tokens = text_tokenizer.encode(
            wrap_with_system_tags(text_prompt)
        )
    else:
        lm_gen.text_prompt_tokens = None

    # 3) Reset all streaming states
    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()

    # 4) Run prompt phases (voice + text injection)
    lm_gen.step_system_prompts(mimi)

    # 5) Reset mimi after voice prompt encoding
    mimi.reset_streaming()

    # 6) Resample input to 24kHz if needed
    if sample_rate != 24000:
        from scipy import signal
        num_samples = int(len(input_audio) * 24000 / sample_rate)
        input_audio = signal.resample(input_audio, num_samples).astype(np.float32)

    # 7) Encode and process user audio frame by frame
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    output_frames = []
    output_text_tokens = []

    with torch.no_grad():
        # Pad input to multiple of frame_size
        pad_len = (frame_size - len(input_audio) % frame_size) % frame_size
        if pad_len > 0:
            input_audio = np.pad(input_audio, (0, pad_len))

        # Process frame by frame
        for i in range(0, len(input_audio), frame_size):
            chunk = input_audio[i:i+frame_size]
            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
            codes = mimi.encode(chunk_tensor)

            # Step with user input
            tokens = lm_gen.step(codes)
            if tokens is not None:
                # Decode agent audio
                pcm = other_mimi.decode(tokens[:, :8, :])  # First 8 codebooks are audio
                output_frames.append(pcm.squeeze().cpu().numpy())

                # Get text token if present
                text_token = tokens[:, 8, :].item() if tokens.shape[1] > 8 else None
                if text_token is not None and text_token > 0:
                    output_text_tokens.append(text_token)

    # Concatenate output frames
    if output_frames:
        output_audio = np.concatenate(output_frames)
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
