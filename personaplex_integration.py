"""
PersonaPlex Integration Module
Handles the full-duplex voice conversation with NVIDIA's PersonaPlex model
"""

import os
import torch
import numpy as np
from typing import Optional, Tuple, AsyncGenerator
from dataclasses import dataclass

@dataclass
class VoiceConfig:
    """Configuration for voice synthesis"""
    language: str = "fr"
    voice_style: str = "warm_teacher"
    speaking_rate: float = 0.9  # Slightly slower for learners
    pitch: float = 1.0


class PersonaPlexWrapper:
    """
    Wrapper for NVIDIA PersonaPlex model.

    PersonaPlex is a full-duplex speech-to-speech model that:
    - Transcribes input speech in real-time
    - Generates responses while still listening
    - Synthesizes speech output
    - Handles interruptions naturally (barge-in)
    """

    def __init__(self, model_path: str = "nvidia/PersonaPlex"):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.voice_config = VoiceConfig()

    def load(self) -> bool:
        """Load the PersonaPlex model"""
        try:
            # Check GPU availability
            if not torch.cuda.is_available():
                print("Warning: CUDA not available. PersonaPlex requires GPU.")
                return False

            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {gpu_memory:.1f} GB")

            if gpu_memory < 40:
                print("Warning: PersonaPlex works best with 40GB+ GPU memory")

            # Load model - adjust import based on actual PersonaPlex API
            # This is conceptual as PersonaPlex's exact API may differ
            try:
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

                print(f"Loading PersonaPlex from {self.model_path}...")

                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    token=os.getenv("HF_TOKEN")
                )

                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=os.getenv("HF_TOKEN")
                )

                self.is_loaded = True
                print("PersonaPlex loaded successfully!")
                return True

            except ImportError as e:
                print(f"Required libraries not installed: {e}")
                return False

        except Exception as e:
            print(f"Failed to load PersonaPlex: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.
        In full PersonaPlex, this happens in real-time with streaming.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Process audio through the model
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)

        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return transcription

    def synthesize(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize text to speech.
        Returns (audio_array, sample_rate)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        config = voice_config or self.voice_config

        # In actual PersonaPlex, synthesis is integrated with response generation
        # This shows the conceptual API
        inputs = self.processor(
            text=text,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            audio_output = self.model.generate_speech(
                **inputs,
                language=config.language,
                speaking_rate=config.speaking_rate
            )

        # Convert to numpy
        audio_array = audio_output.cpu().numpy()
        sample_rate = 24000  # PersonaPlex output sample rate

        return audio_array, sample_rate

    async def stream_conversation(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        response_callback
    ):
        """
        Full-duplex streaming conversation.

        This is where PersonaPlex shines - it can:
        - Listen while speaking
        - Interrupt itself when user speaks
        - Generate responses incrementally
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # This would integrate with PersonaPlex's streaming API
        # Conceptual implementation:

        async for audio_chunk in audio_stream:
            # Check for voice activity (barge-in detection)
            # Transcribe incrementally
            # Generate response when appropriate
            # Stream audio output
            pass


class SimulatedPersonaPlex:
    """
    Simulated PersonaPlex for testing without GPU.
    Uses OpenAI Whisper for STT and browser for TTS.
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.is_loaded = True

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Use Whisper for transcription"""
        import io
        import soundfile as sf

        # Convert numpy to bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='wav')
        buffer.seek(0)
        buffer.name = "audio.wav"

        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer
        )

        return response.text

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Use OpenAI TTS for synthesis"""
        import io
        import soundfile as sf

        response = self.client.audio.speech.create(
            model="tts-1",
            voice="nova",  # Closest to French female
            input=text
        )

        # Convert to numpy array
        audio_data = io.BytesIO(response.content)
        audio_array, sample_rate = sf.read(audio_data)

        return audio_array, sample_rate


def get_voice_model():
    """Get the appropriate voice model based on environment"""

    # Try PersonaPlex first
    personaplex = PersonaPlexWrapper()
    if personaplex.load():
        return personaplex

    # Fall back to simulated version
    print("Using simulated PersonaPlex (OpenAI Whisper + TTS)")
    return SimulatedPersonaPlex()
