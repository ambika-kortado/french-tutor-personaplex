#!/usr/bin/env python3
"""
Dual-Track Speech-to-Speech History Tutor

Architecture:
- Track 1: PersonaPlex S2S for fast fillers/acknowledgments
- Track 2: Claude API for substantive tutoring responses
- Arbitration: Claude wins if < 1.2s, otherwise filler then Claude

Usage:
    python dual_track_tutor.py --topic "World War II"
"""

import os
import sys
import json
import asyncio
import argparse
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API clients
from openai import OpenAI

# Configuration
LLM_MODEL = "gpt-4o-mini"  # Faster than gpt-4o, still smart
LLM_TIMEOUT = 0.8  # seconds before falling back to filler
MOSHI_FIRST_MODE = True  # Moshi handles conversation, GPT knowledge sent as audio prefix
SAMPLE_RATE = 24000

# Session state
@dataclass
class SessionState:
    """Tracks conversation state across turns"""
    topic: str = "World War II"
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    last_speaker: str = "assistant"  # "user" or "assistant"
    barge_in_detected: bool = False
    current_tasks: List[asyncio.Task] = field(default_factory=list)

    def get_system_prompt(self) -> str:
        return f"""You are a Socratic history tutor teaching {self.topic}.
Keep responses to 2–3 conversational sentences.
End every turn with a question.
Never use lists.
Adapt complexity to the student."""

    def add_user_message(self, text: str):
        self.conversation_history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str):
        self.conversation_history.append({"role": "assistant", "content": text})

    def get_messages_for_claude(self) -> List[Dict[str, str]]:
        """Get conversation history formatted for Claude API"""
        return self.conversation_history[-20:]  # Keep last 20 messages


# Global state
session_state: Optional[SessionState] = None
log_file: Optional[str] = None


# =============================================================================
# Track 1: PersonaPlex S2S (Fast Fillers)
# =============================================================================

class PersonaPlexTrack:
    """
    PersonaPlex/Moshi Speech-to-Speech track for fast filler responses.

    Runs Moshi server as subprocess and connects via WebSocket for
    true end-to-end speech-to-speech with contextual responses.
    """

    def __init__(self):
        self.is_loaded = False
        self.moshi_process = None
        self.moshi_ws = None
        self.moshi_port = 8998
        self.use_moshi_s2s = False

    async def load(self) -> bool:
        """Load PersonaPlex/Moshi S2S model"""
        try:
            print("[PersonaPlex] Checking for Moshi S2S...", flush=True)

            import websockets

            # First, check if Moshi server is already running
            print("[PersonaPlex] Checking for existing Moshi server...", flush=True)
            try:
                self.moshi_ws = await asyncio.wait_for(
                    websockets.connect(f"ws://localhost:{self.moshi_port}/api/chat"),
                    timeout=3
                )
                self.use_moshi_s2s = True
                self.is_loaded = True
                print("[PersonaPlex] Connected to existing Moshi server!", flush=True)
                return True
            except:
                print("[PersonaPlex] No existing server found", flush=True)

            # Check for GPU
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[PersonaPlex] No GPU - using text fillers", flush=True)
                    print("[PersonaPlex] To enable Moshi S2S, run: python -m moshi.server --port 8998", flush=True)
                    self.is_loaded = True
                    return True
            except:
                pass

            # Try to start server as subprocess (optional - can skip if slow)
            try:
                from moshi.models import loaders
                import subprocess

                print("[PersonaPlex] Starting Moshi server (this takes 30-60s)...", flush=True)
                print("[PersonaPlex] Or run separately: python -m moshi.server --port 8998", flush=True)

                self.moshi_process = subprocess.Popen(
                    ["python", "-m", "moshi.server", "--port", str(self.moshi_port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )

                # Wait for server with retries
                for attempt in range(15):  # 45 seconds total
                    await asyncio.sleep(3)

                    if self.moshi_process.poll() is not None:
                        stdout = self.moshi_process.stdout.read().decode() if self.moshi_process.stdout else ""
                        print(f"[PersonaPlex] Server exited: {stdout[:300]}", flush=True)
                        break

                    try:
                        self.moshi_ws = await asyncio.wait_for(
                            websockets.connect(f"ws://localhost:{self.moshi_port}/api/chat"),
                            timeout=5
                        )
                        self.use_moshi_s2s = True
                        self.is_loaded = True
                        print("[PersonaPlex] Moshi S2S connected!", flush=True)
                        return True
                    except:
                        print(f"[PersonaPlex] Waiting for server... ({(attempt+1)*3}s)", flush=True)

                if self.moshi_process and self.moshi_process.poll() is None:
                    self.moshi_process.terminate()

            except ImportError:
                pass

            print("[PersonaPlex] Using text filler mode", flush=True)
            self.is_loaded = True
            return True

        except Exception as e:
            print(f"[PersonaPlex] Error: {e}", flush=True)
            self.is_loaded = True
            return True

    async def generate_filler(self, user_audio: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate a fast filler/acknowledgment response.

        If Moshi S2S is available, sends audio and gets contextual response.
        Otherwise returns pre-defined text fillers.
        """
        start_time = time.time()

        # Try Moshi S2S if available
        if self.use_moshi_s2s and self.moshi_ws and user_audio is not None:
            try:
                import sphn
                import scipy.signal as signal

                FRAME_SIZE = 1920  # 80ms at 24kHz

                # Resample to 24kHz for Moshi
                if len(user_audio) > 0:
                    # Assume input is 44.1kHz, resample to 24kHz
                    num_samples_24k = int(len(user_audio) * 24000 / 44100)
                    audio_24k = signal.resample(user_audio, num_samples_24k).astype(np.float32)

                    # Send audio in chunks (frame_size = 1920 samples)
                    opus_writer = sphn.OpusStreamWriter(24000)
                    for i in range(0, len(audio_24k), FRAME_SIZE):
                        chunk = audio_24k[i:i + FRAME_SIZE]
                        if len(chunk) < FRAME_SIZE:
                            # Pad last chunk
                            chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
                        opus_bytes = opus_writer.append_pcm(chunk)
                        if opus_bytes:
                            await self.moshi_ws.send(b"\x01" + opus_bytes)

                    # Get response (collect audio until silence)
                    response_audio = []
                    deadline = time.time() + 3.0  # Up to 3 seconds
                    async for msg in self._recv_moshi_audio():
                        response_audio.append(msg)
                        if time.time() > deadline:
                            break

                    if response_audio:
                        audio_bytes = b"".join(response_audio)
                        latency = (time.time() - start_time) * 1000
                        print(f"[PersonaPlex] Moshi S2S responded in {latency:.0f}ms", flush=True)
                        return {
                            "text": "[Moshi S2S response]",
                            "audio": audio_bytes,
                            "latency_ms": latency
                        }
            except Exception as e:
                print(f"[PersonaPlex] Moshi S2S error: {e}", flush=True)

        # Fallback: Return pre-defined fillers
        fillers = [
            "That's a great question about history!",
            "Interesting point! Let me think about that.",
            "Hmm, that's worth exploring.",
            "Good thinking! Here's what I know...",
            "Ah yes, that's an important topic.",
        ]

        import random
        filler_text = random.choice(fillers)
        latency = (time.time() - start_time) * 1000

        return {
            "text": filler_text,
            "audio": None,
            "latency_ms": latency
        }

    async def _recv_moshi_audio(self):
        """Receive audio frames from Moshi"""
        import sphn
        opus_reader = sphn.OpusStreamReader(24000)

        while True:
            try:
                # Use wait_for to prevent blocking forever
                msg = await asyncio.wait_for(self.moshi_ws.recv(), timeout=0.5)
                if isinstance(msg, bytes) and len(msg) > 0:
                    kind = msg[0]
                    if kind == 1:  # Audio
                        pcm = opus_reader.append_bytes(msg[1:])
                        if pcm is not None and len(pcm) > 0:
                            print(f"[PersonaPlex] Got audio chunk: {len(pcm)} samples", flush=True)
                            yield (pcm * 32767).astype(np.int16).tobytes()
                    elif kind == 2:  # Text token
                        text = msg[1:].decode('utf-8', errors='ignore')
                        print(f"[PersonaPlex] Moshi says: {text}", flush=True)
            except asyncio.TimeoutError:
                print(f"[PersonaPlex] No more audio from Moshi", flush=True)
                break
            except Exception as e:
                print(f"[PersonaPlex] Recv error: {e}", flush=True)
                break

    async def inject_knowledge(self, knowledge_text: str) -> bool:
        """
        Inject GPT's knowledge into Moshi as text context.
        Moshi will naturally incorporate this into its response.
        """
        if not self.use_moshi_s2s or not self.moshi_ws:
            return False

        try:
            # Send text to Moshi (type 2 = text tokens)
            # Format: \x02 + UTF-8 text
            text_bytes = knowledge_text.encode('utf-8')
            await self.moshi_ws.send(b"\x02" + text_bytes)
            print(f"[PersonaPlex] Injected knowledge: {knowledge_text[:50]}...", flush=True)
            return True
        except Exception as e:
            print(f"[PersonaPlex] Failed to inject knowledge: {e}", flush=True)
            return False

    async def converse_with_knowledge(
        self,
        user_audio: np.ndarray,
        knowledge_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full conversation through Moshi S2S, with optional GPT knowledge injection.
        This is the "Moshi-first" mode where Moshi handles conversation naturally.
        """
        start_time = time.time()

        if not self.use_moshi_s2s or not self.moshi_ws:
            # Fallback if Moshi not available
            return await self.generate_filler(user_audio)

        try:
            import sphn
            import scipy.signal as signal

            FRAME_SIZE = 1920  # 80ms at 24kHz

            # Resample to 24kHz for Moshi
            if len(user_audio) > 0:
                num_samples_24k = int(len(user_audio) * 24000 / 44100)
                audio_24k = signal.resample(user_audio, num_samples_24k).astype(np.float32)

                # If we have knowledge to inject, send it first
                if knowledge_text:
                    await self.inject_knowledge(knowledge_text)

                # Send audio in chunks (frame_size = 1920 samples)
                opus_writer = sphn.OpusStreamWriter(24000)
                for i in range(0, len(audio_24k), FRAME_SIZE):
                    chunk = audio_24k[i:i + FRAME_SIZE]
                    if len(chunk) < FRAME_SIZE:
                        chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))
                    opus_bytes = opus_writer.append_pcm(chunk)
                    if opus_bytes:
                        await self.moshi_ws.send(b"\x01" + opus_bytes)

                # Collect response audio with timeout
                response_audio = []
                deadline = time.time() + 2.0  # 2 second timeout for conversation
                async for audio_chunk in self._recv_moshi_audio():
                    response_audio.append(audio_chunk)
                    if time.time() > deadline:
                        break
                    if len(response_audio) >= 10:  # Got enough audio
                        break

                if response_audio:
                    audio_bytes = b"".join(response_audio)
                    latency = (time.time() - start_time) * 1000
                    print(f"[PersonaPlex] Moshi responded in {latency:.0f}ms", flush=True)
                    return {
                        "text": "[Moshi S2S with knowledge]" if knowledge_text else "[Moshi S2S]",
                        "audio": audio_bytes,
                        "latency_ms": latency,
                        "used_knowledge": knowledge_text is not None
                    }

        except Exception as e:
            print(f"[PersonaPlex] Conversation error: {e}", flush=True)

        return await self.generate_filler(user_audio)

    async def cleanup(self):
        """Cleanup Moshi resources"""
        if self.moshi_ws:
            await self.moshi_ws.close()
        if self.moshi_process:
            self.moshi_process.terminate()

    async def process_audio_stream(
        self,
        audio_chunks: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Process audio stream through PersonaPlex S2S."""
        if audio_chunks:
            combined = np.concatenate(audio_chunks)
            return await self.generate_filler(combined)
        return await self.generate_filler()


# =============================================================================
# Track 2: OpenAI GPT-4o (Substantive Responses)
# =============================================================================

class LLMTrack:
    """OpenAI GPT-4o track for substantive tutoring responses"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        self.client = OpenAI(api_key=api_key)

    async def generate_response(
        self,
        user_text: str,
        session: SessionState
    ) -> Dict[str, Any]:
        """
        Generate a substantive tutoring response using GPT-4o.

        Returns:
            Dict with 'text' and 'latency_ms' keys
        """
        start_time = time.time()

        try:
            # Build messages
            messages = [
                {"role": "system", "content": session.get_system_prompt()}
            ]
            messages.extend(session.get_messages_for_claude())
            messages.append({"role": "user", "content": user_text})

            # Call OpenAI API (using asyncio.to_thread for sync client)
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=LLM_MODEL,
                max_tokens=200,  # Keep responses concise
                temperature=0.8,
                messages=messages
            )

            response_text = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000

            return {
                "text": response_text,
                "latency_ms": latency_ms
            }

        except Exception as e:
            print(f"[LLM] Error: {e}")
            return {
                "text": "I'm having trouble thinking right now. Could you repeat that?",
                "latency_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }


# =============================================================================
# TTS Engine (for Claude responses)
# =============================================================================

class TTSEngine:
    """Text-to-Speech using Moshi for PersonaPlex voice quality"""

    # Available voices from kyutai/tts-voices repo
    VOICES = [
        "expresso/ex03-ex01_calm_001_channel1_1143s.wav",
        "expresso/ex03-ex01_happy_001_channel1_334s.wav",
        "expresso/ex01-ex02_default_001_channel1_168s.wav",
    ]

    def __init__(self):
        self.is_loaded = False
        self.model = None
        self.use_moshi = False
        self.voice = self.VOICES[0]  # Default calm voice

    async def load(self) -> bool:
        """Load TTS engine - use Moshi on GPU, browser TTS on CPU"""
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda':
            try:
                from moshi.models.tts import get_default_tts_model
                print(f"[TTS] Loading Moshi TTS on GPU...", flush=True)
                self.model = await asyncio.to_thread(
                    get_default_tts_model,
                    device='cuda'
                )
                self.use_moshi = True
                self.is_loaded = True
                print("[TTS] Moshi TTS loaded on GPU!", flush=True)
                return True
            except Exception as e:
                print(f"[TTS] Moshi failed: {e}", flush=True)

        print("[TTS] Using browser TTS (no GPU)", flush=True)
        self.is_loaded = True
        self.use_moshi = False
        return True

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech using Moshi"""
        if not self.is_loaded:
            return None

        if not self.use_moshi:
            # Return None to trigger browser TTS fallback
            return None

        try:
            import torch

            # Use Moshi TTS with voice from kyutai/tts-voices repo
            print(f"[TTS] Synthesizing with voice {self.voice}: {text[:50]}...", flush=True)
            with torch.no_grad():
                audio_tensors = await asyncio.to_thread(
                    self.model.simple_generate,
                    text=text,
                    voice=self.voice,
                    cfg_coef=2.0,
                    show_progress=False
                )

                if isinstance(audio_tensors, list) and len(audio_tensors) > 0:
                    audio = audio_tensors[0].cpu().numpy()
                else:
                    audio = audio_tensors.cpu().numpy()

                # Flatten if needed
                audio = audio.flatten()

                # Convert to 16-bit PCM
                audio_int16 = (audio * 32767).astype(np.int16)
                print(f"[TTS] Generated {len(audio_int16)} samples", flush=True)
                return audio_int16.tobytes()

        except Exception as e:
            print(f"[TTS] Moshi synthesis error: {e}", flush=True)
            return None


# =============================================================================
# STT Engine (Whisper)
# =============================================================================

class STTEngine:
    """Speech-to-Text using Whisper"""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text"""
        import io
        import soundfile as sf

        print(f"[STT] Transcribing {len(audio)} samples at {sample_rate}Hz...", flush=True)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            import scipy.signal as signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)
            sample_rate = 16000
            print(f"[STT] Resampled to {len(audio)} samples at 16kHz", flush=True)

        # Convert to WAV
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='wav')
        buffer.seek(0)
        buffer.name = "audio.wav"

        try:
            response = await asyncio.to_thread(
                self.client.audio.transcriptions.create,
                model="whisper-1",
                file=buffer
            )
            text = response.text.strip()
            print(f"[STT] Transcribed: '{text}'", flush=True)
            return text
        except Exception as e:
            print(f"[STT] Error: {e}", flush=True)
            return ""


# =============================================================================
# Arbitration Layer
# =============================================================================

class Arbitrator:
    """
    Manages the dual-track race between PersonaPlex and LLM.

    Logic:
    - If LLM responds within LLM_TIMEOUT: use LLM only
    - If LLM is slow: speak PersonaPlex filler, then LLM when ready
    - If barge-in detected: cancel everything
    """

    def __init__(
        self,
        personaplex: PersonaPlexTrack,
        llm: LLMTrack,
        tts: TTSEngine,
        stt: STTEngine
    ):
        self.personaplex = personaplex
        self.llm = llm
        self.tts = tts
        self.stt = stt

    async def process_user_turn(
        self,
        audio: np.ndarray,
        session: SessionState,
        send_audio_callback,
        send_status_callback,
        send_json_callback=None,
        sample_rate: int = 44100
    ) -> Dict[str, Any]:
        """
        Process a user turn with dual-track arbitration.

        Returns:
            Dict with details about what was spoken
        """
        result = {
            "user_text": "",
            "personaplex_response": None,
            "claude_response": None,
            "spoken_response": None,
            "personaplex_used": False,
            "claude_used": False,
            "barge_in": False
        }

        # Reset barge-in flag
        session.barge_in_detected = False

        # MOSHI-FIRST MODE: Moshi handles conversation, GPT whispers knowledge
        if MOSHI_FIRST_MODE and self.personaplex.use_moshi_s2s:
            return await self._process_moshi_first(
                audio, session, send_audio_callback, send_status_callback,
                send_json_callback, sample_rate, result
            )

        # Legacy mode: STT → LLM → TTS with filler
        # Step 1: Transcribe user audio (STT)
        await send_status_callback("Transcribing...")
        user_text = await self.stt.transcribe(audio, sample_rate=sample_rate)
        result["user_text"] = user_text

        if not user_text.strip():
            await send_status_callback("Couldn't hear you. Please try again.")
            return result

        # Add to conversation history
        session.add_user_message(user_text)
        await send_status_callback(f"You said: {user_text}")

        # Step 2: Start both tracks in parallel
        # Pass audio to PersonaPlex for contextual S2S response
        personaplex_task = asyncio.create_task(
            self.personaplex.generate_filler(user_audio=audio)
        )
        llm_task = asyncio.create_task(
            self.llm.generate_response(user_text, session)
        )

        session.current_tasks = [personaplex_task, llm_task]

        # Step 3: Wait for LLM with timeout (use asyncio.wait to avoid cancellation)
        done, pending = await asyncio.wait(
            [llm_task],
            timeout=LLM_TIMEOUT
        )

        if llm_task in done:
            # LLM responded in time! Cancel PersonaPlex
            personaplex_task.cancel()

            llm_result = llm_task.result()
            result["claude_response"] = llm_result
            result["claude_used"] = True
            result["spoken_response"] = llm_result["text"]

            # Synthesize and send LLM's response
            await send_status_callback("Speaking...")
            audio_data = await self.tts.synthesize(llm_result["text"])
            if audio_data:
                await send_audio_callback(audio_data)
            elif send_json_callback:
                # Fallback: send text for browser TTS
                print(f"[Arbitrator] TTS failed, sending text for browser speech", flush=True)
                await send_json_callback({"type": "response", "text": llm_result["text"]})

            # Update conversation history
            session.add_assistant_message(llm_result["text"])

        else:
            # LLM is slow - use PersonaPlex filler first
            print(f"[Arbitrator] LLM timeout after {LLM_TIMEOUT}s, using filler", flush=True)

            # Check for barge-in
            if session.barge_in_detected:
                personaplex_task.cancel()
                llm_task.cancel()
                result["barge_in"] = True
                return result

            # Get PersonaPlex filler
            try:
                personaplex_result = await personaplex_task
                result["personaplex_response"] = personaplex_result
                result["personaplex_used"] = True

                # Speak the filler
                await send_status_callback("Speaking filler...")
                filler_audio = await self.tts.synthesize(personaplex_result["text"])
                if filler_audio:
                    await send_audio_callback(filler_audio)
                elif send_json_callback:
                    await send_json_callback({"type": "filler", "text": personaplex_result["text"]})

            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[Arbitrator] Filler error: {e}", flush=True)

            # Now wait for LLM
            print(f"[Arbitrator] Waiting for LLM response...", flush=True)
            try:
                llm_result = await llm_task
                print(f"[Arbitrator] LLM responded: {llm_result.get('text', '')[:50]}...", flush=True)
                result["claude_response"] = llm_result
                result["claude_used"] = True
                result["spoken_response"] = llm_result["text"]

                # Check for barge-in again
                if session.barge_in_detected:
                    result["barge_in"] = True
                    return result

                # Speak LLM's response
                await send_status_callback("Speaking response...")
                print(f"[Arbitrator] Synthesizing LLM response...", flush=True)
                audio_data = await self.tts.synthesize(llm_result["text"])
                if audio_data:
                    print(f"[Arbitrator] Sending LLM audio: {len(audio_data)} bytes", flush=True)
                    await send_audio_callback(audio_data)
                elif send_json_callback:
                    print(f"[Arbitrator] Sending LLM text for browser TTS", flush=True)
                    await send_json_callback({"type": "response", "text": llm_result["text"]})

                # Update conversation history
                session.add_assistant_message(llm_result["text"])

            except asyncio.CancelledError:
                print(f"[Arbitrator] LLM task was cancelled", flush=True)
            except Exception as e:
                print(f"[Arbitrator] LLM error: {e}", flush=True)

        # Log the exchange
        await self.log_exchange(result)

        session.current_tasks = []
        await send_status_callback("Ready - speak when you want!")

        return result

    async def _process_moshi_first(
        self,
        audio: np.ndarray,
        session: SessionState,
        send_audio_callback,
        send_status_callback,
        send_json_callback,
        sample_rate: int,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        MOSHI-FIRST MODE: Send [GPT knowledge audio] + [user audio] to Moshi.

        Flow:
        1. STT user audio → get transcript
        2. GPT generates knowledge from transcript
        3. TTS knowledge → audio "briefing"
        4. Send [briefing audio] + [user audio] to Moshi S2S
        5. Moshi "hears" the knowledge and responds naturally
        """
        import scipy.signal as signal

        await send_status_callback("Processing your question...")

        # Step 1: Transcribe user audio
        user_text = await self.stt.transcribe(audio, sample_rate=sample_rate)
        result["user_text"] = user_text

        if not user_text.strip():
            await send_status_callback("Couldn't hear you. Please try again.")
            return result

        session.add_user_message(user_text)
        print(f"[Moshi-First] User said: {user_text}", flush=True)

        # Step 2: Get GPT knowledge
        await send_status_callback("Thinking...")
        llm_result = await self.llm.generate_response(user_text, session)
        knowledge = llm_result.get("text", "")

        if not knowledge:
            await send_status_callback("No response available")
            return result

        print(f"[Moshi-First] GPT knowledge: {knowledge[:80]}...", flush=True)
        result["claude_response"] = llm_result
        result["claude_used"] = True

        # Step 3: Convert knowledge to audio using Moshi TTS
        await send_status_callback("Preparing response...")
        # Just synthesize the knowledge directly
        briefing_audio_bytes = await self.tts.synthesize(knowledge)

        if briefing_audio_bytes:
            # Convert TTS audio (16-bit PCM at 24kHz) to float32
            briefing_np = np.frombuffer(briefing_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            print(f"[Moshi-First] Briefing audio: {len(briefing_np)} samples ({len(briefing_np)/24000:.1f}s)", flush=True)

            # Resample user audio from sample_rate to 24kHz
            user_24k_len = int(len(audio) * 24000 / sample_rate)
            user_24k = signal.resample(audio, user_24k_len).astype(np.float32)

            # Combine: [briefing] + [pause] + [user question cue]
            pause = np.zeros(int(24000 * 0.3), dtype=np.float32)  # 0.3s pause
            # Just use a short cue from user audio (last 1 second) to trigger response
            user_cue = user_24k[-24000:] if len(user_24k) > 24000 else user_24k
            combined = np.concatenate([briefing_np, pause, user_cue])
            print(f"[Moshi-First] Combined audio: {len(combined)} samples ({len(combined)/24000:.1f}s)", flush=True)

            # Step 4: Send to Moshi S2S
            await send_status_callback("Responding...")
            moshi_result = await self.personaplex.generate_filler(user_audio=combined)

            if moshi_result.get("audio"):
                audio_bytes = moshi_result["audio"]
                print(f"[Moshi-First] Moshi response: {len(audio_bytes)} bytes", flush=True)
                await send_audio_callback(audio_bytes)
                result["personaplex_response"] = moshi_result
                result["personaplex_used"] = True
                result["spoken_response"] = knowledge
                session.add_assistant_message(knowledge)
            else:
                # Fallback: send the TTS audio directly
                print(f"[Moshi-First] No Moshi response, sending TTS directly", flush=True)
                await send_audio_callback(briefing_audio_bytes)
                result["spoken_response"] = knowledge
                session.add_assistant_message(knowledge)
        else:
            # TTS failed, use browser fallback
            print(f"[Moshi-First] TTS failed, using browser fallback", flush=True)
            if send_json_callback:
                await send_json_callback({"type": "response", "text": knowledge})
            result["spoken_response"] = knowledge
            session.add_assistant_message(knowledge)

        await send_status_callback("Ready - speak when you want!")
        return result

    async def handle_barge_in(self, session: SessionState):
        """Cancel all in-flight tasks on barge-in"""
        session.barge_in_detected = True
        for task in session.current_tasks:
            task.cancel()
        session.current_tasks = []

    async def log_exchange(self, result: Dict[str, Any]):
        """Log exchange to session_log.jsonl"""
        global log_file

        if not log_file:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_text": result["user_text"],
            "personaplex_response": result.get("personaplex_response"),
            "claude_response": result.get("claude_response"),
            "spoken_response": result["spoken_response"],
            "personaplex_used": result["personaplex_used"],
            "claude_used": result["claude_used"],
            "barge_in": result["barge_in"]
        }

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"[Log] Error writing: {e}")


# =============================================================================
# Audio Buffer (for collecting user speech)
# =============================================================================

class AudioBuffer:
    """Collects audio chunks until speech ends"""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.8
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.chunks: List[np.ndarray] = []
        self.silence_samples = 0
        self.has_speech = False

    def append(self, chunk: np.ndarray):
        self.chunks.append(chunk)
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms > self.silence_threshold:
            self.has_speech = True
            self.silence_samples = 0
        else:
            self.silence_samples += len(chunk)

    def speech_ended(self) -> bool:
        if not self.has_speech:
            return False
        silence_seconds = self.silence_samples / self.sample_rate
        return silence_seconds >= self.silence_duration

    def get_audio(self) -> np.ndarray:
        if not self.chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.chunks)

    def clear(self):
        self.chunks = []
        self.silence_samples = 0
        self.has_speech = False


# =============================================================================
# FastAPI Server
# =============================================================================

# Global components
personaplex_track: Optional[PersonaPlexTrack] = None
llm_track: Optional[LLMTrack] = None
tts_engine: Optional[TTSEngine] = None
stt_engine: Optional[STTEngine] = None
arbitrator: Optional[Arbitrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup"""
    global personaplex_track, llm_track, tts_engine, stt_engine, arbitrator
    global session_state, log_file

    print("=" * 60)
    print("Dual-Track History Tutor")
    print("=" * 60)

    # Get topic from command line
    topic = getattr(app.state, 'topic', 'World War II')
    print(f"Topic: {topic}")

    # Initialize session
    session_state = SessionState(topic=topic)
    log_file = "session_log.jsonl"

    # Initialize tracks
    print("\n[1/4] Initializing PersonaPlex track...")
    personaplex_track = PersonaPlexTrack()
    await personaplex_track.load()

    print("\n[2/4] Initializing GPT-4o track...")
    llm_track = LLMTrack()
    print("  ✓ GPT-4o ready")

    print("\n[3/4] Initializing TTS engine...")
    tts_engine = TTSEngine()
    await tts_engine.load()

    print("\n[4/4] Initializing STT engine...")
    stt_engine = STTEngine()
    print("  ✓ STT ready")

    # Create arbitrator
    arbitrator = Arbitrator(personaplex_track, llm_track, tts_engine, stt_engine)

    print("\n" + "=" * 60)
    print("Server ready! Open http://localhost:7860 in your browser")
    print("=" * 60 + "\n")

    yield

    print("\nShutting down...")
    # Cleanup Moshi S2S if running
    if personaplex_track:
        await personaplex_track.cleanup()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/dual_track.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": "dual-track",
        "topic": session_state.topic if session_state else "unknown"
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio"""
    await websocket.accept()
    print("Client connected")

    audio_buffer = AudioBuffer()

    async def send_audio(data: bytes):
        print(f"[WS] Sending audio to browser: {len(data)} bytes", flush=True)
        await websocket.send_bytes(data)

    async def send_status(message: str):
        await websocket.send_json({"type": "status", "message": message})

    async def send_json(data: dict):
        await websocket.send_json(data)

    try:
        await send_status(f"Connected! Topic: {session_state.topic}")
        print(f"WebSocket ready, waiting for audio...", flush=True)

        while True:
            try:
                message = await websocket.receive()

                # Handle text messages (like barge-in)
                if "text" in message:
                    try:
                        msg = json.loads(message["text"])
                        if msg.get("type") == "barge_in":
                            print("[WS] Barge-in received, cancelling tasks...", flush=True)
                            await arbitrator.handle_barge_in(session_state)
                            await send_status("Interrupted - listening...")
                            continue
                    except json.JSONDecodeError:
                        pass
                    continue

                # Handle binary audio data
                if "bytes" not in message:
                    continue

                data = message["bytes"]
            except Exception as e:
                print(f"[WS] Error receiving: {e}", flush=True)
                break

            print(f"[Audio] Received audio! Size: {len(data)} bytes", flush=True)

            # Convert to numpy (16-bit PCM)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Browser may send at various sample rates, assume 44.1kHz if longer than expected
            sample_rate = 44100 if len(audio) > 50000 else 16000
            duration = len(audio) / sample_rate
            print(f"[Audio] Duration: {duration:.1f}s, samples: {len(audio)}, rate: {sample_rate}", flush=True)

            if duration < 0.5:  # Less than 0.5 second
                print(f"[Audio] Too short, ignoring", flush=True)
                continue

            # Process through arbitrator (pass sample rate for STT)
            await arbitrator.process_user_turn(
                audio,
                session_state,
                send_audio,
                send_status,
                send_json,
                sample_rate=sample_rate
            )

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")


@app.post("/reset")
async def reset():
    """Reset conversation"""
    global session_state
    if session_state:
        session_state.conversation_history = []
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Dual-Track History Tutor")
    parser.add_argument("--topic", default="World War II", help="History topic to teach")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    # Store topic in app state
    app.state.topic = args.topic

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
