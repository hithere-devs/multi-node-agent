"""Text-to-speech service for agents."""

import io
from abc import ABC, abstractmethod
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class TTSService(ABC):
    """Abstract base class for text-to-speech services."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio bytes (WAV format)
        """
        pass


class MockTTSService(TTSService):
    """Mock TTS service for testing."""

    async def synthesize(self, text: str) -> bytes:
        """Return mock audio data."""
        # Return a minimal WAV header + silence
        # WAV format: 44100 Hz, 16-bit mono, 1 second of silence
        logger.info("mock_tts_synthesize", text_length=len(text))

        wav_header = (
            b"RIFF"
            + b"\x24\xf0\x00\x00"  # File size - 8
            + b"WAVE"
            + b"fmt "
            + b"\x10\x00\x00\x00"  # Subchunk1Size
            + b"\x01\x00"  # AudioFormat (PCM)
            + b"\x01\x00"  # NumChannels (mono)
            + b"\x44\xac\x00\x00"  # SampleRate (44100)
            + b"\x88\x58\x01\x00"  # ByteRate
            + b"\x02\x00"  # BlockAlign
            + b"\x10\x00"  # BitsPerSample
            + b"data"
            + b"\x00\xf0\x00\x00"  # Subchunk2Size
        )

        # Add silence data (44100 samples at 16-bit = 88200 bytes for 1 second)
        silence = b"\x00" * 88200

        return wav_header + silence


class OpenAITTSService(TTSService):
    """OpenAI TTS service."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "alloy",
    ) -> None:
        """
        Initialize OpenAI TTS service.

        Args:
            api_key: OpenAI API key
            model: TTS model name
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
        """
        self.model = model
        self.voice = voice

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.client = AsyncOpenAI(api_key=api_key)

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text using OpenAI TTS."""
        try:
            response = await self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="wav",
            )

            # Read the audio content
            audio_data = io.BytesIO()
            async for chunk in response:
                audio_data.write(chunk)

            return audio_data.getvalue()

        except Exception as e:
            logger.error("openai_tts_failed", error=str(e))
            raise
