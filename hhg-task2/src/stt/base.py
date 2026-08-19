"""
Abstract base class for Speech-to-Text providers.

All STT implementations must inherit from BaseSTT and implement `transcribe()`.
"""

from abc import ABC, abstractmethod

from src.stt.models import TranscriptionResult


class BaseSTT(ABC):
    """Abstract interface for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WAV/PCM, 16kHz, 16-bit mono preferred).
            language: Language code hint (e.g., 'en', 'hi', 'auto' for auto-detect).

        Returns:
            TranscriptionResult with transcript, confidence, language, and latency.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the STT provider is reachable and operational."""
        ...
