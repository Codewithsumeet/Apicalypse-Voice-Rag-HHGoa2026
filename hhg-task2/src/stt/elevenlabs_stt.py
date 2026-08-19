"""
ElevenLabs Scribe — Speech-to-Text implementation.

Uses ElevenLabs Speech-to-Text API with async HTTP client and connection pooling
for minimal latency overhead.
"""

import time
import structlog
import httpx

from src.stt.base import BaseSTT
from src.stt.models import TranscriptionResult
from src.config import settings

logger = structlog.get_logger(__name__)

# ElevenLabs STT API endpoint
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsSTT(BaseSTT):
    """ElevenLabs Scribe STT provider with connection pooling."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the persistent async HTTP client (connection pooling)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(5.0, connect=2.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                headers={"xi-api-key": settings.elevenlabs_api_key},
            )
        return self._client

    async def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        """
        Transcribe audio using ElevenLabs Scribe API.

        Args:
            audio_bytes: Raw audio data in WAV format.
            language: Language code ('en', 'hin', or 'auto').

        Returns:
            TranscriptionResult with transcript and latency metrics.
        """
        start_time = time.perf_counter()
        client = await self._get_client()

        try:
            # Build multipart form data
            files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
            data = {"model_id": "scribe_v2"}

            if language and language != "auto":
                data["language_code"] = language

            response = await client.post(ELEVENLABS_STT_URL, files=files, data=data)
            response.raise_for_status()
            result = response.json()

            duration_ms = (time.perf_counter() - start_time) * 1000

            transcription = TranscriptionResult(
                transcript=result.get("text", ""),
                confidence=result.get("confidence", 0.0),
                language=result.get("language_code", language),
                duration_ms=round(duration_ms, 2),
            )

            logger.info(
                "stt_transcription_complete",
                transcript_length=len(transcription.transcript),
                duration_ms=transcription.duration_ms,
                language=transcription.language,
            )

            return transcription

        except httpx.HTTPStatusError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error("stt_api_error", status_code=e.response.status_code, duration_ms=round(duration_ms, 2))
            raise
        except httpx.TimeoutException:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error("stt_timeout", duration_ms=round(duration_ms, 2))
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error("stt_unexpected_error", error=str(e), duration_ms=round(duration_ms, 2))
            raise

    async def health_check(self) -> bool:
        """Verify ElevenLabs API is reachable."""
        try:
            client = await self._get_client()
            response = await client.get("https://api.elevenlabs.io/v1/models")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
