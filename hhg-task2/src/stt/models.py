"""
Pydantic models for STT module.
"""

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    """Structured output from speech-to-text transcription."""

    transcript: str = Field(..., description="Transcribed text from audio input")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Transcription confidence score")
    language: str = Field(default="en", description="Detected language code")
    duration_ms: float = Field(default=0.0, ge=0.0, description="Processing duration in milliseconds")

    model_config = {"json_schema_extra": {"examples": [{"transcript": "What is machine learning?", "confidence": 0.95, "language": "en", "duration_ms": 67.3}]}}
