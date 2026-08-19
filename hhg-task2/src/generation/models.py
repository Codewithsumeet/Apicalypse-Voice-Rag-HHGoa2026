"""
Pydantic models for the generation module.
"""

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    """Structured output from LLM answer generation."""

    answer: str = Field(..., description="The generated answer text")
    model: str = Field(default="", description="Model name used for generation")
    prompt_tokens: int = Field(default=0, description="Number of input tokens")
    completion_tokens: int = Field(default=0, description="Number of output tokens")
    duration_ms: float = Field(default=0.0, description="Generation latency in milliseconds")
    is_fallback: bool = Field(default=False, description="Whether a fallback model was used")
