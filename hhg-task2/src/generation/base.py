"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod

from src.generation.models import GenerationResult


class BaseLLM(ABC):
    """Abstract interface for LLM answer generation."""

    @abstractmethod
    async def generate(self, query: str, context: str, system_prompt: str = "") -> GenerationResult:
        """
        Generate an answer grounded in the provided context.

        Args:
            query: The user's question.
            context: Retrieved context chunks concatenated.
            system_prompt: Optional system prompt override.

        Returns:
            GenerationResult with the answer and metrics.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM API is reachable."""
        ...
