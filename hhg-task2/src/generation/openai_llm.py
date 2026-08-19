"""
OpenAI LLM — Fallback answer generation provider.

Uses GPT-4o-mini as a backup when Groq is rate-limited or unavailable.
"""

import time
import structlog
import httpx

from src.generation.base import BaseLLM
from src.generation.models import GenerationResult
from src.config import settings

logger = structlog.get_logger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful and precise assistant. Answer the user's question ONLY based on the provided context. "
    "If the context does not contain sufficient information to answer, say 'I cannot find sufficient information "
    "to answer that question based on the available data.' Do not make up information or go beyond the context. "
    "Keep your answer concise and factual."
)


class OpenAILLM(BaseLLM):
    """OpenAI GPT-4o-mini client (fallback provider)."""

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 150):
        self._model = model
        self._max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(8.0, connect=2.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def generate(self, query: str, context: str, system_prompt: str = "") -> GenerationResult:
        """Generate an answer using OpenAI GPT-4o-mini."""
        start = time.perf_counter()
        client = await self._get_client()

        system = system_prompt or DEFAULT_SYSTEM_PROMPT
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        try:
            response = await client.post(
                OPENAI_API_URL,
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": self._max_tokens,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
            data = response.json()

            duration_ms = (time.perf_counter() - start) * 1000
            usage = data.get("usage", {})

            result = GenerationResult(
                answer=data["choices"][0]["message"]["content"],
                model=self._model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                duration_ms=round(duration_ms, 2),
                is_fallback=True,
            )

            logger.info(
                "openai_generation_complete",
                model=self._model,
                duration_ms=result.duration_ms,
                tokens=result.completion_tokens,
                is_fallback=True,
            )

            return result

        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("openai_generation_failed", error=str(e), duration_ms=round(duration_ms, 2))
            raise

    async def health_check(self) -> bool:
        """Check if OpenAI API is reachable."""
        try:
            client = await self._get_client()
            response = await client.get("https://api.openai.com/v1/models")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
