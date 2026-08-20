"""
FastAPI application — main entry point.

Configures CORS, static files, startup/shutdown lifecycle,
and mounts all API routes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import structlog
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import mimetypes
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/html", ".html")

from src.config import settings
from src.api.routes import router
from src.stt.elevenlabs_stt import ElevenLabsSTT
from src.embeddings.multilingual import EmbeddingService
from src.retrieval import LocalNumpyStore
from src.retrieval.fast_sparse import FastSparseStore
from src.generation.groq_llm import GroqLLM
from src.generation.openai_llm import OpenAILLM
from src.guardrails.off_topic import OffTopicGuardrail
from src.guardrails.unsafe_input import UnsafeInputGuardrail
from src.guardrails.coverage import CoverageGuardrail
from src.guardrails.grounding import GroundingGuardrail
from src.guardrails.language_consistency import LanguageConsistencyGuardrail
import asyncio
import numpy as np
from src.guardrails.answerability import AnswerabilityGuardrail
from src.harness.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)

# Global pipeline instance & synchronization lock
pipeline: RAGPipeline | None = None
_init_lock = asyncio.Lock()
_init_complete = False


def _sync_init_components(app: FastAPI):
    """Synchronous heavy component loading executed in background worker thread."""
    global pipeline
    logger.info("loading_embedding_model")
    embedding_service = EmbeddingService()
    embedding_service.load_model()
    
    warmup_start = time.perf_counter()
    embedding_service.encode_query("warmup")
    logger.info(
        "embedding_model_warmed",
        duration_ms=round((time.perf_counter() - warmup_start) * 1000, 2),
    )

    stt = ElevenLabsSTT()
    store = LocalNumpyStore()
    store.connect()

    llm_primary = GroqLLM()
    llm_fallback = OpenAILLM() if settings.openai_api_key else None

    # Initialize guardrails
    off_topic = OffTopicGuardrail(embedding_service=embedding_service, threshold=0.10)
    if len(store.embeddings) > 0:
        # Efficiently compute centroid from in-memory indexed vectors without extra parquet loading
        off_topic.set_centroid(np.mean(store.embeddings, axis=0))
        logger.info("off_topic_centroid_set_from_store", vectors=len(store.embeddings))

    unsafe = UnsafeInputGuardrail()
    coverage = CoverageGuardrail(threshold=0.15, semantic_threshold=0.40)
    grounding = GroundingGuardrail(embedding_service=embedding_service, threshold=0.40)
    language_consistency = LanguageConsistencyGuardrail(allow_fallback=False)
    answerability = AnswerabilityGuardrail(min_answerability=0.40)

    # Build pipeline with all guardrails
    pipeline = RAGPipeline(
        stt_provider=stt,
        embedding_service=embedding_service,
        vector_store=store,
        llm_primary=llm_primary,
        llm_fallback=llm_fallback,
        off_topic_guardrail=off_topic,
        unsafe_guardrail=unsafe,
        grounding_guardrail=grounding,
        coverage_guardrail=coverage,
        language_consistency_guardrail=language_consistency,
        answerability_guardrail=answerability,
    )

    # Store in app state
    app.state.pipeline = pipeline
    app.state.embedding_service = embedding_service
    app.state.store = store
    app.state.stt = stt
    app.state.llm_primary = llm_primary
    app.state.llm_fallback = llm_fallback


async def _init_rag_pipeline(app: FastAPI):
    """Asynchronous background initialization of heavy RAG components."""
    global _init_complete
    if _init_complete:
        return
    async with _init_lock:
        if _init_complete:
            return
        await asyncio.to_thread(_sync_init_components, app)
        _init_complete = True
        logger.info("app_ready", pipeline="initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup and shutdown hooks."""
    logger.info("app_starting", env=settings.app_env)
    
    # Initialize empty state placeholders
    app.state.pipeline = None
    app.state.embedding_service = None
    app.state.store = None

    # Kick off background initialization task without blocking the HTTP listener
    init_task = asyncio.create_task(_init_rag_pipeline(app))

    # Yield immediately so Uvicorn can open port and answer /health right away
    yield

    # Shutdown
    logger.info("app_shutting_down")
    if not init_task.done():
        init_task.cancel()
    stt = getattr(app.state, "stt", None)
    if stt:
        await stt.close()
    llm_primary = getattr(app.state, "llm_primary", None)
    if llm_primary:
        await llm_primary.close()
    llm_fallback = getattr(app.state, "llm_fallback", None)
    if llm_fallback:
        await llm_fallback.close()


# Create FastAPI app
app = FastAPI(
    title="HHG Voice RAG",
    description="Voice-Enabled RAG Pipeline for HH Goa 2026 Task #2",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow browser mic access from any origin (demo purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router, prefix="/api")

# Serve static frontend files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return {"message": "HHG Voice RAG API is running. Frontend not found at /static/index.html"}


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "env": settings.app_env,
    }


def main():
    """Entry point for running the server."""
    import uvicorn
    import os

    port = int(os.environ.get("PORT", settings.app_port))
    host = os.environ.get("HOST", settings.app_host)

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
        workers=1,
    )


if __name__ == "__main__":
    main()
