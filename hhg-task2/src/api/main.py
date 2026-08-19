"""
FastAPI application — main entry point.

Configures CORS, static files, startup/shutdown lifecycle,
and mounts all API routes.
"""

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
from src.harness.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)

# Global pipeline instance (initialized at startup)
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup and shutdown hooks."""
    global pipeline

    logger.info("app_starting", env=settings.app_env)

    # Initialize all services
    stt = ElevenLabsSTT()
    embedding_service = EmbeddingService()

    # Load embedding model (heavy — do once at startup)
    logger.info("loading_embedding_model")
    embedding_service.load_model()
    warmup_start = time.perf_counter()
    embedding_service.encode_query("warmup")
    logger.info(
        "embedding_model_warmed",
        duration_ms=round((time.perf_counter() - warmup_start) * 1000, 2),
    )

    # Connect to the resident local index; no cloud vector service is required.
    store = LocalNumpyStore()
    store.connect()
    fast_store = FastSparseStore(store)
    logger.info("fast_sparse_ready", vectors=fast_store.vector_count)

    # Initialize LLMs
    llm_primary = GroqLLM()
    llm_fallback = OpenAILLM() if settings.openai_api_key else None

    # Initialize guardrails
    off_topic = OffTopicGuardrail(embedding_service=embedding_service)
    unsafe = UnsafeInputGuardrail()
    coverage = CoverageGuardrail(threshold=0.15)
    grounding = GroundingGuardrail(embedding_service=embedding_service)

    # Compute dataset centroid for off-topic guardrail at startup
    try:
        import pandas as pd
        data_path = Path("data/msmarco_xi_train.parquet")
        if data_path.exists():
            df = pd.read_parquet(data_path)
            sample_queries = df["query"].dropna().head(100).tolist()
            if sample_queries:
                sample_embeddings = [embedding_service.encode_query(q) for q in sample_queries]
                off_topic.compute_centroid(sample_embeddings)
                logger.info("computed_off_topic_centroid", samples=len(sample_queries))
    except Exception as e:
        logger.warning("failed_to_compute_centroid", error=str(e))

    # Build pipeline
    pipeline = RAGPipeline(
        stt_provider=stt,
        embedding_service=embedding_service,
        vector_store=store,
        fast_store=fast_store,
        llm_primary=llm_primary,
        llm_fallback=llm_fallback,
        off_topic_guardrail=off_topic,
        unsafe_guardrail=unsafe,
        grounding_guardrail=grounding,
        coverage_guardrail=coverage,
    )

    # Store pipeline in app state for route access
    app.state.pipeline = pipeline
    app.state.embedding_service = embedding_service
    app.state.store = store

    logger.info("app_ready", pipeline="initialized")

    yield

    # Shutdown
    logger.info("app_shutting_down")
    await stt.close()
    await llm_primary.close()
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

    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
