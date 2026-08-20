"""
API routes for the Voice RAG pipeline.

Endpoints:
  POST /api/query/voice  — Voice query (audio upload)
  POST /api/query/text   — Text query (for testing)
  GET  /api/benchmark    — Return latest benchmark results
  GET  /api/stats        — Vector DB statistics
"""

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from src.harness.models import PipelineResult

router = APIRouter()


class TextQueryRequest(BaseModel):
    """Request body for text-based queries."""

    query: str = Field(..., min_length=1, max_length=500, description="The question to ask")


class TextQueryResponse(BaseModel):
    """Response for text-based queries."""

    answer: str
    query: str
    success: bool
    refused: bool = False
    refusal_reason: str = ""
    refusal_message: str = ""
    model_used: str = ""
    trace_id: str = ""
    latency_ms: float = 0.0
    latency_breakdown: dict = {}
    retrieved_chunks: list[dict] = []


@router.post("/query/text", response_model=TextQueryResponse)
async def text_query(request: Request, body: TextQueryRequest):
    """
    Process a text query through the RAG pipeline.

    This endpoint bypasses STT and goes directly to retrieval + generation.
    Useful for testing and benchmarking without microphone input.
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result: PipelineResult = await pipeline.process_text(body.query)

    return TextQueryResponse(
        answer=result.answer,
        query=result.query,
        success=result.success,
        refused=result.refused,
        refusal_reason=result.refusal_reason.value if result.refused else "",
        refusal_message=result.refusal_message,
        model_used=result.model_used,
        trace_id=result.trace_id,
        latency_ms=result.latency.total_ms,
        latency_breakdown={
            "embedding_ms": result.latency.embedding_ms,
            "retrieval_ms": result.latency.retrieval_ms,
            "guardrail_pre_ms": result.latency.guardrail_pre_ms,
            "generation_ms": result.latency.generation_ms,
            "guardrail_post_ms": result.latency.guardrail_post_ms,
            "total_ms": result.latency.total_ms,
        },
        retrieved_chunks=result.retrieved_chunks,
    )


@router.post("/query/voice", response_model=TextQueryResponse)
async def voice_query(request: Request, audio: UploadFile = File(...)):
    """
    Process a voice query through the full RAG pipeline.

    Accepts audio file (WAV format preferred) and runs the complete pipeline:
    Audio → STT → Retrieval → Guardrails → Generation → Answer
    """
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Read audio bytes
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    result: PipelineResult = await pipeline.process_voice(audio_bytes)

    return TextQueryResponse(
        answer=result.answer,
        query=result.transcript or result.query,
        success=result.success,
        refused=result.refused,
        refusal_reason=result.refusal_reason.value if result.refused else "",
        refusal_message=result.refusal_message,
        model_used=result.model_used,
        trace_id=result.trace_id,
        latency_ms=result.latency.total_ms,
        latency_breakdown={
            "stt_ms": result.latency.stt_ms,
            "embedding_ms": result.latency.embedding_ms,
            "retrieval_ms": result.latency.retrieval_ms,
            "guardrail_pre_ms": result.latency.guardrail_pre_ms,
            "generation_ms": result.latency.generation_ms,
            "guardrail_post_ms": result.latency.guardrail_post_ms,
            "total_ms": result.latency.total_ms,
            "e2e_ms": result.latency.e2e_ms or round(result.latency.stt_ms + result.latency.total_ms, 2),
        },
        retrieved_chunks=result.retrieved_chunks,
    )


@router.get("/stats")
async def get_stats(request: Request):
    """Get vector database statistics."""
    store = request.app.state.store
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialized")

    try:
        stats = store.get_stats()
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}
