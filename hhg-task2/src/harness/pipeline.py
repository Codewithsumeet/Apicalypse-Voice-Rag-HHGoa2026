"""
RAG Pipeline — LangGraph State Machine.

The core orchestration pipeline wiring together:
  transcribe → retrieve → guardrail_check → generate → grounding_check

Uses LangGraph for state machine transitions with conditional edges
for guardrail-triggered refusal paths.
"""

import time
import uuid
import structlog
import httpx

from src.harness.state import RAGState, PipelineStage
from src.harness.models import PipelineResult, LatencyBreakdown
from src.harness.retry import with_retry
from src.guardrails.models import RefusalReason
from src.config import settings
from src.generation.extractive import extractive_answer

logger = structlog.get_logger(__name__)


class RAGPipeline:
    """
    The main RAG pipeline orchestrator.

    Wires together STT, embeddings, retrieval, guardrails, and generation
    into a coherent pipeline with structured I/O and error handling.

    This is implemented as a simple async pipeline (no LangGraph dependency at this stage)
    for minimal overhead. Can be upgraded to LangGraph state machine in Chunk 5.
    """

    def __init__(
        self,
        stt_provider=None,
        embedding_service=None,
        vector_store=None,
        fast_store=None,
        llm_primary=None,
        llm_fallback=None,
        off_topic_guardrail=None,
        unsafe_guardrail=None,
        grounding_guardrail=None,
        coverage_guardrail=None,
    ):
        self.stt = stt_provider
        self.embeddings = embedding_service
        self.store = vector_store
        self.fast_store = fast_store
        self.llm_primary = llm_primary
        self.llm_fallback = llm_fallback
        self.off_topic = off_topic_guardrail
        self.unsafe = unsafe_guardrail
        self.grounding = grounding_guardrail
        self.coverage = coverage_guardrail

    async def process_voice(self, audio_bytes: bytes) -> PipelineResult:
        """
        Full voice-to-answer pipeline.

        Audio → STT → Query → Retrieval → Guardrails → Generation → Answer
        """
        trace_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        latency = LatencyBreakdown()

        try:
            # Stage 1: Transcribe
            stt_result = await with_retry(
                self.stt.transcribe,
                audio_bytes,
                max_retries=1,
                retryable_exceptions=(httpx.TransportError,),
            )
            latency.stt_ms = stt_result.duration_ms
            transcript = stt_result.transcript

            if not transcript.strip():
                return PipelineResult(
                    success=False,
                    refused=True,
                    refusal_reason=RefusalReason.SYSTEM_ERROR,
                    refusal_message="Could not understand the audio. Please try speaking again clearly.",
                    trace_id=trace_id,
                    latency=latency,
                )

            # Continue with text pipeline
            result = await self._process_text(transcript, latency, trace_id)
            result.transcript = transcript
            return result

        except Exception as e:
            total_ms = (time.perf_counter() - start) * 1000
            latency.total_ms = round(total_ms, 2)
            logger.error("pipeline_error", trace_id=trace_id, error=str(e))
            return PipelineResult(
                success=False,
                refused=True,
                refusal_reason=RefusalReason.SYSTEM_ERROR,
                refusal_message="An internal error occurred. Please try again.",
                trace_id=trace_id,
                latency=latency,
            )

    async def process_text(self, query: str) -> PipelineResult:
        """
        Text-only pipeline (bypasses STT).

        Query → Retrieval → Guardrails → Generation → Answer
        """
        trace_id = str(uuid.uuid4())[:8]
        latency = LatencyBreakdown()
        return await self._process_text(query, latency, trace_id)

    async def _process_text(
        self, query: str, latency: LatencyBreakdown, trace_id: str
    ) -> PipelineResult:
        """Internal text processing pipeline."""
        pipeline_start = time.perf_counter()

        try:
            # Stage: Pre-generation guardrails (unsafe input)
            pre_start = time.perf_counter()
            if self.unsafe:
                unsafe_result = self.unsafe.check(query)
                if not unsafe_result.passed:
                    latency.guardrail_pre_ms = round((time.perf_counter() - pre_start) * 1000, 2)
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=unsafe_result.reason,
                        refusal_message=unsafe_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

            query_embedding = None
            if settings.answer_mode.strip().lower() != "fast":
                embed_start = time.perf_counter()
                query_embedding = self.embeddings.encode_query(query)
                latency.embedding_ms = round((time.perf_counter() - embed_start) * 1000, 2)

                if self.off_topic:
                    off_topic_result = self.off_topic.check(query_embedding)
                    if not off_topic_result.passed:
                        latency.guardrail_pre_ms = round((time.perf_counter() - pre_start) * 1000, 2)
                        latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                        return PipelineResult(
                            query=query,
                            success=False,
                            refused=True,
                            refusal_reason=off_topic_result.reason,
                            refusal_message=off_topic_result.message,
                            trace_id=trace_id,
                            latency=latency,
                        )

            latency.guardrail_pre_ms = round((time.perf_counter() - pre_start) * 1000, 2)

            # Stage: Retrieve
            if settings.answer_mode.strip().lower() == "fast":
                if self.fast_store is None:
                    from src.retrieval.fast_sparse import FastSparseStore

                    self.fast_store = FastSparseStore(self.store)
                retrieval_result = self.fast_store.query(query, top_k=settings.retrieval_top_k)
            else:
                retrieval_result = self.store.query(
                    query_embedding,
                    query_str=query,
                    namespace=settings.retrieval_namespace,
                )
            retrieval_result.query = query
            latency.retrieval_ms = retrieval_result.duration_ms

            if not retrieval_result.chunks:
                latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                return PipelineResult(
                    query=query,
                    success=False,
                    refused=True,
                    refusal_reason=RefusalReason.UNGROUNDED,
                    refusal_message="No relevant information found in the knowledge base for your question.",
                    trace_id=trace_id,
                    latency=latency,
                )

            context = retrieval_result.context_text

            # Stage: Pre-generation guardrail (context coverage check)
            if self.coverage:
                coverage_result = self.coverage.check(query, context)
                if not coverage_result.passed:
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=coverage_result.reason,
                        refusal_message=coverage_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

            # Stage: Generate
            if settings.answer_mode.strip().lower() == "fast":
                answer = extractive_answer(query, retrieval_result.chunks)
                if not answer:
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=RefusalReason.UNGROUNDED,
                        refusal_message="No grounded source sentence was found for this question.",
                        trace_id=trace_id,
                        latency=latency,
                    )

                # The fast answer is copied from a retrieved source chunk, so
                # exact source membership is the grounding check.
                latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                return PipelineResult(
                    answer=answer,
                    query=query,
                    success=True,
                    trace_id=trace_id,
                    latency=latency,
                    retrieved_chunks=[
                        {"text": c.text, "score": c.score, "doc_id": c.doc_id}
                        for c in retrieval_result.chunks
                    ],
                )

            try:
                gen_result = await with_retry(
                    self.llm_primary.generate, query, context, max_retries=1
                )
            except Exception:
                # Fallback to secondary LLM
                if self.llm_fallback:
                    logger.warning("llm_primary_failed_using_fallback", trace_id=trace_id)
                    gen_result = await self.llm_fallback.generate(query, context)
                else:
                    raise

            latency.generation_ms = gen_result.duration_ms

            # Stage: Grounding check
            post_start = time.perf_counter()
            if self.grounding:
                grounding_result = self.grounding.check(gen_result.answer, context)
                latency.guardrail_post_ms = round((time.perf_counter() - post_start) * 1000, 2)

                if not grounding_result.passed:
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=grounding_result.reason,
                        refusal_message=grounding_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

            # Success
            latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

            result = PipelineResult(
                answer=gen_result.answer,
                query=query,
                success=True,
                model_used=gen_result.model,
                is_fallback=gen_result.is_fallback,
                trace_id=trace_id,
                latency=latency,
                retrieved_chunks=[
                    {"text": c.text, "score": c.score, "doc_id": c.doc_id}
                    for c in retrieval_result.chunks
                ],
            )

            logger.info(
                "pipeline_complete",
                trace_id=trace_id,
                total_ms=latency.total_ms,
                model=gen_result.model,
                chunks_used=len(retrieval_result.chunks),
            )

            return result

        except Exception as e:
            latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
            logger.error("pipeline_text_error", trace_id=trace_id, error=str(e))
            return PipelineResult(
                query=query,
                success=False,
                refused=True,
                refusal_reason=RefusalReason.SYSTEM_ERROR,
                refusal_message="An internal error occurred while processing your question.",
                trace_id=trace_id,
                latency=latency,
            )
