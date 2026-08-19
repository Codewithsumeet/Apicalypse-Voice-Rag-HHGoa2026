"""Unit tests for chunking strategies."""

import pytest
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.factory import get_chunker, list_strategies


class TestFixedSizeChunker:
    """Tests for fixed-size chunking strategy."""

    def test_empty_text(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_short_text(self):
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk("Short text.", doc_id="doc1")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].source_doc_id == "doc1"

    def test_long_text_splits(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a long text. " * 10
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("Some text", metadata={"source": "test"})
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test"
        assert chunks[0].metadata["strategy"].startswith("fixed_size")

    def test_strategy_name(self):
        chunker = FixedSizeChunker(chunk_size=256, chunk_overlap=50)
        assert chunker.strategy_name == "fixed_size_256_overlap_50"


class TestMetadataAwareChunker:
    """Tests for metadata-aware chunking strategy."""

    def test_preserves_query(self):
        chunker = MetadataAwareChunker()
        chunks = chunker.chunk("Some passage text", metadata={"query": "test query"})
        assert len(chunks) >= 1
        assert chunks[0].metadata["original_query"] == "test query"

    def test_chunk_from_record(self):
        chunker = MetadataAwareChunker()
        record = {"passage": "Some text", "query": "What is this?", "docid": "d1"}
        chunks = chunker.chunk_from_record(record)
        assert len(chunks) >= 1
        assert chunks[0].metadata["original_query"] == "What is this?"

    def test_strategy_name(self):
        chunker = MetadataAwareChunker()
        assert chunker.strategy_name == "metadata_aware_query_passage"


class TestChunkerFactory:
    """Tests for the chunker factory."""

    def test_get_fixed(self):
        chunker = get_chunker("fixed")
        assert isinstance(chunker, FixedSizeChunker)

    def test_get_metadata_aware(self):
        chunker = get_chunker("metadata_aware")
        assert isinstance(chunker, MetadataAwareChunker)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent")

    def test_list_strategies(self):
        strategies = list_strategies()
        assert "fixed" in strategies
        assert "semantic" in strategies
        assert "metadata_aware" in strategies
