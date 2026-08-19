"""Unit tests for the retrieval module."""

import os
import pytest
import numpy as np
from pathlib import Path
from src.retrieval.numpy_store import LocalNumpyStore
from src.retrieval.fast_sparse import FastSparseStore
from src.retrieval.models import RetrievedChunk


class TestLocalNumpyStore:
    """Tests for the local NumPy vector store."""

    @pytest.fixture
    def store_path(self):
        path = "data/test_numpy_store.pkl"
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_empty_store_query(self, store_path):
        store = LocalNumpyStore(storage_path=store_path)
        store.connect()
        result = store.query([0.1, 0.2, 0.3], top_k=2)
        assert len(result.chunks) == 0
        assert result.duration_ms >= 0.0

    def test_upsert_and_query(self, store_path):
        store = LocalNumpyStore(storage_path=store_path)
        store.connect()

        texts = ["apple", "banana", "orange"]
        # 3-dimensional embeddings
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        metadatas = [
            {"type": "fruit", "color": "red"},
            {"type": "fruit", "color": "yellow"},
            {"type": "citrus", "color": "orange"},
        ]

        # Upsert
        count = store.upsert_chunks(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            namespace="fruits",
        )
        assert count == 3
        assert len(store.texts) == 3

        # Query apple (closer to [1.0, 0.0, 0.0])
        result = store.query([0.9, 0.1, 0.0], top_k=1, namespace="fruits")
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "apple"
        assert result.chunks[0].score > 0.8
        assert result.chunks[0].metadata["color"] == "red"

        # Query banana with filter
        result = store.query(
            [0.0, 0.9, 0.1],
            top_k=2,
            namespace="fruits",
            filter_dict={"color": "yellow"},
        )
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "banana"

    def test_delete_namespace(self, store_path):
        store = LocalNumpyStore(storage_path=store_path)
        store.connect()

        store.upsert_chunks(
            texts=["a", "b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            namespace="ns1",
        )
        store.upsert_chunks(
            texts=["c"],
            embeddings=[[1.0, 1.0]],
            namespace="ns2",
        )

        assert len(store.texts) == 3

        # Delete ns1
        store.delete_namespace("ns1")
        assert len(store.texts) == 1
        assert store.metadatas[0]["namespace"] == "ns2"

    def test_bm25_searcher(self):
        from src.retrieval.bm25 import BM25Searcher
        texts = [
            "The quick brown fox",
            "कॉर्पोरेशन क्या है?",
            "Apple banana orange",
            "Some other random text content",
            "Mango strawberry cherry fruits",
            "Pineapple grapes melon fruits"
        ]
        metadatas = [
            {"namespace": "default"},
            {"namespace": "default"},
            {"namespace": "fruits"},
            {"namespace": "default"},
            {"namespace": "fruits"},
            {"namespace": "fruits"},
        ]
        
        searcher = BM25Searcher(texts, metadatas)
        
        # Test default namespace query
        results = searcher.query("brown fox", top_k=1, namespace="default")
        assert len(results) == 1
        assert results[0][0] == 0  # original index of first text
        assert results[0][1] > 0.0

        # Test Hindi query
        results = searcher.query("कॉर्पोरेशन", top_k=1, namespace="default")
        print("DEBUG HINDI QUERY:", results)
        assert len(results) == 1
        assert results[0][0] == 1  # original index of second text

        # Test namespace filtering
        results = searcher.query("apple", top_k=1, namespace="fruits")
        assert len(results) == 1
        assert results[0][0] == 2

    def test_hybrid_query(self, store_path):
        store = LocalNumpyStore(storage_path=store_path)
        store.connect()

        # Ingest documents
        texts = [
            "What is a corporation?",
            "Apple is a technology company",
            "Banana and orange are fruits"
        ]
        embeddings = [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ]
        metadatas = [
            {"namespace": "default", "doc_id": "1"},
            {"namespace": "default", "doc_id": "2"},
            {"namespace": "default", "doc_id": "3"},
        ]
        store.upsert_chunks(texts, embeddings, metadatas, namespace="default")

        # Query using hybrid mode (provide both embedding and query_str)
        result = store.query(
            query_embedding=[0.9, 0.1],
            query_str="corporation",
            top_k=1,
            namespace="default"
        )
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "What is a corporation?"
        assert "retrieval_mode" in result.chunks[0].metadata
        assert result.chunks[0].metadata["retrieval_mode"] == "hybrid_rrf"


class TestFastSparseStore:
    def test_indexes_only_selected_chunks(self, tmp_path):
        store = LocalNumpyStore(storage_path=str(tmp_path / "fast_store.pkl"))
        store.upsert_chunks(
            texts=["selected corporation answer", "another selected passage", "third selected item", "other text"],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]],
            metadatas=[{"is_selected": 1}, {"is_selected": 1}, {"is_selected": 1}, {"is_selected": 0}],
            namespace="fixed",
        )

        fast_store = FastSparseStore(store)
        result = fast_store.query("corporation", top_k=1)

        assert fast_store.vector_count == 3
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "selected corporation answer"
        assert result.chunks[0].metadata["retrieval_mode"] == "fast_bm25"

