import unittest
from unittest.mock import MagicMock
from typing import List, Dict, Any

from src.retrieval.vector_search import VectorRetriever
from src.retrieval.bm25_search import BM25Retriever
from src.retrieval.hybrid_search import HybridRetriever
from src.indexing.vector_store import BaseVectorStore
from src.indexing.bm25_index import BM25Index


class TestVectorRetriever(unittest.TestCase):
    def test_retrieve_calls_vector_store(self):
        mock_store = MagicMock(spec=BaseVectorStore)
        expected_results = [
            {"id": "c1", "text": "text 1", "metadata": {}, "score": 0.9}
        ]
        mock_store.search.return_value = expected_results

        retriever = VectorRetriever(vector_store=mock_store)
        results = retriever.retrieve("test query", k=5)

        mock_store.search.assert_called_once_with("test query", k=5)
        self.assertEqual(results, expected_results)


class TestBM25Retriever(unittest.TestCase):
    def test_retrieve_calls_bm25_index(self):
        mock_index = MagicMock(spec=BM25Index)
        expected_results = [
            {"id": "c2", "text": "text 2", "metadata": {}, "score": 12.5}
        ]
        mock_index.search.return_value = expected_results

        retriever = BM25Retriever(bm25_index=mock_index)
        results = retriever.retrieve("test query", k=5)

        mock_index.search.assert_called_once_with("test query", k=5)
        self.assertEqual(results, expected_results)


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        # Definiamo retriever mockati
        self.mock_vector = MagicMock(spec=VectorRetriever)
        self.mock_bm25 = MagicMock(spec=BM25Retriever)

        # Setup dei risultati fittizi:
        # Vettoriale: doc1 (rank 1), doc2 (rank 2)
        self.vector_results = [
            {"id": "doc1", "text": "Vettore doc 1", "metadata": {"tag": "v1"}},
            {"id": "doc2", "text": "Comune doc 2", "metadata": {"tag": "shared"}},
        ]
        # BM25: doc2 (rank 1), doc3 (rank 2)
        self.bm25_results = [
            {"id": "doc2", "text": "Comune doc 2", "metadata": {"tag": "shared"}},
            {"id": "doc3", "text": "BM25 doc 3", "metadata": {"tag": "b3"}},
        ]

        self.mock_vector.retrieve.return_value = self.vector_results
        self.mock_bm25.retrieve.return_value = self.bm25_results

    def test_hybrid_retrieve_calls_underlying_retrievers(self):
        retriever = HybridRetriever(
            vector_retriever=self.mock_vector,
            bm25_retriever=self.mock_bm25,
            alpha=0.5,
            rrf_k=60
        )
        _ = retriever.retrieve("query di prova", k=10)

        # Dovrebbe richiedere più candidati per la fusione (default max(k*2, 50) = 50)
        self.mock_vector.retrieve.assert_called_once_with("query di prova", k=50)
        self.mock_bm25.retrieve.assert_called_once_with("query di prova", k=50)

    def test_rrf_scoring_balanced_alpha(self):
        # RRF_K = 60, alpha = 0.5
        # doc1: rank_vector = 1, rank_bm25 = None
        #   score = 0.5 * (1 / 61) + 0.5 * 0 = 0.5 / 61 = 0.0081967
        # doc2: rank_vector = 2, rank_bm25 = 1
        #   score = 0.5 * (1 / 62) + 0.5 * (1 / 61) = 0.0080645 + 0.0081967 = 0.0162612
        # doc3: rank_vector = None, rank_bm25 = 2
        #   score = 0.5 * 0 + 0.5 * (1 / 62) = 0.5 / 62 = 0.0080645
        #
        # Ordine atteso: doc2, doc1, doc3

        retriever = HybridRetriever(
            vector_retriever=self.mock_vector,
            bm25_retriever=self.mock_bm25,
            alpha=0.5,
            rrf_k=60
        )
        results = retriever.retrieve("query", k=3)

        self.assertEqual(len(results), 3)

        # doc2 in prima posizione
        self.assertEqual(results[0]["id"], "doc2")
        self.assertAlmostEqual(results[0]["score"], 0.5 / 62 + 0.5 / 61)

        # doc1 in seconda posizione
        self.assertEqual(results[1]["id"], "doc1")
        self.assertAlmostEqual(results[1]["score"], 0.5 / 61)

        # doc3 in terza posizione
        self.assertEqual(results[2]["id"], "doc3")
        self.assertAlmostEqual(results[2]["score"], 0.5 / 62)

    def test_rrf_scoring_pure_vector_alpha(self):
        # alpha = 1.0 (solo vettoriale)
        # doc1: score = 1.0 * (1 / 61) = 0.01639
        # doc2: score = 1.0 * (1 / 62) = 0.01612
        # doc3: score = 0.0
        # Ordine atteso: doc1, doc2 (doc3 ha score 0.0 e sta dopo, o viene troncato se k limitato)

        retriever = HybridRetriever(
            vector_retriever=self.mock_vector,
            bm25_retriever=self.mock_bm25,
            alpha=1.0,
            rrf_k=60
        )
        results = retriever.retrieve("query", k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc1")
        self.assertEqual(results[1]["id"], "doc2")

    def test_rrf_scoring_pure_bm25_alpha(self):
        # alpha = 0.0 (solo BM25)
        # doc2: score = 1.0 * (1 / 61)
        # doc3: score = 1.0 * (1 / 62)
        # doc1: score = 0.0
        # Ordine atteso: doc2, doc3

        retriever = HybridRetriever(
            vector_retriever=self.mock_vector,
            bm25_retriever=self.mock_bm25,
            alpha=0.0,
            rrf_k=60
        )
        results = retriever.retrieve("query", k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc2")
        self.assertEqual(results[1]["id"], "doc3")

    def test_alpha_validation(self):
        retriever = HybridRetriever(
            vector_retriever=self.mock_vector,
            bm25_retriever=self.mock_bm25,
            alpha=0.5
        )
        # Valori validi non dovrebbero sollevare eccezioni
        retriever.retrieve("query", alpha=0.0)
        retriever.retrieve("query", alpha=1.0)
        retriever.retrieve("query", alpha=0.75)

        # Valori non validi devono sollevare ValueError
        with self.assertRaises(ValueError):
            retriever.retrieve("query", alpha=-0.1)
        with self.assertRaises(ValueError):
            retriever.retrieve("query", alpha=1.1)


if __name__ == "__main__":
    unittest.main()
