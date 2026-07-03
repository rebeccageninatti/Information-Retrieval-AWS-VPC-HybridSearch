import unittest
from unittest.mock import MagicMock, patch
import requests

from src.retrieval.reranker import JinaReranker, RerankRetriever
from src.retrieval.base import BaseRetriever
from src.indexing.embedder import BaseEmbedder


class TestJinaReranker(unittest.TestCase):
    def setUp(self):
        self.api_key = "fake_jina_key"
        self.documents = ["doc content 1", "doc content 2", "doc content 3"]
        self.query = "test query"

    @patch("requests.post")
    def test_rerank_success(self, mock_post):
        # Configura la risposta mock di Jina Rerank API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "jina-reranker-v3",
            "usage": {"total_tokens": 120},
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.85},
                {"index": 2, "relevance_score": 0.70}
            ]
        }
        mock_post.return_value = mock_response

        # Instanzia e testa
        reranker = JinaReranker(api_key=self.api_key)
        results = reranker.rerank(self.query, self.documents)

        # Verifica chiamata API
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://api.jina.ai/v1/rerank")
        self.assertEqual(kwargs["json"]["query"], self.query)
        self.assertEqual(kwargs["json"]["documents"], self.documents)
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.api_key}")

        # Verifica formato ed ordinamento output
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["index"], 1)
        self.assertEqual(results[0]["relevance_score"], 0.95)
        self.assertEqual(results[1]["index"], 0)
        self.assertEqual(results[2]["index"], 2)

    @patch("requests.post")
    @patch("time.sleep")
    def test_rerank_retry_on_429(self, mock_sleep, mock_post):
        # Prima chiamata restituisce 429, seconda chiamata successo
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_429)

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            "model": "jina-reranker-v3",
            "usage": {"total_tokens": 100},
            "results": [{"index": 0, "relevance_score": 0.9}]
        }

        mock_post.side_effect = [mock_response_429, mock_response_200]

        reranker = JinaReranker(api_key=self.api_key)
        results = reranker.rerank(self.query, ["doc 1"], max_retries=3, initial_backoff=1.0)

        # Dovrebbe aver fatto 2 chiamate totali ed 1 sleep di backoff
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["relevance_score"], 0.9)


class TestRerankRetriever(unittest.TestCase):
    def test_rerank_retriever_retrieve(self):
        # Mock del retriever di base
        mock_base_retriever = MagicMock(spec=BaseRetriever)
        base_results = [
            {"id": "doc1", "text": "testo doc 1", "metadata": {"tag": "v1"}, "score": 0.8},
            {"id": "doc2", "text": "testo doc 2", "metadata": {"tag": "v2"}, "score": 0.7},
            {"id": "doc3", "text": "testo doc 3", "metadata": {"tag": "v3"}, "score": 0.6}
        ]
        mock_base_retriever.retrieve.return_value = base_results

        # Mock del reranker
        mock_reranker = MagicMock(spec=JinaReranker)
        rerank_api_results = [
            {"index": 1, "relevance_score": 0.99},  # doc2 diventa primo
            {"index": 0, "relevance_score": 0.88},  # doc1 secondo
            {"index": 2, "relevance_score": 0.11}   # doc3 terzo
        ]
        mock_reranker.rerank.return_value = rerank_api_results

        # Inizializzazione RerankRetriever
        rerank_retriever = RerankRetriever(
            base_retriever=mock_base_retriever,
            reranker=mock_reranker,
            k_candidate=3
        )

        results = rerank_retriever.retrieve("my query", k=2)

        # Verifica recupero candidati dal retriever base
        mock_base_retriever.retrieve.assert_called_once_with("my query", k=3)

        # Verifica chiamata al reranker
        mock_reranker.rerank.assert_called_once_with("my query", ["testo doc 1", "testo doc 2", "testo doc 3"])

        # Verifica ordinamento e formattazione finale dei top K=2
        self.assertEqual(len(results), 2)
        
        # Primo elemento deve essere doc2 con punteggio aggiornato
        self.assertEqual(results[0]["id"], "doc2")
        self.assertEqual(results[0]["text"], "testo doc 2")
        self.assertEqual(results[0]["score"], 0.99)
        self.assertEqual(results[0]["source_details"]["base_score"], 0.7)
        self.assertEqual(results[0]["source_details"]["rerank_rank"], 1)

        # Secondo elemento deve essere doc1 con punteggio aggiornato
        self.assertEqual(results[1]["id"], "doc1")
        self.assertEqual(results[1]["text"], "testo doc 1")
        self.assertEqual(results[1]["score"], 0.88)
        self.assertEqual(results[1]["source_details"]["base_score"], 0.8)
        self.assertEqual(results[1]["source_details"]["rerank_rank"], 2)


if __name__ == "__main__":
    unittest.main()
