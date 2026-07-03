import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import json

import requests

from src.indexing.embedder import TokenRateLimiter, JinaEmbedder
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import ChromaVectorStore


class TestTokenRateLimiter(unittest.TestCase):
    def test_daily_limit_exceeded(self):
        # Limite giornaliero a 3 richieste
        limiter = TokenRateLimiter(rpm_limit=10, tpm_limit=100, daily_limit=3)
        
        # Effettua 3 registrazioni
        limiter.record_request(10)
        limiter.record_request(10)
        limiter.record_request(10)
        
        # La quarta richiesta deve fallire sollevando RuntimeError
        with self.assertRaises(RuntimeError) as context:
            limiter.check_and_wait(10)
        self.assertIn("LIMITE GIORNALIERO RAGGIUNTO", str(context.exception))

    @patch("time.sleep")
    @patch("time.time")
    def test_rpm_limit_wait(self, mock_time, mock_sleep):
        # Limite a 2 RPM
        limiter = TokenRateLimiter(rpm_limit=2, tpm_limit=100)
        
        current_time = 100.0
        def get_time():
            return current_time
        mock_time.side_effect = get_time
        
        def mock_sleep_effect(seconds):
            nonlocal current_time
            current_time += seconds
        mock_sleep.side_effect = mock_sleep_effect
        
        limiter.record_request(10) # Richiesta 1 al tempo 100.0
        
        current_time = 110.0
        limiter.record_request(10) # Richiesta 2 al tempo 110.0
        
        # Ora cerchiamo di mandare la richiesta 3 (supera limite RPM di 2 in un minuto)
        # Dovrà calcolare l'attesa basata sulla prima richiesta (tempo 100.0)
        # Finestra di 60s da 100.0 scade a 160.0. Attesa stimata = 60.1 - (110.0 - 100.0) = 50.1s
        current_time = 110.0
        limiter.check_and_wait(10)
        
        mock_sleep.assert_called_once()
        # Il valore dell'attesa passato a sleep deve essere circa 50.1
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 50.1)

    @patch("time.sleep")
    @patch("time.time")
    def test_tpm_limit_wait(self, mock_time, mock_sleep):
        # Limite a 100 TPM, 10 RPM
        limiter = TokenRateLimiter(rpm_limit=10, tpm_limit=100)
        
        current_time = 100.0
        def get_time():
            return current_time
        mock_time.side_effect = get_time
        
        def mock_sleep_effect(seconds):
            nonlocal current_time
            current_time += seconds
        mock_sleep.side_effect = mock_sleep_effect
        
        limiter.record_request(60) # 60 token consumati al tempo 100.0
        
        current_time = 120.0
        # Inviamo altri 50 token. 60+50 = 110 > 100 TPM. Dovrà attendere che la prima richiesta scada.
        # Attesa: 60.1 - (120.0 - 100.0) = 40.1s
        limiter.check_and_wait(50)
        
        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 40.1)


class TestJinaEmbedder(unittest.TestCase):
    def setUp(self):
        from src.indexing.embedder import BaseEmbedder
        BaseEmbedder._query_cache.clear()

    @patch("requests.post")
    def test_jina_embed_success(self, mock_post):
        # Mocking della risposta corretta di Jina
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "jina-embeddings-v2-base-en",
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]}
            ],
            "usage": {"total_tokens": 10}
        }
        mock_post.return_value = mock_response
        
        embedder = JinaEmbedder(api_key="mock_key", batch_size=2)
        embeddings = embedder(["text1", "text2"])
        
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(list(embeddings[0]), [0.1, 0.2, 0.3])
        self.assertEqual(list(embeddings[1]), [0.4, 0.5, 0.6])
        mock_post.assert_called_once()

    @patch("requests.post")
    @patch("time.sleep")
    def test_jina_embed_retry_on_429(self, mock_sleep, mock_post):
        # Primo tentativo fallisce con 429, secondo ha successo
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 429
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response_fail)
        
        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.json.return_value = {
            "model": "jina-embeddings-v2-base-en",
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"total_tokens": 5}
        }
        
        mock_post.side_effect = [mock_response_fail, mock_response_ok]
        
        embedder = JinaEmbedder(api_key="mock_key", batch_size=2)
        embeddings = embedder(["text1"])
        
        self.assertEqual([list(emb) for emb in embeddings], [[0.1, 0.2]])
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(2.0) # Primo backoff di default




class TestBM25Index(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            {"chunk_id": "c1", "text": "Amazon VPC allows you to launch AWS resources in a virtual network.", "headers": {"h1": "VPC Introduction"}},
            {"chunk_id": "c2", "text": "Subnets can be public or private, depending on routing configuration.", "headers": {"h1": "Subnets"}},
            {"chunk_id": "c3", "text": "A Route Table contains a set of rules, called routes, that are used to determine where network traffic is directed.", "headers": {"h1": "Route Tables"}}
        ]
        self.index = BM25Index(corpus=self.corpus)

    def test_tokenize(self):
        text = "Hello, world! This is a test."
        tokens = BM25Index.tokenize(text)
        self.assertEqual(tokens, ["hello", "world", "this", "is", "a", "test"])

    def test_search(self):
        # Cerchiamo "virtual network"
        results = self.index.search("virtual network", k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "c1")
        self.assertIn("Amazon VPC allows you", results[0]["text"])
        self.assertEqual(results[0]["metadata"]["h1"], "VPC Introduction")

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "bm25.pkl"
            
            # Salva
            self.index.save(str(file_path))
            self.assertTrue(file_path.exists())
            
            # Carica
            loaded_index = BM25Index.load(str(file_path))
            self.assertEqual(len(loaded_index.corpus), 3)
            
            # Verifica ricerca su indice caricato
            results = loaded_index.search("routing configuration", k=1)
            self.assertEqual(results[0]["id"], "c2")


class TestChromaVectorStore(unittest.TestCase):
    @patch("chromadb.PersistentClient")
    def test_sanitize_metadata(self, mock_client):
        # Mock client initialization
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = ChromaVectorStore(collection_name="test_col", persist_directory="/tmp/fake_chroma")
        
        raw_metadata = {
            "source_file": "file1.md",
            "chunk_index": 5,
            "headers": {"Header 1": "VPC", "Header 2": None},
            "unsupported_list": [1, 2, 3],
            "none_val": None
        }
        
        sanitized = store._sanitize_metadata(raw_metadata)
        
        self.assertEqual(sanitized["source_file"], "file1.md")
        self.assertEqual(sanitized["chunk_index"], 5)
        self.assertEqual(sanitized["headers_Header 1"], "VPC")
        self.assertEqual(sanitized["headers_Header 2"], "")
        self.assertEqual(sanitized["unsupported_list"], "[1, 2, 3]")
        self.assertEqual(sanitized["none_val"], "")


if __name__ == "__main__":
    unittest.main()
