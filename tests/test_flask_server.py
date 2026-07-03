import json
import unittest
from unittest.mock import patch, MagicMock

# Aggiunge la root del progetto al path per gli import
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flask_server.app import app
from src.config import DATASET_NAME

class TestFlaskServer(unittest.TestCase):
    def setUp(self):
        # Configura il client di test di Flask
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        """Verifica che la rotta index restituisca lo stato 200."""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)

    @patch("flask_server.app.retrievers_cache")
    @patch("flask_server.app.INDICES_DIR")
    def test_api_stats_route(self, mock_indices_dir, mock_cache):
        """Verifica che l'endpoint delle statistiche restituisca le info corrette."""
        # Setup mocks
        mock_indices_dir.glob.return_value = [
            Path(f"/mock/dir/bm25_index_{DATASET_NAME}_markdown-two-stage_800.pkl"),
            Path(f"/mock/dir/bm25_index_{DATASET_NAME}_markdown-two-stage_400.pkl"),
        ]
        mock_indices_dir.__truediv__.return_value.exists.return_value = True
        mock_cache.__contains__.side_effect = lambda x: x == (DATASET_NAME, 800)
        mock_cache.__getitem__.return_value = {
            "collection_count": 4427
        }

        response = self.app.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn("default_embedder", data)
        self.assertIn("default_dataset", data)
        self.assertIn("available_datasets", data)
        self.assertIn("available_sizes", data)
        self.assertIn("details", data)
        
        # Verifica dettagli della taglia 800 (caricata)
        ds = data["default_dataset"]
        self.assertTrue(data["details"][ds]["800"]["loaded"])
        self.assertEqual(data["details"][ds]["800"]["count"], 4427)
        # Verifica dettagli della taglia 400 (non caricata ma esistente)
        self.assertFalse(data["details"][ds]["400"]["loaded"])

    @patch("flask_server.app.get_retrievers")
    def test_api_search_missing_query(self, mock_get_retrievers):
        """Verifica che una query vuota ritorni codice 400."""
        response = self.app.post(
            "/api/search",
            data=json.dumps({"query": ""}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    @patch("flask_server.app.get_retrievers")
    def test_api_search_invalid_strategy(self, mock_get_retrievers):
        """Verifica che una strategia non valida ritorni codice 400."""
        response = self.app.post(
            "/api/search",
            data=json.dumps({"query": "vpc subnet", "strategy": "invalid-strategy"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn("error", data)

    @patch("flask_server.app.get_retrievers")
    def test_api_search_success(self, mock_get_retrievers):
        """Verifica una ricerca ibrida con successo."""
        # Configura i mock
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {
                "id": "subnet_0",
                "text": "Subnets are segments of VPC's IP address range",
                "score": 0.85,
                "metadata": {"source_file": "subnets.md", "header_path": "VPC > Subnets"},
                "source_details": {"vector_rank": 1, "bm25_rank": 2}
            }
        ]
        
        mock_get_retrievers.return_value = {
            "hybrid": mock_retriever
        }

        # Esegui chiamata POST
        response = self.app.post(
            "/api/search",
            data=json.dumps({
                "query": "what is a subnet?",
                "chunk_size": 800,
                "strategy": "hybrid-rrf",
                "alpha": 0.6,
                "k": 5
            }),
            content_type="application/json"
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data["query"], "what is a subnet?")
        self.assertEqual(data["chunk_size"], 800)
        self.assertEqual(data["strategy"], "hybrid-rrf")
        self.assertEqual(data["alpha"], 0.6)
        self.assertEqual(data["k"], 5)
        self.assertIn("elapsed_ms", data)
        self.assertEqual(len(data["results"]), 1)
        
        # Verifica i campi del risultato singolo
        result = data["results"][0]
        self.assertEqual(result["id"], "subnet_0")
        self.assertEqual(result["text"], "Subnets are segments of VPC's IP address range")
        self.assertEqual(result["score"], 0.85)
        self.assertEqual(result["metadata"]["source_file"], "subnets.md")
        self.assertEqual(result["source_details"]["vector_rank"], 1)
        self.assertEqual(result["source_details"]["bm25_rank"], 2)
        
        # Verifica che sia stato chiamato l'ibrido corretto
        mock_retriever.retrieve.assert_called_once_with("what is a subnet?", k=5, alpha=0.6)

    @patch("flask_server.app.initialize_reranker")
    @patch("flask_server.app.get_retrievers")
    def test_api_search_with_rerank_success(self, mock_get_retrievers, mock_init_reranker):
        """Verifica una ricerca con Reranker abilitato."""
        # Configura base retriever mock
        mock_base_retriever = MagicMock()
        mock_base_retriever.retrieve.return_value = [
            {
                "id": "subnet_0",
                "text": "Subnets are segments of VPC's IP address range",
                "score": 0.85,
                "metadata": {"source_file": "subnets.md", "header_path": "VPC > Subnets"},
                "source_details": {"vector_rank": 1, "bm25_rank": 2}
            }
        ]
        
        mock_get_retrievers.return_value = {
            "hybrid": mock_base_retriever
        }

        # Configura reranker mock
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"index": 0, "relevance_score": 0.98}
        ]
        mock_init_reranker.return_value = mock_reranker

        # Esegui chiamata POST con use_rerank=True
        response = self.app.post(
            "/api/search",
            data=json.dumps({
                "query": "what is a subnet?",
                "chunk_size": 800,
                "strategy": "hybrid-rrf",
                "alpha": 0.6,
                "k": 5,
                "use_rerank": True
            }),
            content_type="application/json"
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data["query"], "what is a subnet?")
        self.assertEqual(data["use_rerank"], True)
        self.assertEqual(len(data["results"]), 1)
        
        result = data["results"][0]
        self.assertEqual(result["id"], "subnet_0")
        self.assertEqual(result["score"], 0.98) # Punteggio aggiornato dal Reranker
        self.assertEqual(result["source_details"]["base_score"], 0.85)
        self.assertEqual(result["source_details"]["rerank_rank"], 1)
        
        mock_init_reranker.assert_called_once()

    def test_api_example_queries(self):
        """Verifica che l'endpoint delle query di esempio funzioni."""
        response = self.app.get("/api/example_queries?dataset_name=vpc_latest&chunk_size=800")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("queries", data)
        self.assertTrue(isinstance(data["queries"], list))
        self.assertGreater(len(data["queries"]), 0)

    @patch("flask_server.app.load_ground_truth_and_chunks")
    def test_api_search_ground_truth_mock(self, mock_load):
        """Verifica che se la query è nel ground truth, ritorni risultati finti senza chiamare il retriever."""
        # Configura il mock per caricare una query di test
        mock_load.return_value = (
            {"test query": ["chunk_test_123"]},
            {"chunk_test_123": {
                "chunk_id": "chunk_test_123",
                "text": "Questo è un testo di test nel ground truth",
                "source_file": "test.md",
                "chunk_index": 0,
                "strategy": "markdown-two-stage",
                "header_path": "Test > Doc",
                "headers": {"Header 1": "Test"}
            }}
        )

        response = self.app.post(
            "/api/search",
            data=json.dumps({
                "query": "test query?",  # Test normalizzazione (rimozione punto interrogativo e lowercase)
                "chunk_size": 800,
                "strategy": "hybrid-rrf",
                "k": 5
            }),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["is_mocked"], True)
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["id"], "chunk_test_123")
        self.assertEqual(data["results"][0]["text"], "Questo è un testo di test nel ground truth")

    @patch("flask_server.app.load_ground_truth_and_chunks")
    def test_api_search_ground_truth_mock_context_fill(self, mock_load):
        """Verifica che il riempimento di contesto funzioni se ci sono altri chunk dello stesso file."""
        mock_load.return_value = (
            {"test query": ["chunk_test_1"]},
            {
                "chunk_test_1": {
                    "chunk_id": "chunk_test_1",
                    "text": "Target text",
                    "source_file": "doc.md",
                    "chunk_index": 1,
                    "strategy": "markdown-two-stage"
                },
                "chunk_test_0": {
                    "chunk_id": "chunk_test_0",
                    "text": "Previous text",
                    "source_file": "doc.md",
                    "chunk_index": 0,
                    "strategy": "markdown-two-stage"
                },
                "chunk_test_2": {
                    "chunk_id": "chunk_test_2",
                    "text": "Next text",
                    "source_file": "doc.md",
                    "chunk_index": 2,
                    "strategy": "markdown-two-stage"
                }
            }
        )

        response = self.app.post(
            "/api/search",
            data=json.dumps({
                "query": "test query",
                "chunk_size": 800,
                "strategy": "hybrid-rrf",
                "k": 3
            }),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["is_mocked"], True)
        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["results"][0]["id"], "chunk_test_1")
        self.assertEqual(data["results"][1]["id"], "chunk_test_0")
        self.assertEqual(data["results"][2]["id"], "chunk_test_2")

if __name__ == "__main__":
    unittest.main()
