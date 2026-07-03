"""
Test unitari per verificare l'integrazione e il corretto calcolo delle metriche IR tramite la libreria ranx.
"""

import unittest
from ranx import Qrels, Run, evaluate


class TestMetrics(unittest.TestCase):
    """
    Classe di test per la suite di metriche.
    """

    def setUp(self):
        # Definiamo un semplice caso di test con valori noti
        # Query 1: doc_A è rilevante
        # Query 2: doc_B e doc_C sono rilevanti
        self.qrels_dict = {
            "q1": {"doc_A": 1},
            "q2": {"doc_B": 1, "doc_C": 1}
        }
        self.qrels = Qrels(self.qrels_dict)

    def test_ranx_perfect_retrieval(self):
        """
        Verifica che un retrieval perfetto produca metriche pari a 1.0.
        """
        # I primi risultati sono esattamente quelli rilevanti
        run_dict = {
            "q1": {"doc_A": 0.9, "doc_X": 0.1},
            "q2": {"doc_B": 0.9, "doc_C": 0.8, "doc_Y": 0.1}
        }
        run = Run(run_dict)

        results = evaluate(
            self.qrels,
            run,
            ["precision@1", "recall@2", "ndcg@2", "mrr", "map@2"]
        )

        self.assertAlmostEqual(results["precision@1"], 1.0)
        self.assertAlmostEqual(results["recall@2"], 1.0)
        self.assertAlmostEqual(results["ndcg@2"], 1.0)
        self.assertAlmostEqual(results["mrr"], 1.0)
        self.assertAlmostEqual(results["map@2"], 1.0)

    def test_ranx_imperfect_retrieval(self):
        """
        Verifica il calcolo di precision, recall e mrr con risultati imperfetti.
        """
        # Per q1: doc_A è al secondo posto (rank 2)
        # Per q2: doc_B è al primo posto (rank 1), doc_C non è nei top-3
        run_dict = {
            "q1": {"doc_X": 0.9, "doc_A": 0.8, "doc_Y": 0.1},
            "q2": {"doc_B": 0.9, "doc_Z": 0.8, "doc_W": 0.7}
        }
        run = Run(run_dict)

        results = evaluate(
            self.qrels,
            run,
            ["precision@1", "precision@2", "recall@1", "recall@2", "mrr", "map@2"]
        )

        # q1 precision@1 = 0, q2 precision@1 = 1 -> avg = 0.5
        self.assertAlmostEqual(results["precision@1"], 0.5)
        
        # q1 precision@2 = 0.5, q2 precision@2 = 0.5 -> avg = 0.5
        self.assertAlmostEqual(results["precision@2"], 0.5)
        
        # q1 recall@1 = 0.0, q2 recall@1 = 0.5 -> avg = 0.25
        self.assertAlmostEqual(results["recall@1"], 0.25)

        # q1 recall@2 = 1.0, q2 recall@2 = 0.5 -> avg = 0.75
        self.assertAlmostEqual(results["recall@2"], 0.75)

        # q1 Reciprocal Rank = 1/2 = 0.5
        # q2 Reciprocal Rank = 1/1 = 1.0
        # MRR = (0.5 + 1.0) / 2 = 0.75
        self.assertAlmostEqual(results["mrr"], 0.75)


if __name__ == "__main__":
    unittest.main()
