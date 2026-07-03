"""
Modulo per l'esecuzione di benchmark comparativi delle strategie di retrieval.
Carica il ground truth, interroga i retrievers, e calcola le metriche IR tramite ranx.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from ranx import Qrels, Run, evaluate

from src.config import (
    GROUND_TRUTH_FILE,
    EVAL_RESULTS_FILE,
    DEFAULT_EMBEDDER,
    DEFAULT_CHUNK_STRATEGY,
    CHUNK_SIZE_DEFAULT,
    CHROMA_COLLECTION_NAME,
    BM25_INDEX_FILE,
    DATASET_NAME
)
from src.indexing.vector_store import ChromaVectorStore
from src.indexing.bm25_index import BM25Index
from src.indexing.embedder import get_embedder, BaseEmbedder
from src.retrieval.vector_search import VectorRetriever
from src.retrieval.bm25_search import BM25Retriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import JinaReranker, RerankRetriever
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """
    Classe che coordina la suite di test ed esegue la valutazione con ranx.
    """

    def __init__(self, is_fast_mode: bool = False):
        self.is_fast_mode = is_fast_mode
        # Definisce il percorso del file dei risultati (standard o fast) con suffisso _rerank
        if self.is_fast_mode:
            self.results_file = EVAL_RESULTS_FILE.with_name(EVAL_RESULTS_FILE.stem + "_rerank_fast.json")
        else:
            self.results_file = EVAL_RESULTS_FILE.with_name(EVAL_RESULTS_FILE.stem + "_rerank.json")

        # 1. Carica il Ground Truth
        self.ground_truth = self.load_ground_truth()
        
        # 2. Carica gli indici
        logger.info("Caricamento degli indici ed inizializzazione dei retrievers...")
        embedder = get_embedder()
        
        # ChromaDB
        self.vector_store = ChromaVectorStore(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedder
        )
        self.vector_retriever = VectorRetriever(self.vector_store)
        
        # BM25
        if not BM25_INDEX_FILE.exists():
            raise FileNotFoundError(f"Indice BM25 non trovato su: {BM25_INDEX_FILE}. Esegui prima run_indexing.py.")
        self.bm25_index = BM25Index.load(str(BM25_INDEX_FILE))
        self.bm25_retriever = BM25Retriever(self.bm25_index)
        
        # Reranker
        from src.config import JINA_API_KEY
        self.reranker = JinaReranker(api_key=JINA_API_KEY)

    def load_ground_truth(self) -> List[Dict[str, Any]]:
        """
        Carica le query ed i relativi documenti rilevanti dal file del ground truth.
        """
        if not GROUND_TRUTH_FILE.exists():
            raise FileNotFoundError(
                f"File di ground truth non trovato: {GROUND_TRUTH_FILE}. "
                f"Generalo eseguendo prima scripts/generate_ground_truth.py."
            )
            
        logger.info(f"Caricamento ground truth da: {GROUND_TRUTH_FILE}...")
        records = []
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.info(f"Caricate {len(records)} query di valutazione.")
        return records

    def _get_checkpoint_path(self) -> Path:
        return self.results_file.with_name(self.results_file.stem + "_checkpoint.json")

    def _load_checkpoint(self) -> Dict[str, Any]:
        path = self._get_checkpoint_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Errore nel caricamento del checkpoint: {e}. Ricomincio da capo.")
        return {"results": {}}

    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        path = self._get_checkpoint_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Errore nel salvataggio del checkpoint: {e}")

    def run_configuration(self, strategy_name: str, retriever: Any, k_search: int = 15) -> Tuple[Dict[str, Dict[str, float]], float, Dict[str, Any]]:
        """
        Esegue tutte le query del ground truth per un determinato retriever.
        Supporta l'avvio incrementale tramite checkpoint.
        """
        checkpoint = self._load_checkpoint()
        strategy_data = checkpoint["results"].get(strategy_name, {})
        
        run_dict: Dict[str, Dict[str, float]] = strategy_data.get("run_dict", {})
        latencies: Dict[str, float] = strategy_data.get("latencies", {})
        queries_details: Dict[str, Any] = strategy_data.get("queries_details", {})

        # Converte le chiavi latencies a stringhe se caricate da json
        latencies = {str(k): float(v) for k, v in latencies.items()}

        logger.info(f"Esecuzione della configurazione '{strategy_name}' per {len(self.ground_truth)} query...")
        
        already_done = len(run_dict)
        if already_done > 0:
            logger.info(f"Trovate {already_done}/{len(self.ground_truth)} query già eseguite nel checkpoint per '{strategy_name}'.")

        unsaved_changes = False

        for i, record in enumerate(self.ground_truth):
            query_id = record["query_id"]
            query_text = record["query"]

            # Salta se già eseguita
            if query_id in run_dict:
                continue

            # Monitora le chiamate API prima del retrieval per calcolare i cache hit
            api_calls_before = BaseEmbedder._api_calls_count
            sleep_before = BaseEmbedder._total_sleep_time

            # Misura il tempo di esecuzione preciso del retrieval
            start_time = time.perf_counter()
            results = retriever.retrieve(query_text, k=k_search)
            elapsed = (time.perf_counter() - start_time) * 1000.0  # in ms
            
            api_calls_after = BaseEmbedder._api_calls_count
            sleep_after = BaseEmbedder._total_sleep_time

            # Detrae l'eventuale tempo di sleep introdotto dal rate-limiter
            sleep_duration_ms = (sleep_after - sleep_before) * 1000.0
            if sleep_duration_ms > 0:
                elapsed -= sleep_duration_ms
                elapsed = max(elapsed, 0.0)

            # Se siamo in run standard (non fast) e si tratta di retrieval vettoriale o ibrido,
            # in caso di cache hit (nessuna chiamata API extra) aggiungiamo la latenza API media.
            if not self.is_fast_mode and isinstance(retriever, (VectorRetriever, HybridRetriever)):
                if api_calls_after == api_calls_before:
                    if BaseEmbedder._api_calls_count > 0:
                        avg_api_latency = (BaseEmbedder._total_api_time / BaseEmbedder._api_calls_count) * 1000.0
                    else:
                        avg_api_latency = 600.0  # Fallback a 600ms se non ci sono ancora chiamate registrate
                    
                    elapsed += avg_api_latency

            # Mappa i risultati sul formato run di ranx: {doc_id: score}
            # Se doc_id compare più volte (raro), conserva il punteggio maggiore
            query_run = {}
            for doc in results:
                doc_id = doc["id"]
                score = doc["score"]
                query_run[doc_id] = max(query_run.get(doc_id, -float("inf")), score)
                
            run_dict[query_id] = query_run
            latencies[query_id] = elapsed
            
            # Dettagli completi per singola query
            queries_details[query_id] = {
                "query": query_text,
                "category": record.get("category", "unknown"),
                "latency_ms": elapsed,
                "retrieved_docs": [{"id": doc["id"], "score": doc["score"]} for doc in results]
            }

            unsaved_changes = True

            # Salva il checkpoint ogni 10 query per non degradare le performance
            if (i + 1) % 10 == 0:
                checkpoint["results"][strategy_name] = {
                    "run_dict": run_dict,
                    "latencies": latencies,
                    "queries_details": queries_details
                }
                self._save_checkpoint(checkpoint)
                unsaved_changes = False
                logger.info(f"[{strategy_name}] Salvato checkpoint a query {i+1}/{len(self.ground_truth)}")

        # Salva al termine della configurazione se ci sono modifiche pendenti
        if unsaved_changes:
            checkpoint["results"][strategy_name] = {
                "run_dict": run_dict,
                "latencies": latencies,
                "queries_details": queries_details
            }
            self._save_checkpoint(checkpoint)
            logger.info(f"[{strategy_name}] Salvato checkpoint finale della configurazione.")

        avg_latency = sum(latencies.values()) / len(latencies) if latencies else 0.0
        logger.info(f"Completato '{strategy_name}'. Latenza media: {avg_latency:.2f} ms")
        return run_dict, avg_latency, queries_details

    def evaluate_strategy(
        self,
        qrels_dict: Dict[str, Dict[str, int]],
        run_dict: Dict[str, Dict[str, float]],
        category_queries: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Calcola le metriche complessive e disaggregate usando la libreria ranx.
        """
        metrics = [
            "precision@3", "precision@5", "precision@10",
            "recall@3", "recall@5", "recall@10",
            "ndcg@5", "ndcg@10",
            "map@10", "mrr"
        ]

        qrels = Qrels(qrels_dict)
        run = Run(run_dict)

        # 1. Valutazione complessiva (Overall)
        overall_results = evaluate(qrels, run, metrics, make_comparable=True)
        results = {"overall": overall_results}

        # 2. Valutazione disaggregata per categoria di query
        for category, query_ids in category_queries.items():
            if not query_ids:
                continue
                
            # Filtra qrels e run per questa categoria
            sub_qrels_dict = {qid: qrels_dict[qid] for qid in query_ids if qid in qrels_dict}
            sub_run_dict = {qid: run_dict[qid] for qid in query_ids if qid in run_dict}
            
            if not sub_qrels_dict or not sub_run_dict:
                continue
                
            sub_qrels = Qrels(sub_qrels_dict)
            sub_run = Run(sub_run_dict)
            
            category_results = evaluate(sub_qrels, sub_run, metrics, make_comparable=True)
            results[category] = category_results

        return results

    def execute_all(self) -> Dict[str, Any]:
        """
        Esegue la suite completa di esperimenti:
        1. Vector-only
        2. BM25-only
        3. Hybrid RRF per diversi valori di alpha (0.1, 0.3, 0.5, 0.7, 0.9)
        """
        # Prepara qrels_dict nel formato ranx: {query_id: {doc_id: 1}}
        qrels_dict: Dict[str, Dict[str, int]] = {}
        # Prepara il mappaggio delle query per categoria
        category_queries: Dict[str, List[str]] = defaultdict(list)

        for record in self.ground_truth:
            qid = record["query_id"]
            category = record["category"]
            category_queries[category].append(qid)
            
            # Assegna rilevanza pari a 1 per i doc ID associati
            qrels_dict[qid] = {doc_id: 1 for doc_id in record["relevant_doc_ids"]}

        # Definisce le configurazioni da testare
        configurations = []
        
        # 1. Vettoriale Puro
        configurations.append(("Vector-only", self.vector_retriever))
        
        # 2. BM25 Puro
        configurations.append(("BM25-only", self.bm25_retriever))
        
        # 3. Ibrido con diversi Alpha
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            hybrid = HybridRetriever(self.vector_retriever, self.bm25_retriever, alpha=alpha)
            configurations.append((f"Hybrid-RRF (α={alpha})", hybrid))

        # 4. Rerankers (Vector, BM25, e Ibrido con α=0.5)
        configurations.append(("Vector-only + Rerank", RerankRetriever(self.vector_retriever, self.reranker)))
        configurations.append(("BM25-only + Rerank", RerankRetriever(self.bm25_retriever, self.reranker)))
        
        hybrid_05 = HybridRetriever(self.vector_retriever, self.bm25_retriever, alpha=0.5)
        configurations.append(("Hybrid-RRF (α=0.5) + Rerank", RerankRetriever(hybrid_05, self.reranker)))

        evaluation_report: Dict[str, Any] = {
            "metadata": {
                "dataset": DATASET_NAME,
                "embedding_model": DEFAULT_EMBEDDER,
                "chunk_size": CHUNK_SIZE_DEFAULT,
                "chunk_strategy": DEFAULT_CHUNK_STRATEGY,
                "num_queries": len(self.ground_truth),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "results": {}
        }

        # Esecuzione e valutazione di ogni configurazione
        for config_name, retriever in configurations:
            run_dict, avg_latency, queries_details = self.run_configuration(config_name, retriever)
            metrics_results = self.evaluate_strategy(qrels_dict, run_dict, category_queries)
            
            evaluation_report["results"][config_name] = {
                "avg_latency_ms": avg_latency,
                "metrics": metrics_results,
                "queries": queries_details
            }

            # Salva su disco i risultati parziali dopo ciascuna configurazione completata
            with open(self.results_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
            logger.info(f"Salvati risultati parziali per '{config_name}' in {self.results_file.name}")

        logger.info(f"Valutazione completata. Risultati finali salvati in {self.results_file}")
        
        # Rimosso il checkpoint temporaneo al completamento con successo
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info("Rimosso file di checkpoint temporaneo.")
            except Exception as e:
                logger.warning(f"Impossibile rimuovere il file di checkpoint: {e}")
                
        self.print_markdown_table(evaluation_report)
        
        return evaluation_report

    def print_markdown_table(self, report: Dict[str, Any]):
        """
        Stampa a console una tabella comparativa in formato Markdown con i risultati principali.
        """
        results_dict = report["results"]
        
        print("\n" + "=" * 80)
        print("📊 TABELLA COMPARATIVA DEI RISULTATI (OVERALL)")
        print("=" * 80)
        
        headers = ["Strategy", "NDCG@10", "Recall@10", "Recall@5", "Precision@10", "MRR", "Latency (ms)"]
        header_line = " | ".join(headers)
        separator = " | ".join(["---"] * len(headers))
        
        print(f"| {header_line} |")
        print(f"| {separator} |")
        
        for name, data in results_dict.items():
            metrics = data["metrics"]["overall"]
            latency = data["avg_latency_ms"]
            
            ndcg_10 = metrics.get("ndcg@10", 0.0)
            rec_10 = metrics.get("recall@10", 0.0)
            rec_5 = metrics.get("recall@5", 0.0)
            p_10 = metrics.get("precision@10", 0.0)
            mrr = metrics.get("mrr", 0.0)
            
            row = [
                name,
                f"{ndcg_10:.4f}",
                f"{rec_10:.4f}",
                f"{rec_5:.4f}",
                f"{p_10:.4f}",
                f"{mrr:.4f}",
                f"{latency:.1f}"
            ]
            print(f"| {' | '.join(row)} |")
        print("=" * 80 + "\n")
