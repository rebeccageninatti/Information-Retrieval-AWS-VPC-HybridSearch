"""
Modulo per la ricerca ibrida tramite fusione dei risultati lessicali e vettoriali.
Implementa l'algoritmo Reciprocal Rank Fusion (RRF) pesato.
"""

from typing import List, Dict, Any, Optional
from src.retrieval.base import BaseRetriever
from src.retrieval.vector_search import VectorRetriever
from src.retrieval.bm25_search import BM25Retriever
from src.config import RRF_K, DEFAULT_ALPHA
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Retriever ibrido che unisce la ricerca semantica vettoriale e lessicale BM25
    utilizzando l'algoritmo Reciprocal Rank Fusion (RRF) pesato tramite un parametro alpha.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = DEFAULT_ALPHA,
        rrf_k: int = RRF_K
    ):
        """
        Inizializza il retriever ibrido.

        Args:
            vector_retriever: Retriever vettoriale.
            bm25_retriever: Retriever lessicale BM25.
            alpha: Peso relativo del contributo vettoriale [0.0, 1.0].
                   0.0 corrisponde a ricerca solo lessicale, 1.0 a solo vettoriale.
            rrf_k: Costante k dell'algoritmo RRF (standard di letteratura = 60).
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k
        logger.info(f"HybridRetriever inizializzato: alpha={alpha}, rrf_k={rrf_k}")

    def retrieve(self, query: str, k: int = 10, alpha: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Esegue la ricerca ibrida con fusione RRF.

        Args:
            query: La stringa da cercare.
            k: Numero di risultati finali richiesti.
            alpha: Eventuale override locale del peso alpha per questa query.

        Returns:
            Lista ordinata dei primi K chunk fusi.
        """
        active_alpha = self.alpha if alpha is None else alpha
        if not (0.0 <= active_alpha <= 1.0):
            raise ValueError(f"Il parametro alpha deve essere compreso tra 0.0 e 1.0, ricevuto: {active_alpha}")

        # 1. Recupero dei candidati da entrambi i motori.
        # Recuperiamo un numero maggiore di candidati (es. max(k * 2, 50))
        # per garantire che elementi rilevanti penalizzati da un motore ma promossi
        # dall'altro non vengano esclusi prima della fusione.
        candidate_k = max(k * 2, 50)
        logger.info(f"Richiesta di {candidate_k} candidati per fusione RRF con alpha={active_alpha}")

        vector_results = self.vector_retriever.retrieve(query, k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query, k=candidate_k)

        # 2. Estrazione delle graduatorie (rank 1-based)
        vector_ranks = {doc["id"]: idx + 1 for idx, doc in enumerate(vector_results)}
        bm25_ranks = {doc["id"]: idx + 1 for idx, doc in enumerate(bm25_results)}

        # Memorizzazione di testi e metadati associati a ciascun ID (unione dei due insiemi)
        doc_store = {}
        for doc in vector_results:
            doc_store[doc["id"]] = doc
        for doc in bm25_results:
            if doc["id"] not in doc_store:
                doc_store[doc["id"]] = doc

        # 3. Calcolo del punteggio RRF pesato per ogni documento
        rrf_scores = {}
        for doc_id in doc_store.keys():
            v_rank = vector_ranks.get(doc_id)
            b_rank = bm25_ranks.get(doc_id)

            # Contributo vettoriale
            v_term = 0.0
            if v_rank is not None:
                v_term = 1.0 / (self.rrf_k + v_rank)

            # Contributo BM25
            b_term = 0.0
            if b_rank is not None:
                b_term = 1.0 / (self.rrf_k + b_rank)

            # Calcolo punteggio finale combinato
            rrf_scores[doc_id] = active_alpha * v_term + (1.0 - active_alpha) * b_term

        # 4. Ordinamento per score decrescente
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 5. Formattazione finale dei risultati e selezione dei primi K
        fused_results = []
        for doc_id in sorted_ids[:k]:
            doc = doc_store[doc_id]
            fused_results.append({
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(rrf_scores[doc_id]),
                "source_details": {
                    "vector_rank": vector_ranks.get(doc_id, None),
                    "bm25_rank": bm25_ranks.get(doc_id, None)
                }
            })

        logger.info(f"Ricerca ibrida completata. Trovati {len(fused_results)} risultati.")
        return fused_results
