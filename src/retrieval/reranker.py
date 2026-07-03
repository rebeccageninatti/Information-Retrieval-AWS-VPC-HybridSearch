"""
Modulo per il Reranking avanzato tramite Jina Reranker v3.
Implementa il wrapper dell'API Jina AI ed il RerankRetriever compatibile con BaseRetriever.
"""

import time
import requests
from typing import List, Dict, Any, Optional

from src.retrieval.base import BaseRetriever
from src.indexing.embedder import TokenRateLimiter, BaseEmbedder
from src.utils.logging_utils import get_logger
from src.config import (
    JINA_API_KEY,
    JINA_RERANK_MODEL,
    JINA_RERANK_API_URL,
    JINA_RERANK_RPM_LIMIT,
    JINA_RERANK_TPM_LIMIT,
    JINA_RERANK_DEFAULT_K
)

logger = get_logger(__name__)


class JinaReranker:
    """
    Wrapper per l'API di Reranking di Jina AI (modello jina-reranker-v3).
    Gestisce il rate-limiting tramite TokenRateLimiter e implementa la retry-logic.
    """

    def __init__(
        self,
        api_key: str = JINA_API_KEY,
        model_name: str = JINA_RERANK_MODEL,
        api_url: str = JINA_RERANK_API_URL,
        rpm_limit: int = JINA_RERANK_RPM_LIMIT,
        tpm_limit: int = JINA_RERANK_TPM_LIMIT
    ):
        if not api_key:
            raise ValueError(
                "La chiave API di Jina AI (JINA_API_KEY) è obbligatoria per il reranker! "
                "Configurala nel file .env."
            )
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        
        # Inizializza il rate limiter
        self.rate_limiter = TokenRateLimiter(rpm_limit=rpm_limit, tpm_limit=tpm_limit)
        logger.info(f"JinaReranker inizializzato con modello: {self.model_name}")

    def estimate_tokens(self, text: str) -> int:
        """
        Stima il numero di token per un dato testo (circa 1 token ogni 4 caratteri).
        """
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    def rerank(self, query: str, documents: List[str], max_retries: int = 5, initial_backoff: float = 2.0) -> List[Dict[str, Any]]:
        """
        Esegue la chiamata all'API Jina Reranker per riordinare una lista di documenti rispetto a una query.

        Args:
            query: La query di ricerca.
            documents: La lista di testi dei documenti candidati.
            max_retries: Numero massimo di tentativi in caso di errore 429/503.
            initial_backoff: Tempo di attesa iniziale per il backoff esponenziale.

        Returns:
            Lista di dizionari ordinati per rilevanza decrescente. Ciascun elemento contiene:
            - "index": l'indice originale del documento nella lista passata come input.
            - "relevance_score": il punteggio di rilevanza restituito dal reranker.
        """
        if not documents:
            return []

        # 1. Stima dei token totali (query + tutti i documenti candidati)
        estimated_tokens = self.estimate_tokens(query) + sum(self.estimate_tokens(doc) for doc in documents)
        
        # 2. Controllo del rate limit ed eventuale attesa
        slept = self.rate_limiter.check_and_wait(estimated_tokens)
        BaseEmbedder._total_sleep_time += slept

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False  # Risparmia banda non richiedendo indietro i testi
        }

        backoff = initial_backoff
        for attempt in range(max_retries):
            try:
                logger.info(f"Chiamata API Rerank (Tentativo {attempt + 1}/{max_retries}) per {len(documents)} documenti...")
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
                
                # Rileva errori HTTP e solleva eccezione per triggerare la retry logic
                response.raise_for_status()
                
                data = response.json()
                results = data.get("results", [])
                
                # Registra la transazione nel rate limiter
                actual_tokens = data.get("usage", {}).get("total_tokens", 0)
                if not actual_tokens:
                    actual_tokens = estimated_tokens
                self.rate_limiter.record_request(actual_tokens)

                logger.info(f"Reranking API completato. Token consumati: {actual_tokens}")
                return results

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else 500
                if status_code in [403, 429, 502, 503, 504] and attempt < max_retries - 1:
                    logger.warning(
                        f"Rerank API fallita con codice {status_code}. "
                        f"In attesa di {backoff}s prima di riprovare..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error(f"Errore HTTP non gestibile durante il reranking: {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Errore di connessione o rete durante il reranking ({e}). "
                        f"Riprovo tra {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error(f"Errore irreversibile durante il reranking: {e}")
                    raise

        raise RuntimeError("Impossibile eseguire il reranking dopo il numero massimo di tentativi.")


class RerankRetriever(BaseRetriever):
    """
    Implementazione di BaseRetriever che esegue un passo aggiuntivo di reranking
    sui risultati ottenuti da un retriever di base (ad es. Vector o Hybrid).
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: JinaReranker,
        k_candidate: Optional[int] = None
    ):
        """
        Inizializza il RerankRetriever.

        Args:
            base_retriever: Il retriever di base (es. HybridRetriever) che fornisce i candidati iniziali.
            reranker: Istanza di JinaReranker per ordinare i candidati.
            k_candidate: Numero di candidati da richiedere al retriever di base. 
                         Se None, viene usato un valore predefinito calcolato dinamicamente.
        """
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.k_candidate = k_candidate
        logger.info("RerankRetriever inizializzato correttamente.")

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Esegue il recupero dei documenti in due stadi:
        1. Recupero iniziale di N candidati dal base_retriever.
        2. Reranking dei candidati estratti tramite il modello Jina Reranker v3.
        3. Ritorno dei primi K risultati riordinati.
        """
        # Calcola quanti candidati recuperare dallo stadio 1 (limitato a un massimo di 25)
        candidates_to_retrieve = min(self.k_candidate or JINA_RERANK_DEFAULT_K, 25)
        
        logger.info(f"RerankRetriever: Recupero di {candidates_to_retrieve} candidati tramite {self.base_retriever.__class__.__name__}...")
        candidates = self.base_retriever.retrieve(query, k=candidates_to_retrieve)
        
        if not candidates:
            logger.warning("Nessun candidato restituito dallo stadio di base. Salto il reranking.")
            return []

        # Estrae il testo da ciascun candidato per passarlo all'API del Reranker
        doc_texts = [c["text"] for c in candidates]
        
        logger.info(f"RerankRetriever: Richiesta di Rerank per {len(doc_texts)} candidati...")
        rerank_results = self.reranker.rerank(query, doc_texts)

        # Riorganizza i candidati originali in base al nuovo ordinamento
        final_results = []
        for rank_idx, item in enumerate(rerank_results[:k]):
            orig_idx = item["index"]
            relevance_score = item["relevance_score"]
            orig_doc = candidates[orig_idx]
            
            # Combina i metadati e traccia l'origine
            source_details = orig_doc.get("source_details", {}) or {}
            source_details["base_score"] = orig_doc.get("score", 0.0)
            source_details["rerank_rank"] = rank_idx + 1

            reranked_doc = {
                "id": orig_doc["id"],
                "text": orig_doc["text"],
                "metadata": orig_doc["metadata"],
                "score": float(relevance_score),
                "source_details": source_details
            }
            final_results.append(reranked_doc)

        logger.info(f"RerankRetriever completato. Restituiti {len(final_results)} risultati su {len(candidates)} candidati.")
        return final_results
