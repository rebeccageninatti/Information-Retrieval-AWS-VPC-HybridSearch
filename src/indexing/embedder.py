"""
Wrapper per modelli di embedding con gestione del rate-limiting (RPM, TPM, limiti giornalieri)
e retry-logic per le API di Jina AI e Google Gemini.
"""

import time
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Dict, Any, Optional, cast
import requests

from chromadb import EmbeddingFunction, Documents, Embeddings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TokenRateLimiter:
    """
    Sliding window rate limiter per limitare RPM, TPM e richieste giornaliere (RPD).
    """

    def __init__(self, rpm_limit: int, tpm_limit: int, daily_limit: Optional[int] = None):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.daily_limit = daily_limit

        # Ciascun elemento è una tupla (timestamp, tokens)
        self.requests_minute = deque()
        # Per il calcolo giornaliero, teniamo traccia dei timestamp delle richieste degli ultimi 86400 secondi
        self.requests_day = deque()

    def _clean_old_requests(self):
        now = time.time()
        # Pulisce le richieste più vecchie di 60 secondi
        while self.requests_minute and now - self.requests_minute[0][0] > 60:
            self.requests_minute.popleft()
        
        # Pulisce le richieste più vecchie di 24 ore
        if self.daily_limit:
            while self.requests_day and now - self.requests_day[0] > 86400:
                self.requests_day.popleft()

    def check_and_wait(self, tokens_to_send: int) -> float:
        """
        Verifica se l'invio di tokens_to_send causerebbe il superamento dei limiti di rate.
        Se sì, attende (sleep) fino a quando la finestra temporale non si libera.
        Se viene superato il limite giornaliero, solleva una RuntimeError per bloccare l'esecuzione.
        Ritorna il tempo totale speso in sleep (in secondi).
        """
        self._clean_old_requests()
        
        # 1. Verifica limite giornaliero
        if self.daily_limit and len(self.requests_day) >= self.daily_limit:
            # Calcoliamo quanto manca prima che la richiesta più vecchia esca dalla finestra giornaliera
            oldest_time = self.requests_day[0]
            time_left = 86400 - (time.time() - oldest_time)
            hours_left = time_left / 3600
            error_msg = (
                f"LIMITE GIORNALIERO RAGGIUNTO! Sono state effettuate {len(self.requests_day)} richieste nelle ultime 24 ore. "
                f"Il limite giornaliero è di {self.daily_limit}. La prima richiesta scadrà tra circa {hours_left:.2f} ore. "
                f"Blocco l'esecuzione per evitare addebiti o errori dell'API."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # 2. Verifica limiti al minuto (RPM e TPM)
        now = time.time()
        current_rpm = len(self.requests_minute)
        current_tpm = sum(req[1] for req in self.requests_minute)

        total_slept = 0.0
        while current_rpm >= self.rpm_limit or (current_tpm + tokens_to_send) > self.tpm_limit:
            if not self.requests_minute:
                break
                
            # Calcola il tempo di attesa necessario basato sul primo elemento della coda da eliminare
            oldest_time, oldest_tokens = self.requests_minute[0]
            wait_time = 60.1 - (now - oldest_time)
            
            if wait_time > 0:
                logger.warning(
                    f"Rate limit in avvicinamento. RPM attuali: {current_rpm}/{self.rpm_limit}, "
                    f"TPM attuali: {current_tpm}/{self.tpm_limit} (richiesti: {tokens_to_send}). "
                    f"In attesa per {wait_time:.2f} secondi..."
                )
                time.sleep(wait_time)
                total_slept += wait_time
                
            self._clean_old_requests()
            now = time.time()
            current_rpm = len(self.requests_minute)
            current_tpm = sum(req[1] for req in self.requests_minute)
            
        return total_slept

    def record_request(self, tokens_sent: int):
        """
        Registra la richiesta corrente con timestamp e numero di token associati.
        """
        now = time.time()
        self.requests_minute.append((now, tokens_sent))
        if self.daily_limit:
            self.requests_day.append(now)


class BaseEmbedder(EmbeddingFunction, ABC):
    """
    Classe base astratta per gli embedder del progetto.
    Implementa l'interfaccia di ChromaDB `EmbeddingFunction` per integrazione immediata.
    """

    # Cache condivisa tra tutte le istanze degli embedder per evitare doppie chiamate API
    _query_cache: Dict[str, List[float]] = {}
    
    # Tracking delle chiamate API per la stima realistica delle latenze
    _total_api_time: float = 0.0
    _api_calls_count: int = 0
    _total_sleep_time: float = 0.0

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Stima realistica dei token basata sul numero di caratteri (1 token ~= 4 caratteri per inglese/codice).
        """
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    @abstractmethod
    def _embed_batch(self, batch: List[str]) -> tuple[List[List[float]], int]:
        """
        Metodo interno per effettuare la chiamata API per un singolo batch.
        Ritorna una tupla (lista di embeddings, token consumati).
        """
        pass

    def __call__(self, input: Documents) -> Embeddings:
        """
        Metodo principale invocato da ChromaDB e dal codice client.
        Gestisce lo splitting in batch, il rate-limiting trasparente e la cache.
        """
        # Se viene passata una singola stringa anziché una lista, la convertiamo
        if isinstance(input, str):
            input = [input]

        # Proviamo a soddisfare l'intero input dalla cache se possibile
        cached_all = True
        cached_results = []
        for text in input:
            if text in BaseEmbedder._query_cache:
                cached_results.append(BaseEmbedder._query_cache[text])
            else:
                cached_all = False
                break

        if cached_all:
            return cast(Embeddings, cached_results)

        # Se non è tutto in cache, eseguiamo normalmente caricando i mancanti nella cache
        embeddings: List[List[float]] = []
        
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            
            # Filtriamo gli elementi del batch che non sono già in cache
            batch_to_embed = []
            for text in batch:
                if text not in BaseEmbedder._query_cache:
                    batch_to_embed.append(text)

            if batch_to_embed:
                batch_tokens = sum(self.estimate_tokens(text) for text in batch_to_embed)
                
                # Controlla se dobbiamo attendere
                self.check_rate_limits(batch_tokens)
                
                # Esegue la chiamata all'API con retry logic
                batch_embeddings, actual_tokens = self._embed_batch_with_retry(batch_to_embed)
                
                # Registra la transazione nel rate limiter con i token reali consumati
                self.record_rate_limits(actual_tokens)
                
                # Salva nella cache
                for text, emb in zip(batch_to_embed, batch_embeddings):
                    BaseEmbedder._query_cache[text] = emb

            # Ricostruisce il batch completo leggendo dalla cache
            full_batch_embeddings = []
            for text in batch:
                full_batch_embeddings.append(BaseEmbedder._query_cache[text])
                
            embeddings.extend(full_batch_embeddings)

        return cast(Embeddings, embeddings)

    @abstractmethod
    def check_rate_limits(self, tokens_to_send: int):
        pass

    @abstractmethod
    def record_rate_limits(self, tokens_sent: int):
        pass

    def _embed_batch_with_retry(self, batch: List[str], max_retries: int = 5, initial_backoff: float = 2.0) -> tuple[List[List[float]], int]:
        """
        Implementazione robusta con retry ed exponential backoff per gestire i codici 429/503 delle API.
        """
        backoff = initial_backoff
        for attempt in range(max_retries):
            try:
                start_time = time.perf_counter()
                embeddings, actual_tokens = self._embed_batch(batch)
                elapsed = time.perf_counter() - start_time
                BaseEmbedder._total_api_time += elapsed
                BaseEmbedder._api_calls_count += len(batch)
                return embeddings, actual_tokens
            except requests.exceptions.HTTPError as e:
                # Se l'errore è dovuto a rate limit (429) o problemi server temporanei (503/504/502)
                status_code = e.response.status_code if e.response is not None else 500
                if status_code in [429, 502, 503, 504] and attempt < max_retries - 1:
                    logger.warning(
                        f"Richiesta fallita con codice {status_code} (Tentativo {attempt + 1}/{max_retries}). "
                        f"Backoff di {backoff}s prima di riprovare..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error(f"Errore HTTP non gestibile durante l'embedding: {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Errore di rete o connessione (Tentativo {attempt + 1}/{max_retries}): {e}. "
                        f"Riprovo tra {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error(f"Errore fatale durante l'embedding: {e}")
                    raise
        raise RuntimeError("Impossibile generare gli embeddings dopo il numero massimo di tentativi.")


class JinaEmbedder(BaseEmbedder):
    """
    Embedder concreto per l'API di Jina AI.
    Limiti imposti: 100 RPM & 100,000 TPM
    """

    def __init__(
        self, 
        api_key: str, 
        model_name: str = "jina-embeddings-v5-text-small", 
        batch_size: int = 32,
        task: Optional[str] = "retrieval.passage",
        dimensions: Optional[int] = 1024
    ):
        super().__init__(model_name, batch_size)
        if not api_key:
            raise ValueError("La chiave API di Jina AI (JINA_API_KEY) è obbligatoria!")
        self.api_key = api_key
        self.task = task
        self.dimensions = dimensions
        
        # Inizializza il rate limiter con i limiti specificati
        from src.config import JINA_RPM_LIMIT, JINA_TPM_LIMIT
        self.rate_limiter = TokenRateLimiter(rpm_limit=JINA_RPM_LIMIT, tpm_limit=JINA_TPM_LIMIT)

    def check_rate_limits(self, tokens_to_send: int):
        slept = self.rate_limiter.check_and_wait(tokens_to_send)
        BaseEmbedder._total_sleep_time += slept

    def record_rate_limits(self, tokens_sent: int):
        self.rate_limiter.record_request(tokens_sent)

    def _embed_batch(self, batch: List[str]) -> tuple[List[List[float]], int]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": batch,
            "embedding_type": "float"
        }
        if self.task:
            payload["task"] = self.task
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        
        url = "https://api.jina.ai/v1/embeddings"
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        # Solleva HTTPError per codici di stato 4xx o 5xx
        response.raise_for_status()
        
        data = response.json()
        
        # Estrae i vettori ordinati per l'indice originale
        embeddings_data = data.get("data", [])
        # Jina restituisce i risultati ordinati per index o già nello stesso ordine dell'input
        embeddings_sorted = sorted(embeddings_data, key=lambda x: x.get("index", 0))
        
        # Estrae i token reali consumati (o ripiega sulla stima se mancante)
        actual_tokens = data.get("usage", {}).get("total_tokens", 0)
        if not actual_tokens:
            actual_tokens = sum(self.estimate_tokens(text) for text in batch)
            
        embeddings = [item["embedding"] for item in embeddings_sorted]
        return embeddings, actual_tokens


def get_embedder() -> BaseEmbedder:
    """
    Inizializza l'embedder corretto in base alle configurazioni centrali di src.config.
    Carica le chiavi API opportune, esegue i controlli e restituisce l'istanza concreta.
    """
    # Import differito per evitare dipendenze circolari
    from src.config import (
        DEFAULT_EMBEDDER,
        JINA_API_KEY,
        JINA_EMBEDDING_MODEL,
        JINA_EMBEDDING_DIMENSION,
    )

    logger.info(f"Inizializzazione embedder: {DEFAULT_EMBEDDER}")

    if DEFAULT_EMBEDDER == "jina":
        if not JINA_API_KEY or JINA_API_KEY == "your_jina_api_key_here" or JINA_API_KEY == "":
            raise ValueError(
                "Chiave API di Jina AI non impostata nel file .env! "
                "Configura JINA_API_KEY prima di procedere."
            )
        return JinaEmbedder(
            api_key=JINA_API_KEY,
            model_name=JINA_EMBEDDING_MODEL,
            dimensions=JINA_EMBEDDING_DIMENSION,
        )

    else:
        raise ValueError(f"Embedder non supportato nelle configurazioni: {DEFAULT_EMBEDDER}")


