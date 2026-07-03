"""
Modulo per l'indicizzazione lessicale tramite BM25.
Utilizza la libreria rank-bm25 per calcolare la rilevanza testuale e supporta la serializzazione su disco.
"""

import re
import pickle
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BM25Index:
    """
    Indice lessicale BM25. Tokenizza i documenti, calcola le frequenze dei termini,
    salva lo stato su disco e fornisce metodi di ricerca con lo stesso formato di ChromaDB.
    """

    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            corpus: Lista di record dei chunk (dizionari con 'chunk_id', 'text', 'headers', ecc.)
        """
        self.corpus = corpus or []
        self.bm25: Optional[BM25Okapi] = None
        
        if self.corpus:
            self._build_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizzazione semplice: converte in minuscolo e divide in parole ignorando la punteggiatura.
        """
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def _build_index(self):
        """
        Tokenizza tutti i documenti nel corpus e inizializza l'oggetto BM25Okapi.
        """
        logger.info(f"Costruzione dell'indice BM25 per {len(self.corpus)} documenti...")
        tokenized_corpus = [self.tokenize(doc["text"]) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Indice BM25 costruito con successo.")

    def save(self, file_path: str) -> None:
        """
        Salva l'intero stato dell'indice (corpus + oggetto BM25) su disco usando pickle.
        """
        if not self.bm25:
            raise ValueError("Impossibile salvare un indice non ancora inizializzato.")
            
        logger.info(f"Salvataggio dell'indice BM25 su: {file_path}")
        state = {
            "corpus": self.corpus,
            "bm25": self.bm25
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Salvataggio completato.")

    @classmethod
    def load(cls, file_path: str) -> "BM25Index":
        """
        Carica un indice salvato precedentemente da un file pickle.
        """
        logger.info(f"Caricamento dell'indice BM25 da: {file_path}")
        with open(file_path, "rb") as f:
            state = pickle.load(f)
            
        index = cls()
        index.corpus = state["corpus"]
        index.bm25 = state["bm25"]
        logger.info(f"Indice BM25 caricato con successo. Documenti: {len(index.corpus)}")
        return index

    def search(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Effettua la ricerca BM25 per la query specificata.
        Restituisce una lista di risultati formattata in modo compatibile con ChromaVectorStore.
        
        Formato restituito:
        [
            {
                "id": "chunk_1",
                "text": "contenuto...",
                "metadata": {...},
                "score": 14.52
            },
            ...
        ]
        """
        if not self.bm25:
            raise ValueError("L'indice BM25 non è stato inizializzato o caricato.")

        tokenized_query = self.tokenize(query_text)
        if not tokenized_query:
            return []

        logger.info(f"Ricerca BM25 per: '{query_text}' con K={k}")
        
        # Ottiene i punteggi per tutti i documenti nel corpus
        scores = self.bm25.get_scores(tokenized_query)
        
        # Associa ciascun punteggio al corrispondente documento
        results = []
        for doc_idx, score in enumerate(scores):
            if score > 0.0:  # Restringe solo a documenti che hanno almeno un match di termine
                doc = self.corpus[doc_idx]
                results.append({
                    "id": doc["chunk_id"],
                    "text": doc["text"],
                    "metadata": {
                        "source_file": doc.get("source_file", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "strategy": doc.get("strategy", ""),
                        "header_path": doc.get("header_path", ""),
                        **doc.get("headers", {})
                    },
                    "score": float(score)
                })

        # Ordina per punteggio decrescente e restituisce i primi K risultati
        results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
        return results_sorted[:k]
