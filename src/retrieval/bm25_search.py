"""
Modulo per la ricerca lessicale classica tramite BM25.
"""

from typing import List, Dict, Any
from src.retrieval.base import BaseRetriever
from src.indexing.bm25_index import BM25Index
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BM25Retriever(BaseRetriever):
    """
    Retriever basato esclusivamente sulla ricerca lessicale BM25.
    """

    def __init__(self, bm25_index: BM25Index):
        """
        Inizializza il BM25Retriever.

        Args:
            bm25_index: Istanza di BM25Index.
        """
        self.bm25_index = bm25_index

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Esegue la ricerca lessicale usando BM25.

        Args:
            query: Testo da cercare.
            k: Numero di risultati.

        Returns:
            Lista di dizionari con i risultati, formattata uniformemente.
        """
        logger.info(f"Esecuzione ricerca BM25 per la query: '{query}' (K={k})")
        return self.bm25_index.search(query, k=k)
