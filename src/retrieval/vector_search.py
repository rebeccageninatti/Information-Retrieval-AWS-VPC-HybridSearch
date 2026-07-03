"""
Modulo per la ricerca semantica densa (vettoriale) tramite ChromaDB.
"""

from typing import List, Dict, Any
from src.retrieval.base import BaseRetriever
from src.indexing.vector_store import BaseVectorStore
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VectorRetriever(BaseRetriever):
    """
    Retriever basato esclusivamente sulla ricerca vettoriale semantica.
    """

    def __init__(self, vector_store: BaseVectorStore):
        """
        Inizializza il VectorRetriever.

        Args:
            vector_store: Istanza di un Vector Store compatibile con l'interfaccia BaseVectorStore.
        """
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Esegue la ricerca semantica nel database vettoriale.

        Args:
            query: Testo da cercare.
            k: Numero di risultati.

        Returns:
            Lista di dizionari con i risultati, formattata uniformemente.
        """
        logger.info(f"Esecuzione ricerca vettoriale per la query: '{query}' (K={k})")
        return self.vector_store.search(query, k=k)
