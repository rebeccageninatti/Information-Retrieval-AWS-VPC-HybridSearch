"""
Modulo contenente la classe base astratta per le strategie di retrieval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    """
    Interfaccia base per tutti i motori di ricerca (retrievers).
    Consente di scambiare facilmente diverse strategie di retrieval.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Esegue il recupero dei documenti rilevanti rispetto alla query.

        Args:
            query: La stringa di ricerca inserita dall'utente.
            k: Il numero massimo di risultati da restituire.

        Returns:
            Lista di dizionari, ognuno contenente:
            - "id": ID univoco del chunk
            - "text": Il testo del chunk
            - "metadata": Metadati associati al chunk
            - "score": Punteggio di rilevanza (maggiore è meglio)
        """
        pass
