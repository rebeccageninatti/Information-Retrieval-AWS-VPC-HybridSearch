"""
Modulo per l'interazione con ChromaDB.
Definisce l'interfaccia astratta BaseVectorStore e l'implementazione concreta ChromaVectorStore.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, cast
import chromadb
from chromadb.api.types import EmbeddingFunction

from src.config import CHROMA_DB_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseVectorStore(ABC):
    """
    Classe base astratta per i Vector Database.
    Permette di isolare l'applicazione dall'implementazione specifica del DB.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        Aggiunge i documenti al database vettoriale.
        """
        pass

    @abstractmethod
    def search(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Effettua la ricerca vettoriale a partire da una query testuale (usa l'embedder interno).
        """
        pass

    @abstractmethod
    def search_by_vector(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """
        Effettua la ricerca vettoriale a partire da un vettore di embedding già calcolato.
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """
        Cancella la collezione corrente per consentire una reinstallazione pulita.
        """
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    Implementazione concreta di BaseVectorStore basata su ChromaDB.
    Gestisce la pulizia dei metadati e il mapping delle risposte.
    """

    def __init__(
        self,
        collection_name: str = "aws_vpc_chunks",
        embedding_function: Optional[EmbeddingFunction] = None,
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_dir = persist_directory or str(CHROMA_DB_DIR)

        logger.info(f"Inizializzazione client ChromaDB persistente su: {self.persist_dir}")
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Inizializza o recupera la collezione
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        logger.info(f"Collezione ChromaDB '{self.collection_name}' pronta. Chunk presenti: {self.collection.count()}")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB accetta solo tipi primitivi (str, int, float, bool) nei metadati.
        Dizionari nidificati o liste generano errori. Questo metodo appiattisce/sanifica i dati.
        """
        sanitized = {}
        for key, val in metadata.items():
            if val is None:
                sanitized[key] = ""
            elif isinstance(val, (str, int, float, bool)):
                sanitized[key] = val
            elif isinstance(val, dict):
                # Appiattisce le chiavi (es: "headers" -> "headers_Header 1")
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, (str, int, float, bool)):
                        sanitized[f"{key}_{sub_key}"] = sub_val
                    elif sub_val is None:
                        sanitized[f"{key}_{sub_key}"] = ""
                    else:
                        sanitized[f"{key}_{sub_key}"] = str(sub_val)
            else:
                # Converte tipi complessi (liste, tuple) in stringa
                sanitized[key] = str(val)
        return sanitized

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        Aggiunge chunk e metadati a ChromaDB. Se gli embedding sono pre-calcolati, li inserisce direttamente.
        """
        if not documents:
            logger.warning("Nessun documento passato per l'aggiunta.")
            return

        if len(documents) != len(ids):
            raise ValueError(f"La lunghezza dei documenti ({len(documents)}) deve coincidere con quella degli ID ({len(ids)})")

        sanitized_metadatas = None
        if metadatas:
            if len(metadatas) != len(documents):
                raise ValueError(f"La lunghezza dei metadati ({len(metadatas)}) deve coincidere con quella dei documenti ({len(documents)})")
            sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]

        # ChromaDB consiglia di inserire in piccoli lotti per evitare problemi di timeout di rete/memoria
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            b_docs = documents[i:end_idx]
            b_ids = ids[i:end_idx]
            b_meta = sanitized_metadatas[i:end_idx] if sanitized_metadatas else None
            b_embeds = embeddings[i:end_idx] if embeddings else None
            
            logger.info(f"Caricamento batch {i // batch_size + 1} in ChromaDB (Elementi {i} a {end_idx})...")
            
            self.collection.add(
                documents=b_docs,
                ids=b_ids,
                metadatas=cast(Any, b_meta),
                embeddings=cast(Any, b_embeds)
            )

        logger.info(f"Caricamento completato con successo. Numero totale di chunk ora presenti: {self.collection.count()}")

    def search(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Effettua la query usando il testo, delegando a ChromaDB la chiamata all'embedding_function.
        Durante la query, imposta temporaneamente il task su query-mode per ottimizzare la ricerca.
        """
        if not self.embedding_function:
            raise ValueError("Impossibile cercare via testo: nessuna embedding_function è configurata in questo store.")

        # Gestione temporanea del task type per ottimizzare la query
        old_task_type = None
        old_task = None
        
        # Controlla se la embedding_function ha attributi per il task type
        if hasattr(self.embedding_function, "task_type"):
            old_task_type = getattr(self.embedding_function, "task_type")
            setattr(self.embedding_function, "task_type", "RETRIEVAL_QUERY")
        if hasattr(self.embedding_function, "task"):
            old_task = getattr(self.embedding_function, "task")
            setattr(self.embedding_function, "task", "retrieval.query")

        try:
            logger.info(f"Ricerca testuale in ChromaDB per: '{query_text}' con K={k}")
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k
            )
        finally:
            # Ripristina lo stato precedente
            if old_task_type is not None:
                setattr(self.embedding_function, "task_type", old_task_type)
            if old_task is not None:
                setattr(self.embedding_function, "task", old_task)

        return self._format_results(results)

    def search_by_vector(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """
        Effettua la query fornendo direttamente il vettore di embedding.
        """
        logger.info(f"Ricerca vettoriale in ChromaDB con K={k}")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return self._format_results(results)

    def delete_collection(self) -> None:
        """
        Elimina la collezione dal DB.
        """
        logger.warning(f"Eliminazione della collezione ChromaDB '{self.collection_name}'...")
        try:
            self.client.delete_collection(self.collection_name)
            # Ricrea la collezione vuota
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info("Collezione eliminata e ricreata vuota.")
        except Exception as e:
            logger.error(f"Errore durante l'eliminazione della collezione: {e}")

    def _format_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """
        Formatta l'output raw di ChromaDB in una lista uniforme di dizionari.
        Formato restituito:
        [
            {
                "id": "chunk_1",
                "text": "contenuto...",
                "metadata": {...},
                "distance": 0.23
            },
            ...
        ]
        """
        formatted = []
        # Poiché passiamo una singola query alla volta, prendiamo il primo elemento dei risultati
        ids = raw_results.get("ids", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]

        for idx in range(len(ids)):
            formatted.append({
                "id": ids[idx],
                "text": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "score": 1 - distances[idx] if idx < len(distances) else 0.0,  # Convertiamo distanza in score di similarità
                "distance": distances[idx] if idx < len(distances) else 0.0
            })
        return formatted
