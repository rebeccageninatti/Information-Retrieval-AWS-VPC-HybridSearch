"""
Script per eseguire la Fase 2 (Indicizzazione):
1. Carica i chunk prodotti nella Fase 1.
2. Inizializza l'embedder configurato (Jina o Gemini) caricando le API key dal file .env.
3. Genera gli embedding degli chunks (con caching su disco per evitare chiamate duplicate all'API).
4. Indicizza i chunk ed i vettori in ChromaDB.
5. Costruisce l'indice BM25 con i medesimi chunk e lo salva su disco.
6. Esegue una query di test su entrambi gli indici per validare il corretto funzionamento.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

from src.config import (
    DEFAULT_EMBEDDER,
    CHUNKS_FILE,
    BM25_INDEX_FILE,
    EMBEDDINGS_CACHE_FILE,
    CHROMA_COLLECTION_NAME,
    EMBEDDINGS_DIR,
    INDICES_DIR,
    DEFAULT_CHUNK_STRATEGY,
    CHUNK_SIZE_DEFAULT,
    DATASET_NAME
)
from src.utils.logging_utils import get_logger
from src.indexing import ChromaVectorStore, BM25Index, get_embedder

logger = get_logger(__name__)


def load_chunks() -> List[Dict[str, Any]]:
    """
    Carica i chunk salvati nella Fase 1.
    """
    chunk_file = CHUNKS_FILE
    if not chunk_file.exists():
        logger.error(f"File dei chunk non trovato: {chunk_file}. Esegui prima scripts/run_ingestion.py!")
        sys.exit(1)

    logger.info(f"Caricamento chunk da {chunk_file}...")
    chunks = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    logger.info(f"Caricati {len(chunks)} chunk.")
    return chunks





def get_embeddings_with_cache(chunks: List[Dict[str, Any]], embedder) -> List[List[float]]:
    """
    Calcola gli embedding per tutti i chunk.
    Utilizza una cache locale su file per evitare di chiamare ripetutamente le API remote.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = EMBEDDINGS_CACHE_FILE
    
    cache: Dict[str, List[float]] = {}
    if cache_file.exists():
        logger.info(f"Trovata cache di embedding locale su: {cache_file}. Caricamento in corso...")
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            logger.info(f"Caricati {len(cache)} embedding dalla cache.")
        except Exception as e:
            logger.warning(f"Errore nel caricamento della cache, procedo senza cache: {e}")
    else:
        # Se il file di cache globale non esiste, facciamo bootstrap caricando le cache delle singole sottocartelle
        # Identifica le sottocartelle di provenienza dai chunk caricati
        subdirs = set()
        for c in chunks:
            parts = Path(c["source_file"]).parts
            if len(parts) > 1:
                subdirs.add(parts[0])
                
        for subdir_name in subdirs:
            # Cerca se esiste una cache specifica per questo sotto-dataset
            subdir_cache_file = EMBEDDINGS_DIR / f"embeddings_cache_{DEFAULT_EMBEDDER}_{DATASET_NAME}_{subdir_name}_{DEFAULT_CHUNK_STRATEGY}_{CHUNK_SIZE_DEFAULT}.pkl"
            if subdir_cache_file.exists():
                logger.info(f"Trovata cache di embedding per '{subdir_name}' su {subdir_cache_file.name}. Caricamento (bootstrap)...")
                try:
                    with open(subdir_cache_file, "rb") as f:
                        subdir_cache = pickle.load(f)
                    cache.update(subdir_cache)
                    logger.info(f"Caricati {len(subdir_cache)} embedding di '{subdir_name}' nella cache.")
                except Exception as e:
                    logger.warning(f"Errore nel caricamento della cache di '{subdir_name}': {e}")

    # Identifica quali chunk non sono in cache
    missing_chunks = [c for c in chunks if c["chunk_id"] not in cache]
    
    if missing_chunks:
        logger.info(f"Generazione di {len(missing_chunks)} nuovi embedding tramite API {DEFAULT_EMBEDDER}...")
        texts_to_embed = [c["text"] for c in missing_chunks]
        
        # Effettua la chiamata API (che internamente gestisce batching e rate limiting)
        new_embeddings = embedder(texts_to_embed)
        
        # Salva i nuovi embedding nella cache
        for chunk, emb in zip(missing_chunks, new_embeddings):
            cache[chunk["chunk_id"]] = emb
            
        # Serializza la cache aggiornata
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache, f)
            logger.info(f"Cache degli embedding aggiornata e salvata su: {cache_file}")
        except Exception as e:
            logger.error(f"Impossibile salvare la cache su file: {e}")
    else:
        logger.info("Tutti gli embedding sono già presenti in cache! Nessuna chiamata API necessaria.")

    # Costruisce la lista finale ordinata di embedding nello stesso ordine dei chunk in input
    final_embeddings = [cache[c["chunk_id"]] for c in chunks]
    return final_embeddings


def main():
    logger.info("=== INIZIO FASE 2: INDICIZZAZIONE ===")
    
    # 1. Caricamento chunk
    chunks = load_chunks()
    
    # 2. Inizializzazione dell'embedder
    try:
        embedder = get_embedder()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # 3. Ottenimento degli embeddings (con cache)
    embeddings = get_embeddings_with_cache(chunks, embedder)
    
    # 4. Indicizzazione su ChromaDB
    logger.info("Inizializzazione del Vector Store ChromaDB...")
    # Creiamo ChromaVectorStore associando l'embedder come embedding_function 
    # così che sia in grado di embeddare le query utente automaticamente durante la ricerca
    vector_store = ChromaVectorStore(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedder
    )
    
    # Rimuoviamo eventuali record precedenti per garantire l'idempotenza dello script
    vector_store.delete_collection()
    
    documents = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    # Usiamo direttamente la struttura originaria dei chunk come metadati (saranno sanificati internamente)
    metadatas = chunks
    
    logger.info("Caricamento dei documenti e dei vettori in ChromaDB...")
    vector_store.add_documents(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
        embeddings=embeddings
    )
    
    # 5. Indicizzazione lessicale BM25
    logger.info("Costruzione dell'indice BM25...")
    bm25_index = BM25Index(corpus=chunks)
    
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    bm25_file = INDICES_DIR / f"bm25_index_{DEFAULT_CHUNK_STRATEGY}_{CHUNK_SIZE_DEFAULT}.pkl"
    bm25_index.save(str(BM25_INDEX_FILE))
    
    # 6. Esecuzione query di test per verifica
    logger.info("=== VERIFICA DEGLI INDICI CON QUERY DI TEST ===")
    test_query = "What is the maximum number of VPCs allowed in an AWS region?"
    logger.info(f"Query di test: '{test_query}'")
    
    # Test ChromaDB
    chroma_results = vector_store.search(test_query, k=3)
    logger.info("\n--- Primi 3 risultati da ChromaDB (Vettoriale) ---")
    for r in chroma_results:
        print(f"ID: {r['id']} (Score/Similarity: {r['score']:.4f})")
        print(f"File sorgente: {r['metadata'].get('source_file')}")
        print(f"Header Path: {r['metadata'].get('header_path')}")
        # Stampa i primi 120 caratteri per brevità
        text_preview = r['text'].replace('\n', ' ')[:120]
        print(f"Preview: {text_preview}...\n")
        
    # Test BM25
    bm25_results = bm25_index.search(test_query, k=3)
    logger.info("\n--- Primi 3 risultati da BM25 (Lessicale) ---")
    for r in bm25_results:
        print(f"ID: {r['id']} (Score: {r['score']:.4f})")
        print(f"File sorgente: {r['metadata'].get('source_file')}")
        print(f"Header Path: {r['metadata'].get('header_path')}")
        text_preview = r['text'].replace('\n', ' ')[:120]
        print(f"Preview: {text_preview}...\n")
        
    logger.info("=== FASE 2 COMPLETATA CON SUCCESSO ===")


if __name__ == "__main__":
    main()
