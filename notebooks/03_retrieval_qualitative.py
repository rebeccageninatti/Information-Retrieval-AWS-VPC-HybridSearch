"""
Script per verificare qualitativamente le strategie di retrieval (Fase 3):
1. Carica l'indice BM25 e ChromaDB.
2. Inizializza VectorRetriever, BM25Retriever e HybridRetriever.
3. Esegue alcune query di esempio confrontando i risultati delle tre strategie.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

from src.config import (
    CHROMA_COLLECTION_NAME,
    BM25_INDEX_FILE
)
from src.utils.logging_utils import get_logger
from src.indexing import ChromaVectorStore, BM25Index, get_embedder

from src.retrieval.vector_search import VectorRetriever
from src.retrieval.bm25_search import BM25Retriever
from src.retrieval.hybrid_search import HybridRetriever

logger = get_logger(__name__)





def print_results(title: str, results: List[Dict[str, Any]], limit: int = 3):
    """
    Stampa in modo leggibile i risultati di una strategia.
    """
    print(f"\n--- {title} (Primi {limit} risultati) ---")
    if not results:
        print("Nessun risultato trovato.")
        return

    for idx, r in enumerate(results[:limit]):
        # Rileva dettagli di sorgente se presenti (per ibrido)
        details_str = ""
        if "source_details" in r:
            v_rank = r["source_details"]["vector_rank"]
            b_rank = r["source_details"]["bm25_rank"]
            details_str = f" [Vector Rank: {v_rank}, BM25 Rank: {b_rank}]"

        print(f"{idx+1}. ID: {r['id']} (Score: {r['score']:.5f}){details_str}")
        print(f"   Header Path: {r['metadata'].get('header_path', 'N/A')}")
        # Preview del testo (primi 100 caratteri in una linea sola)
        preview = r['text'].replace('\n', ' ')[:100]
        print(f"   Preview: {preview}...\n")


def main():
    logger.info("=== INIZIO TEST QUALITATIVO RETRIEVAL (FASE 3) ===")

    # 1. Caricamento Indice BM25
    bm25_file = BM25_INDEX_FILE
    if not bm25_file.exists():
        logger.error(f"Indice BM25 non trovato su: {bm25_file}. Esegui prima scripts/run_indexing.py!")
        sys.exit(1)
 
    logger.info(f"Caricamento indice BM25 da {bm25_file}...")
    bm25_index = BM25Index.load(str(bm25_file))

    # 2. Caricamento Vector Store ChromaDB
    try:
        embedder = get_embedder()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    collection_name = CHROMA_COLLECTION_NAME
    logger.info(f"Caricamento Vector Store ChromaDB, collezione: {collection_name}...")
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        embedding_function=embedder
    )

    # 3. Inizializzazione dei Retriever
    vector_retriever = VectorRetriever(vector_store)
    bm25_retriever = BM25Retriever(bm25_index)
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        alpha=0.5
    )

    # 4. Definizione delle query di test
    queries = [
        # Query lessicale diretta (Needle)
        "What is the maximum number of VPCs allowed in an AWS region?",
        # Query semantica / parafrasata
        "limit of virtual private clouds per region in Amazon Web Services",
        # Query concettuale
        "how does routing work between public and private subnets?",
        # Query ambigua
        "VPC security"
    ]

    # 5. Esecuzione delle query di test
    for q_idx, query in enumerate(queries, 1):
        print("\n" + "="*80)
        print(f"QUERY {q_idx}: '{query}'")
        print("="*80)

        # Vector search
        v_results = vector_retriever.retrieve(query, k=3)
        print_results("VECTOR-ONLY (Semantica)", v_results)

        # BM25 search
        b_results = bm25_retriever.retrieve(query, k=3)
        print_results("BM25-ONLY (Lessicale)", b_results)

        # Hybrid search alpha = 0.5
        h_results = hybrid_retriever.retrieve(query, k=3, alpha=0.5)
        print_results("HYBRID (RRF, alpha=0.5)", h_results)

    logger.info("=== TEST QUALITATIVO COMPLETATO ===")


if __name__ == "__main__":
    main()
