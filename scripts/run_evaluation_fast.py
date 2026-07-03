"""
Script CLI per eseguire la valutazione comparativa rapida (Fase 4: Benchmark Fast).
Pre-calcola tutti gli embedding delle query in batch per azzerare i tempi di rete.
"""

import sys
import time
from src.evaluation.benchmark import BenchmarkRunner
from src.indexing.embedder import get_embedder
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== INIZIO FASE 4: RUN DI VALUTAZIONE COMPARATIVA RAPIDA (FAST) ===")
    
    try:
        runner = BenchmarkRunner(is_fast_mode=True)
        
        # 1. Recupera l'embedder per pre-caricare gli embedding
        logger.info("Pre-calcolo in batch degli embedding di tutte le query...")
        embedder = get_embedder()
        
        # Estrae tutte le query dal ground truth
        queries = [record["query"] for record in runner.ground_truth]
        
        # Chiama l'embedder passandogli tutte le query.
        # L'embedder le elaborerà a lotti (batch_size=32) salvandole in cache
        start_embed_time = time.perf_counter()
        embedder(queries)
        elapsed_embed = time.perf_counter() - start_embed_time
        logger.info(f"Pre-calcolo completato in {elapsed_embed:.2f} secondi. Avvio del benchmark...")
        
        # 2. Esegue il benchmark
        runner.execute_all()
        logger.info("=== FASE 4 (FAST) COMPLETATA CON SUCCESSO ===")
        
    except Exception as e:
        logger.critical(f"Errore durante l'esecuzione del benchmark rapido: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
