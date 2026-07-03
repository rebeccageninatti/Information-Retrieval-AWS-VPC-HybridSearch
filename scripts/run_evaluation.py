#!/usr/bin/env python
"""
Script CLI per eseguire la valutazione comparativa (Fase 4: Benchmark).
"""

import sys
from src.evaluation.benchmark import BenchmarkRunner
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== INIZIO FASE 4: RUN DI VALUTAZIONE COMPARATIVA ===")
    
    try:
        runner = BenchmarkRunner()
        runner.execute_all()
        logger.info("=== FASE 4 COMPLETATA CON SUCCESSO ===")
        
    except Exception as e:
        logger.critical(f"Errore durante l'esecuzione del benchmark: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
