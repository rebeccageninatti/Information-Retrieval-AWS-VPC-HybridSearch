#!/usr/bin/env python
"""
Script CLI per avviare la generazione sintetica del Ground Truth tramite Gemini.
"""

import argparse
import sys
from src.config import EVAL_LLM_MODEL, EVAL_NUM_SOURCE_CHUNKS, EVAL_NUM_MULTIHOP_QUERIES
from src.evaluation.ground_truth import GroundTruthGenerator
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    """
    Funzione principale che gestisce il parsing degli argomenti da CLI ed esegue il generatore.
    """
    parser = argparse.ArgumentParser(
        description="Genera un dataset di Ground Truth sintetico per la valutazione di sistemi di retrieval."
    )
    
    # Parametri richiesti configurati all'interno di main() come richiesto dall'utente
    parser.add_argument(
        "--model",
        type=str,
        default=EVAL_LLM_MODEL,
        help=f"Nome del modello Gemini/Gemma da utilizzare per la generazione (default: {EVAL_LLM_MODEL})."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=EVAL_NUM_SOURCE_CHUNKS,
        help=f"Numero di chunk singoli da campionare per generare 4 query per ciascuno (default: {EVAL_NUM_SOURCE_CHUNKS})."
    )
    parser.add_argument(
        "--multihop-samples",
        type=int,
        default=EVAL_NUM_MULTIHOP_QUERIES,
        help=f"Numero di coppie di chunk da campionare per generare query multi-hop (default: {EVAL_NUM_MULTIHOP_QUERIES})."
    )

    args = parser.parse_args()

    logger.info("Avvio dello script di generazione Ground Truth...")
    logger.info(f"Parametri configurati - Modello: {args.model}, Campioni Single-Chunk: {args.samples}, Campioni Multi-Hop: {args.multihop_samples}")

    try:
        generator = GroundTruthGenerator(
            model_name=args.model,
            samples=args.samples,
            multihop_samples=args.multihop_samples
        )
        records = generator.generate()
        logger.info(f"Esecuzione completata correttamente. Generati {len(records)} record totali.")
    except Exception as e:
        logger.critical(f"Errore fatale durante l'esecuzione: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
