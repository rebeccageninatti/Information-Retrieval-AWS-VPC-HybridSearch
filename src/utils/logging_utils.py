"""
Utility di logging strutturato per il progetto.

Uso:
    from src.utils.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("Messaggio di log")
"""

import logging
import sys

from src.config import LOG_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Crea e configura un logger con formato e livello standard.

    Args:
        name: Nome del logger (tipicamente __name__ del modulo chiamante).

    Returns:
        Logger configurato.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    return logger
