"""
Configurazione centralizzata del progetto.

Tutti i percorsi, parametri e impostazioni sono definiti qui.
Nessun hard-coding di valori nel resto del codice.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# ============================================================
# Percorsi del progetto
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
# Carica le variabili d'ambiente dal file .env nella root del progetto
load_dotenv(PROJECT_ROOT / ".env")

# Opt-out ChromaDB telemetry to prevent console warnings
os.environ["CHROMA_TELEMETRY_NOUSER"] = "True"

DATA_DIR = PROJECT_ROOT / "data"

# Sottocartelle dati
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
RESULTS_DIR = DATA_DIR / "results"
INDICES_DIR = DATA_DIR / "indices"


# Sottocartella da elaborare/indicizzare per i test (es: "vpc" per tutto, "vpc/latest/userguide" per un subset)
DATASET_SUBDIR = "vpc/latest"
DATASET_NAME = DATASET_SUBDIR.replace("/", "_")  # Nome del dataset (es: 'vpc_latest_userguide')



# ChromaDB
CHROMA_DB_DIR = INDICES_DIR / "chroma_db"

# ============================================================
# Parametri di Chunking
# ============================================================

# FISSATO FASE 1: Dimensione e strategia di chunking congelate dopo l'analisi
CHUNK_SIZE_DEFAULT = 800
CHUNK_SIZES = [800]
DEFAULT_CHUNK_STRATEGY = "markdown-two-stage"

# CONGELATO: overlap come percentuale della dimensione del chunk (10%)
CHUNK_OVERLAP_PCT = 0.10
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # CONGELATO: separatori standard

# ============================================================
# Parametri di Retrieval
# ============================================================

DEFAULT_TOP_K = 10        # Numero di risultati da ritornare
RRF_K = 60                # CONGELATO: costante k per RRF (standard dal paper originale)
DEFAULT_ALPHA = 0.5       # Bilanciamento di default tra ricerca lessicale (0.0) e vettoriale (1.0)

# VARIABILE: strategie di retrieval da testare
RETRIEVAL_STRATEGIES = ["vector-only", "bm25-only", "hybrid-rrf"]

# ============================================================
# Parametri di Valutazione
# ============================================================

EVAL_K_VALUES = [3, 5, 10]  # Valori di K per Precision@K, Recall@K, nDCG@K, MAP@K

# Configurazione del modello LLM per la generazione sintetica delle query (Fase 4)
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "gemini-3.1-flash-lite")
EVAL_NUM_SOURCE_CHUNKS = int(os.getenv("EVAL_NUM_SOURCE_CHUNKS", "100"))
EVAL_NUM_MULTIHOP_QUERIES = int(os.getenv("EVAL_NUM_MULTIHOP_QUERIES", "15"))

# Selezione dell'embedder di default: "jina" (Google Gemini rimosso)
DEFAULT_EMBEDDER = "jina"

# Percorsi dei file e collezioni (legati alla strategia, dimensione del chunk e dataset specifico)
CHUNKS_FILE = CHUNKS_DIR / f"chunks_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_{CHUNK_SIZE_DEFAULT}.jsonl"
BM25_INDEX_FILE = INDICES_DIR / f"bm25_index_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_{CHUNK_SIZE_DEFAULT}.pkl"
EMBEDDINGS_CACHE_FILE = EMBEDDINGS_DIR / f"embeddings_cache_{DEFAULT_EMBEDDER}_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_{CHUNK_SIZE_DEFAULT}.pkl"
CHROMA_COLLECTION_NAME = f"{DATASET_NAME}_{DEFAULT_EMBEDDER}_{CHUNK_SIZE_DEFAULT}"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / f"ground_truth_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_{CHUNK_SIZE_DEFAULT}.jsonl"
EVAL_RESULTS_FILE = RESULTS_DIR / f"evaluation_results_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_{CHUNK_SIZE_DEFAULT}.json"

# ============================================================
# Embedding & API Settings
# ============================================================

import os

# Jina AI Embeddings Config
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBEDDING_MODEL = "jina-embeddings-v5-text-small"
JINA_EMBEDDING_DIMENSION = 1024
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_BATCH_SIZE = 32
JINA_RPM_LIMIT = 100
JINA_TPM_LIMIT = 100000

# Jina AI Reranker Config
JINA_RERANK_MODEL = "jina-reranker-v3"
JINA_RERANK_API_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_RPM_LIMIT = 100
JINA_RERANK_TPM_LIMIT = 100000
JINA_RERANK_DEFAULT_K = 25


# Google Gemini Config (per generazione sintetica query con gemini-3.1-flash-lite)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "15"))  # Limite RPM free tier (15)
GEMINI_TPM_LIMIT = int(os.getenv("GEMINI_TPM_LIMIT", "32000"))  # Limite TPM
GEMINI_DAILY_LIMIT = int(os.getenv("GEMINI_DAILY_LIMIT", "1000"))  # Limite giornaliero

# Assegnazione del modello e delle dimensioni di embedding (Jina AI)
EMBEDDING_MODEL = JINA_EMBEDDING_MODEL
EMBEDDING_DIMENSION = JINA_EMBEDDING_DIMENSION

# ============================================================
# Logging
# ============================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
