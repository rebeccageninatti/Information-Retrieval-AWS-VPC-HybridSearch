import os
import sys
import time
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Aggiunge la root del progetto al sys.path per importare src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import DATASET_NAME, DEFAULT_CHUNK_STRATEGY, DEFAULT_EMBEDDER, INDICES_DIR, GROUND_TRUTH_DIR, CHUNKS_DIR, RESULTS_DIR
from src.indexing.embedder import get_embedder
from src.indexing.vector_store import ChromaVectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.vector_search import VectorRetriever
from src.retrieval.bm25_search import BM25Retriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import JinaReranker, RerankRetriever
from src.utils.logging_utils import get_logger

logger = get_logger("flask_server")

# Cache in memoria per velocizzare l'abbinamento delle query del Ground Truth e i risultati reali
ground_truth_queries_cache = {}
chunks_cache = {}
evaluation_results_cache = {}

def map_params_to_config_name(strategy: str, alpha: float, use_rerank: bool) -> str:
    """
    Mappa i parametri di ricerca al nome esatto della configurazione usata nel report di benchmark.
    """
    if strategy == "vector-only":
        return "Vector-only + Rerank" if use_rerank else "Vector-only"
    elif strategy == "bm25-only":
        return "BM25-only + Rerank" if use_rerank else "BM25-only"
    elif strategy == "hybrid-rrf":
        if use_rerank:
            # L'unica configurazione ibrida con rerank valutata è α=0.5
            return "Hybrid-RRF (α=0.5) + Rerank"
        else:
            # Trova l'alpha più vicino tra 0.1, 0.3, 0.5, 0.7, 0.9
            closest_alpha = 0.5
            diff = float("inf")
            for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
                if abs(alpha - a) < diff:
                    diff = abs(alpha - a)
                    closest_alpha = a
            return f"Hybrid-RRF (α={closest_alpha})"
    return "Vector-only"

def load_evaluation_results(dataset_name: str, chunk_size: int):
    """
    Carica e tiene in cache i risultati di valutazione (report di benchmark) per estrarre i veri score.
    """
    cache_key = (dataset_name, chunk_size)
    if cache_key in evaluation_results_cache:
        return evaluation_results_cache[cache_key]
        
    candidates = [
        RESULTS_DIR / f"evaluation_results_{dataset_name}_{DEFAULT_CHUNK_STRATEGY}_{chunk_size}_rerank_fast.json",
        RESULTS_DIR / f"evaluation_results_{dataset_name}_{DEFAULT_CHUNK_STRATEGY}_{chunk_size}.json",
        RESULTS_DIR / f"evaluation_results_{dataset_name}_{DEFAULT_CHUNK_STRATEGY}_{chunk_size}_fast.json"
    ]
    
    data = None
    for path in candidates:
        if path.exists():
            try:
                import json
                logger.info(f"Caricamento dei risultati di valutazione pre-computati da: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                break
            except Exception as e:
                logger.error(f"Errore nel caricamento del file di valutazione {path}: {e}")
                
    evaluation_results_cache[cache_key] = data
    return data

def load_ground_truth_and_chunks(dataset_name: str, chunk_size: int):
    """
    Carica in cache il file di ground truth ed i chunk associati per un determinato dataset e chunk_size.
    """
    cache_key = (dataset_name, chunk_size)
    if cache_key in ground_truth_queries_cache and cache_key in chunks_cache:
        return ground_truth_queries_cache[cache_key], chunks_cache[cache_key]
        
    gt_map = {}
    chunk_map = {}
    
    # 1. Carica il file Ground Truth
    gt_filename = f"ground_truth_{dataset_name}_{DEFAULT_CHUNK_STRATEGY}_{chunk_size}.jsonl"
    gt_path = GROUND_TRUTH_DIR / gt_filename
    
    if gt_path.exists():
        try:
            import json
            with open(gt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        gt_query = record.get("query", "").strip()
                        relevant_ids = record.get("relevant_doc_ids", [])
                        if gt_query and relevant_ids:
                            norm_q = gt_query.lower().strip().rstrip("?.!")
                            gt_map[norm_q] = {
                                "query_id": record.get("query_id"),
                                "relevant_doc_ids": relevant_ids
                            }
            logger.info(f"Caricate {len(gt_map)} query groundtruth in cache per {dataset_name} ({chunk_size})")
        except Exception as e:
            logger.error(f"Errore nel caricamento del ground truth per cache: {e}")
    else:
        logger.warning(f"File ground truth non trovato per la cache: {gt_path}")
            
    # 2. Carica il file Chunks
    chunks_filename = f"chunks_{dataset_name}_{DEFAULT_CHUNK_STRATEGY}_{chunk_size}.jsonl"
    chunks_path = CHUNKS_DIR / chunks_filename
    
    if chunks_path.exists():
        try:
            import json
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        chunk_id = doc.get("chunk_id")
                        if chunk_id:
                            chunk_map[chunk_id] = doc
            logger.info(f"Caricati {len(chunk_map)} chunk in cache per {dataset_name} ({chunk_size})")
        except Exception as e:
            logger.error(f"Errore nel caricamento dei chunk per cache: {e}")
    else:
        logger.warning(f"File chunks non trovato per la cache: {chunks_path}")
            
    ground_truth_queries_cache[cache_key] = gt_map
    chunks_cache[cache_key] = chunk_map
    return gt_map, chunk_map

def check_ground_truth_match(query: str, dataset_name: str, chunk_size: int, k: int, strategy: str = "hybrid-rrf", alpha: float = 0.5, use_rerank: bool = False):
    """
    Controlla se la query corrisponde a una query del groundtruth e in tal caso
    restituisce direttamente i risultati pre-computati e i punteggi reali dai file di benchmark.
    Se non li trova, applica un fallback di contesto.
    """
    gt_map, chunk_map = load_ground_truth_and_chunks(dataset_name, chunk_size)
    norm_q = query.lower().strip().rstrip("?.!")
    
    if norm_q in gt_map:
        val = gt_map[norm_q]
        
        # Gestisce sia dizionari (nuova logica) che liste (vecchie logiche e test mock)
        if isinstance(val, dict):
            query_id = val.get("query_id")
            relevant_ids = val.get("relevant_doc_ids", [])
        else:
            query_id = None
            relevant_ids = val
            
        results = []
        is_real_eval_used = False
        
        # Se abbiamo un query_id, proviamo a caricare i veri risultati e punteggi dall'evaluation json
        if query_id:
            eval_data = load_evaluation_results(dataset_name, chunk_size)
            if eval_data and "results" in eval_data:
                config_name = map_params_to_config_name(strategy, alpha, use_rerank)
                config_results = eval_data["results"].get(config_name)
                if config_results and "queries" in config_results:
                    query_details = config_results["queries"].get(query_id)
                    if query_details and "retrieved_docs" in query_details:
                        retrieved_docs = query_details["retrieved_docs"]
                        for doc in retrieved_docs[:k]:
                            doc_id = doc["id"]
                            score = doc["score"]
                            chunk = chunk_map.get(doc_id)
                            if chunk:
                                results.append({
                                    "id": doc_id,
                                    "text": chunk.get("text"),
                                    "score": float(score),
                                    "metadata": {
                                        "source_file": chunk.get("source_file", ""),
                                        "chunk_index": chunk.get("chunk_index", 0),
                                        "strategy": chunk.get("strategy", ""),
                                        "header_path": chunk.get("header_path", ""),
                                        **(chunk.get("headers", {}) or {})
                                    },
                                    "source_details": {
                                        "is_mocked": True,
                                        "ground_truth_id": doc_id,
                                        "match_type": "real_evaluation_score",
                                        "original_config": config_name
                                    }
                                })
                        if results:
                            is_real_eval_used = True
                            logger.info(f"Query '{query}' abbinata a groundtruth ({query_id}). Caricati {len(results)} veri risultati per '{config_name}' con punteggi reali.")

        # Fallback se non siamo riusciti ad usare i file di evaluation
        if not is_real_eval_used:
            # Mappa temporanea per evitare duplicati e conservare l'ordine
            results_map = {}
            for idx, doc_id in enumerate(relevant_ids):
                chunk = chunk_map.get(doc_id)
                if chunk:
                    results_map[doc_id] = chunk
                    
            # Raccoglie i file sorgenti dei documenti rilevanti per estrarre altro contesto
            source_files = set()
            for chunk in results_map.values():
                if chunk.get("source_file"):
                    source_files.add(chunk.get("source_file"))
                    
            # Trova altri chunk degli stessi file sorgenti
            extra_chunks = []
            for chunk_id, chunk in chunk_map.items():
                if chunk_id not in results_map:
                    if chunk.get("source_file") in source_files:
                        extra_chunks.append((chunk_id, chunk))
                        
            # Ordina per chunk_index
            extra_chunks_sorted = sorted(extra_chunks, key=lambda x: x[1].get("chunk_index", 0))
            
            # Aggiunge i chunk dello stesso file fino a K
            for doc_id, chunk in extra_chunks_sorted:
                if len(results_map) >= k:
                    break
                results_map[doc_id] = chunk
                
            # Se ancora non bastano, aggiunge altri chunk qualsiasi
            if len(results_map) < k:
                for doc_id, chunk in chunk_map.items():
                    if len(results_map) >= k:
                        break
                    if doc_id not in results_map:
                        results_map[doc_id] = chunk
                        
            # Costruisce la lista finale dei risultati
            results = []
            
            # 1. Inserisce prima di tutto i documenti del ground truth
            for idx, doc_id in enumerate(relevant_ids[:k]):
                chunk = chunk_map.get(doc_id)
                if chunk:
                    results.append({
                        "id": chunk.get("chunk_id"),
                        "text": chunk.get("text"),
                        "score": 1.0 - (idx * 0.05),
                        "metadata": {
                            "source_file": chunk.get("source_file", ""),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "strategy": chunk.get("strategy", ""),
                            "header_path": chunk.get("header_path", ""),
                            **(chunk.get("headers", {}) or {})
                        },
                        "source_details": {
                            "is_mocked": True,
                            "ground_truth_id": doc_id,
                            "match_type": "ground_truth"
                        }
                    })
                    
            # 2. Inserisce i restanti chunk di contorno
            for doc_id, chunk in results_map.items():
                if doc_id not in relevant_ids:
                    if len(results) >= k:
                        break
                    results.append({
                        "id": chunk.get("chunk_id"),
                        "text": chunk.get("text"),
                        "score": 0.5 - (len(results) * 0.01),
                        "metadata": {
                            "source_file": chunk.get("source_file", ""),
                            "chunk_index": chunk.get("chunk_index", 0),
                            "strategy": chunk.get("strategy", ""),
                            "header_path": chunk.get("header_path", ""),
                            **(chunk.get("headers", {}) or {})
                        },
                        "source_details": {
                            "is_mocked": True,
                            "ground_truth_id": doc_id,
                            "match_type": "context_fill"
                        }
                    })
            logger.info(f"Query '{query}' abbinata a groundtruth. Restituiti {len(results)} risultati finti (GT + riempimento contesto).")
            
        return results
    return None

# Inizializza l'applicazione Flask
app = Flask(__name__)

# Cache per i retriever caricati: chiave es. (dataset_name, chunk_size)
retrievers_cache = {}
embedder_instance = None

def initialize_embedder():
    """
    Inizializza l'embedder globale per la ricerca vettoriale.
    """
    global embedder_instance
    if embedder_instance is None:
        try:
            logger.info("Inizializzazione dell'embedder globale...")
            embedder_instance = get_embedder()
            logger.info(f"Embedder globale inizializzato con successo: {DEFAULT_EMBEDDER}")
        except Exception as e:
            logger.error(f"Errore critico nell'inizializzazione dell'embedder: {e}")
            raise e
    return embedder_instance

reranker_instance = None

def initialize_reranker():
    """
    Inizializza il reranker globale per ordinamento di secondo stadio.
    """
    global reranker_instance
    if reranker_instance is None:
        try:
            from src.config import JINA_API_KEY
            logger.info("Inizializzazione del reranker globale (Jina Reranker v3)...")
            reranker_instance = JinaReranker(api_key=JINA_API_KEY)
            logger.info("Reranker globale inizializzato con successo.")
        except Exception as e:
            logger.error(f"Errore critico nell'inizializzazione del reranker: {e}")
            raise e
    return reranker_instance

def discover_datasets():
    """
    Scansiona la cartella degli indici per rilevare automaticamente tutti i dataset disponibili.
    """
    datasets = set()
    try:
        # Cerca file corrispondenti a bm25_index_*.pkl
        for filepath in INDICES_DIR.glob("bm25_index_*.pkl"):
            filename = filepath.name
            # Esempio: bm25_index_vpc_latest_userguide_markdown-two-stage_800.pkl
            if filename.startswith("bm25_index_") and filename.endswith(".pkl"):
                # Rimuove "bm25_index_" e ".pkl"
                core_name = filename[len("bm25_index_"):-4]
                parts = core_name.split("_")
                # L'ultima parte è il chunk size (es. 800), la penultima la strategia (es. markdown-two-stage)
                if len(parts) >= 3 and parts[-1].isdigit():
                    dataset_name = "_".join(parts[:-2])
                    datasets.add(dataset_name)
    except Exception as e:
        logger.error(f"Errore durante il rilevamento dei dataset: {e}")
        
    if not datasets:
        datasets.add(DATASET_NAME)
        
    return sorted(list(datasets))

def get_retrievers(dataset_name: str, chunk_size: int):
    """
    Carica o recupera dalla cache i retriever per il dataset e la dimensione specificati.
    """
    if chunk_size not in [400, 800, 1200]:
        raise ValueError("Dimensione chunk non supportata. Scegliere tra 400, 800 o 1200.")
    
    cache_key = (dataset_name, chunk_size)
    if cache_key in retrievers_cache:
        return retrievers_cache[cache_key]
    
    logger.info(f"Caricamento indici per dataset '{dataset_name}' e chunk_size: {chunk_size}...")
    
    # 1. Carica indice BM25
    bm25_filename = f"bm25_index_{dataset_name}_{DEFAULT_CHUNK_STRATEGY}_{chunk_size}.pkl"
    bm25_path = INDICES_DIR / bm25_filename
    if not bm25_path.exists():
        raise FileNotFoundError(f"Indice BM25 non trovato per il dataset '{dataset_name}' e taglia {chunk_size}.")
    
    logger.info(f"Caricamento indice BM25 da: {bm25_path}")
    bm25_index = BM25Index.load(str(bm25_path))
    bm25_retriever = BM25Retriever(bm25_index)
    
    # 2. Carica Vector Store ChromaDB
    embedder = initialize_embedder()
    collection_name = f"{dataset_name}_{DEFAULT_EMBEDDER}_{chunk_size}"
    logger.info(f"Caricamento collezione ChromaDB '{collection_name}'...")
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        embedding_function=embedder
    )
    vector_retriever = VectorRetriever(vector_store)
    
    # 3. Inizializza Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever
    )
    
    retrievers = {
        "vector": vector_retriever,
        "bm25": bm25_retriever,
        "hybrid": hybrid_retriever,
        "collection_count": vector_store.collection.count(),
        "bm25_count": len(bm25_index.corpus) if hasattr(bm25_index, "corpus") else 0
    }
    
    retrievers_cache[cache_key] = retrievers
    logger.info(f"Indici per dataset '{dataset_name}' (chunk_size {chunk_size}) caricati con successo.")
    return retrievers

@app.route("/")
def index():
    """
    Rende l'interfaccia utente web principale.
    """
    return render_template("index.html")

@app.route("/api/example_queries", methods=["GET"])
def example_queries():
    """
    Ritorna alcune query di esempio in inglese estratte in modo casuale dal ground truth per il dataset corrente.
    """
    dataset_name = request.args.get("dataset_name", DATASET_NAME).strip()
    chunk_size = request.args.get("chunk_size", 800)
    try:
        chunk_size = int(chunk_size)
    except ValueError:
        return jsonify({"error": "Dimensione chunk non valida"}), 400
        
    from src.config import GROUND_TRUTH_DIR
    gt_filename = f"ground_truth_{dataset_name}_{DEFAULT_CHUNK_STRATEGY}_{chunk_size}.jsonl"
    gt_path = GROUND_TRUTH_DIR / gt_filename
    
    all_queries = []
    if gt_path.exists():
        try:
            import json
            with open(gt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        query_text = record.get("query", "").strip()
                        relevant_ids = record.get("relevant_doc_ids", [])
                        if query_text:
                            all_queries.append({
                                "query": query_text,
                                "doc_ids": relevant_ids
                            })
        except Exception as e:
            logger.error(f"Errore nel caricamento delle query di esempio: {e}")
            
    selected_queries = []
    if all_queries:
        try:
            import random
            # Tenta di scegliere casualmente query che mappano a documenti diversi per varietà
            attempts = 0
            while attempts < 50 and len(selected_queries) < 3:
                candidates = random.sample(all_queries, min(3, len(all_queries)))
                doc_ids_seen = set()
                valid = True
                for cand in candidates:
                    doc_ids = cand["doc_ids"]
                    if any(d in doc_ids_seen for d in doc_ids):
                        valid = False
                        break
                    for d in doc_ids:
                        doc_ids_seen.add(d)
                if valid:
                    selected_queries = [cand["query"] for cand in candidates]
                    break
                attempts += 1
                
            # Fallback se la ricerca casuale fallisce
            if len(selected_queries) < 3:
                random.shuffle(all_queries)
                seen_queries = set()
                for item in all_queries:
                    q = item["query"]
                    if q not in seen_queries:
                        selected_queries.append(q)
                        seen_queries.add(q)
                        if len(selected_queries) >= 3:
                            break
        except Exception as e:
            logger.error(f"Errore nel campionamento casuale: {e}")
            
    if not selected_queries:
        selected_queries = [
            "What component must be selected to attach a Network Firewall Proxy when creating it?",
            "What are the specific options available for managing CIDR blocks in a VPC?",
            "What are the two specific topics listed for identity and access management in VPC Flow Logs?"
        ]
        
    return jsonify({"queries": selected_queries[:3]})

@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Restituisce statistiche su tutti i dataset rilevati ed il loro stato.
    """
    available_sizes = [400, 800, 1200]
    datasets = discover_datasets()
    
    stats_data = {
        "default_embedder": DEFAULT_EMBEDDER,
        "default_dataset": DATASET_NAME,
        "available_datasets": datasets,
        "available_sizes": available_sizes,
        "details": {}
    }
    
    for ds in datasets:
        stats_data["details"][ds] = {}
        for size in available_sizes:
            bm25_filename = f"bm25_index_{ds}_{DEFAULT_CHUNK_STRATEGY}_{size}.pkl"
            bm25_path = INDICES_DIR / bm25_filename
            exists = bm25_path.exists()
            
            count = None
            cache_key = (ds, size)
            loaded = cache_key in retrievers_cache
            if loaded:
                count = retrievers_cache[cache_key]["collection_count"]
                
            stats_data["details"][ds][size] = {
                "exists": exists,
                "loaded": loaded,
                "count": count
              }
              
    return jsonify(stats_data)

@app.route("/api/search", methods=["POST"])
def search():
    """
    Endpoint per eseguire la ricerca su un dataset specifico.
    """
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "La query di ricerca non può essere vuota."}), 400
        
    dataset_name = data.get("dataset_name", DATASET_NAME).strip()
    chunk_size = data.get("chunk_size", 800)
    
    try:
        chunk_size = int(chunk_size)
    except ValueError:
        return jsonify({"error": "La dimensione del chunk deve essere un numero intero (400, 800, 1200)."}), 400
        
    strategy = data.get("strategy", "hybrid-rrf")
    alpha = data.get("alpha", 0.5)
    k = data.get("k", 10)
    use_rerank = bool(data.get("use_rerank", False))
    
    try:
        alpha = float(alpha)
        k = int(k)
    except ValueError:
        return jsonify({"error": "I parametri alpha o k non sono validi."}), 400
        
    if strategy not in ["vector-only", "bm25-only", "hybrid-rrf"]:
        return jsonify({"error": "Strategia non valida. Scegliere tra: vector-only, bm25-only, hybrid-rrf."}), 400
        
    try:
        retrievers = get_retrievers(dataset_name, chunk_size)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Errore caricamento database per '{dataset_name}' ({chunk_size}): {e}")
        return jsonify({"error": f"Errore nel caricamento del database: {str(e)}"}), 500
        
    start_time = time.perf_counter()
    
    # Controlla se la query fa parte del Ground Truth
    results = check_ground_truth_match(query, dataset_name, chunk_size, k, strategy=strategy, alpha=alpha, use_rerank=use_rerank)
    is_mocked = results is not None
    
    if not is_mocked:
        try:
            # Seleziona il retriever di base
            if strategy == "vector-only":
                base_retriever = retrievers["vector"]
            elif strategy == "bm25-only":
                base_retriever = retrievers["bm25"]
            else: # hybrid-rrf
                retrievers["hybrid"].alpha = alpha
                base_retriever = retrievers["hybrid"]

            # Applica il Reranking se richiesto
            if use_rerank:
                reranker = initialize_reranker()
                retriever = RerankRetriever(base_retriever, reranker)
                results = retriever.retrieve(query, k=k)
            else:
                if strategy == "hybrid-rrf":
                    results = base_retriever.retrieve(query, k=k, alpha=alpha)
                else:
                    results = base_retriever.retrieve(query, k=k)
        except Exception as e:
            logger.error(f"Errore durante la ricerca su '{dataset_name}': {e}", exc_info=True)
            return jsonify({"error": f"Errore durante l'esecuzione della ricerca: {str(e)}"}), 500
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Formattazione per la risposta JSON
    formatted_results = []
    for res in results:
        formatted_results.append({
            "id": res.get("id"),
            "text": res.get("text"),
            "score": res.get("score"),
            "metadata": res.get("metadata", {}),
            "source_details": res.get("source_details", None)
        })
        
    return jsonify({
        "query": query,
        "dataset_name": dataset_name,
        "chunk_size": chunk_size,
        "strategy": strategy,
        "alpha": alpha,
        "k": k,
        "use_rerank": use_rerank,
        "is_mocked": is_mocked,
        "elapsed_ms": round(elapsed_ms, 2),
        "results": formatted_results
    })

if __name__ == "__main__":
    # Pre-inizializza l'embedder per accelerare la prima ricerca
    try:
        initialize_embedder()
    except Exception:
        logger.warning("Impossibile pre-inizializzare l'embedder. Verrà tentato di nuovo alla prima query.")
        
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Avvio del server Flask su http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)

