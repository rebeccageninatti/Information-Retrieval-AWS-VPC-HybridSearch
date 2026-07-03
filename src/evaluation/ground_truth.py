"""
Modulo per la generazione sintetica di Ground Truth tramite modelli LLM Gemini/Gemma.
Genera query Needle, Paraphrase, Conceptual, Ambiguous e Multi-hop con rate-limiting e output strutturato.
"""

import os
import json
import random
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import (
    CHUNKS_FILE,
    GROUND_TRUTH_FILE,
    GEMINI_API_KEY,
    GEMINI_RPM_LIMIT,
    GEMINI_TPM_LIMIT,
    GEMINI_DAILY_LIMIT,
    DATASET_NAME,
    GROUND_TRUTH_DIR,
    DEFAULT_CHUNK_STRATEGY,
    CHUNK_SIZE_DEFAULT
)
from src.indexing.embedder import TokenRateLimiter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================
# Schemi Pydantic per output strutturati
# ============================================================

class SingleChunkQueries(BaseModel):
    needle_query: str = Field(description="A highly specific question targeting precise parameters, limits, or configurations in the text.")
    paraphrase_query: str = Field(description="An equivalent question to the needle_query but formulated with different words and syntax to avoid exact keyword matches.")
    conceptual_query: str = Field(description="A high-level conceptual question focused on the 'how' or 'why' of a mechanism described in the text.")
    ambiguous_query: str = Field(description="A short, vague search-like query consisting of only 1-3 keywords from the text.")


class MultiHopQueries(BaseModel):
    multi_hop_query: str = Field(description="A complex question that requires connecting and combining information from both provided texts to answer.")


# ============================================================
# Generatore di Ground Truth
# ============================================================

class GroundTruthGenerator:
    """
    Classe per campionare chunk e generare query di test sintetiche tramite Gemini API.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.1-flash-lite",
        samples: int = 100,
        multihop_samples: int = 15,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.samples = samples
        self.multihop_samples = multihop_samples
        self.api_key = api_key or GEMINI_API_KEY

        if not self.api_key:
            raise ValueError(
                "API Key di Gemini non trovata! Assicurati che GEMINI_API_KEY sia impostata "
                "nel file .env o passata al costruttore."
            )

        # Inizializza il client ufficiale google-genai
        self.client = genai.Client(api_key=self.api_key)

        # Inizializza il rate limiter riutilizzando TokenRateLimiter (15 RPM)
        # Limite TPM e limite giornaliero da configurazione
        self.rate_limiter = TokenRateLimiter(
            rpm_limit=GEMINI_RPM_LIMIT,
            tpm_limit=GEMINI_TPM_LIMIT,
            daily_limit=GEMINI_DAILY_LIMIT
        )

        logger.info(
            f"GroundTruthGenerator inizializzato con modello: {model_name}. "
            f"Target: {samples} single-chunk, {multihop_samples} multi-hop."
        )

    def load_chunks(self) -> List[Dict[str, Any]]:
        """
        Carica i chunk dal file JSONL generato in Fase 1.
        """
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(f"File dei chunk non trovato: {CHUNKS_FILE}. Esegui prima l'ingestion.")

        chunks = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        logger.info(f"Caricati {len(chunks)} chunk da {CHUNKS_FILE.name}")
        return chunks

    def sample_chunks_round_robin(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Campiona i chunk in modalità Round-Robin tra i diversi file sorgente
        per garantire la massima diversità tematica.
        """
        # Raggruppa i chunk per file sorgente
        chunks_by_file = defaultdict(list)
        for chunk in chunks:
            source = chunk.get("source_file", "unknown.md")
            chunks_by_file[source].append(chunk)

        # Rimescola i file e l'ordine dei chunk all'interno di ciascun file
        files = list(chunks_by_file.keys())
        random.seed(42)  # Per riproducibilità
        random.shuffle(files)
        for file in files:
            random.shuffle(chunks_by_file[file])

        sampled_chunks: List[Dict[str, Any]] = []
        file_idx = 0
        
        while len(sampled_chunks) < self.samples and files:
            current_file = files[file_idx % len(files)]
            if chunks_by_file[current_file]:
                # Estrae il primo chunk disponibile per il file corrente
                sampled_chunks.append(chunks_by_file[current_file].pop(0))
            else:
                # Se il file ha esaurito i chunk, lo rimuove dal pool
                files.remove(current_file)
                if not files:
                    break
                continue
            file_idx += 1

        logger.info(
            f"Campionati {len(sampled_chunks)} chunk da {len(set(c['source_file'] for c in sampled_chunks))} file diversi "
            f"(su un totale di {len(chunks_by_file)} file disponibili)."
        )
        return sampled_chunks

    def sample_multihop_pairs(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Seleziona coppie di chunk adiacenti/correlati nello stesso file per le query multi-hop.
        """
        # Raggruppa i chunk per file sorgente per mantenere l'ordine originale (chunk_index)
        chunks_by_file = defaultdict(list)
        for chunk in chunks:
            source = chunk.get("source_file", "unknown.md")
            chunks_by_file[source].append(chunk)

        # Ordina per chunk_index all'interno di ciascun file
        for file in chunks_by_file:
            chunks_by_file[file].sort(key=lambda x: x.get("chunk_index", 0))

        # Filtra i file che hanno almeno due chunk
        candidate_files = [f for f, chs in chunks_by_file.items() if len(chs) >= 2]
        
        random.seed(43)  # Per riproducibilità
        random.shuffle(candidate_files)

        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for file in candidate_files[:self.multihop_samples]:
            # Sceglie un indice di partenza casuale per la coppia
            chs = chunks_by_file[file]
            idx = random.randint(0, len(chs) - 2)
            pairs.append((chs[idx], chs[idx + 1]))

        logger.info(f"Selezionate {len(pairs)} coppie di chunk per query multi-hop.")
        return pairs

    def _generate_with_retry(self, prompt: str, schema: Any, max_retries: int = 5) -> Any:
        """
        Esegue la chiamata all'LLM gestendo rate limit (RPM/TPM) e retry in caso di errore 429.
        """
        # Stima dei token (1 token per 4 caratteri per input, 500 per output schema)
        estimated_tokens = (len(prompt) // 4) + 500
        
        # Controlla ed attende se superiamo i limiti temporali
        self.rate_limiter.check_and_wait(estimated_tokens)

        backoff = 2.0
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.7,
                    )
                )
                
                # Registra la transazione nel rate limiter
                self.rate_limiter.record_request(estimated_tokens)
                
                return response.parsed
                
            except Exception as e:
                # Controlla se l'errore è un rate limit (tipicamente contiene '429' o 'RESOURCE_EXHAUSTED')
                error_str = str(e)
                if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "Limit" in error_str) and attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limit API riscontrato (Tentativo {attempt + 1}/{max_retries}). "
                        f"In attesa di {backoff}s per backoff..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error(f"Errore durante la generazione dell'LLM: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(backoff)
                    backoff *= 1.5

        raise RuntimeError("Impossibile generare contenuto dopo i tentativi massimi.")

    def generate(self) -> List[Dict[str, Any]]:
        """
        Esegue l'intero workflow di generazione del Ground Truth.
        """
        chunks = self.load_chunks()
        
        # 1. Campionamento
        single_chunks = self.sample_chunks_round_robin(chunks)
        multihop_pairs = self.sample_multihop_pairs(chunks)

        ground_truth_records: List[Dict[str, Any]] = []
        query_counter = 1

        # Dizionari per mappare i chunk alle query esistenti per fare riutilizzo
        existing_single_queries = {}  # chunk_id -> list of records
        existing_multihop_queries = {}  # tuple of sorted (chunk_id_1, chunk_id_2) -> record

        # Rileva dinamicamente quali sottocartelle sono presenti nei chunk campionati
        subdirs = set()
        for chunk in single_chunks:
            parts = Path(chunk.get("source_file", "")).parts
            if len(parts) > 1:
                subdirs.add(parts[0])
        for c1, c2 in multihop_pairs:
            for c in [c1, c2]:
                parts = Path(c.get("source_file", "")).parts
                if len(parts) > 1:
                    subdirs.add(parts[0])

        for subdir_name in subdirs:
            # Costruiamo il nome del file di ground truth atteso per questa sottocartella
            subdir_gt_file = GROUND_TRUTH_DIR / f"ground_truth_{DATASET_NAME}_{subdir_name}_{DEFAULT_CHUNK_STRATEGY}_{CHUNK_SIZE_DEFAULT}.jsonl"
            if subdir_gt_file.exists():
                logger.info(f"Caricamento delle query di ground truth per '{subdir_name}' da {subdir_gt_file.name} per riutilizzo...")
                try:
                    with open(subdir_gt_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                rec = json.loads(line)
                                if rec["category"] == "multi-hop":
                                    c1 = rec["source_metadata"]["chunk_id_1"]
                                    c2 = rec["source_metadata"]["chunk_id_2"]
                                    pair_key = tuple(sorted([c1, c2]))
                                    existing_multihop_queries[pair_key] = rec
                                else:
                                    cid = rec["relevant_doc_ids"][0]
                                    if cid not in existing_single_queries:
                                        existing_single_queries[cid] = []
                                    existing_single_queries[cid].append(rec)
                except Exception as e:
                    logger.warning(f"Errore nel caricamento delle query di ground truth per '{subdir_name}': {e}")
                    
        if existing_single_queries or existing_multihop_queries:
            logger.info(f"Caricate in totale {len(existing_single_queries)} query a chunk singolo e {len(existing_multihop_queries)} query multi-hop riutilizzabili.")

        # Assicura che la directory di destinazione esista
        os.makedirs(os.path.dirname(GROUND_TRUTH_FILE), exist_ok=True)

        # 2. Generazione query Single-Chunk (Needle, Paraphrase, Conceptual, Ambiguous)
        logger.info("Avvio generazione query a chunk singolo...")
        for i, chunk in enumerate(single_chunks):
            cid = chunk['chunk_id']
            # Se abbiamo già le 4 query per questo chunk, le riutilizziamo
            if cid in existing_single_queries and len(existing_single_queries[cid]) >= 4:
                logger.info(f"[{i+1}/{len(single_chunks)}] Riutilizzo query esistenti per il chunk: {cid}")
                for rec in existing_single_queries[cid]:
                    record = {
                        "query_id": f"Q_{query_counter:04d}",
                        "query": rec["query"],
                        "category": rec["category"],
                        "relevant_doc_ids": rec["relevant_doc_ids"],
                        "source_metadata": rec["source_metadata"]
                    }
                    ground_truth_records.append(record)
                    query_counter += 1
                continue

            logger.info(f"[{i+1}/{len(single_chunks)}] Generazione query per il chunk: {cid}")
            
            prompt = f"""Analyze the following text (e.g., a technical documentation fragment) and generate exactly 4 search queries in English based ONLY on the provided content:

1. NEEDLE QUERY: A highly specific question targeting precise parameters, numerical limits, configurations, or parameter names present in the text.
2. PARAPHRASE QUERY: An equivalent question to the needle query, but formulated using different words and sentence structure to avoid exact keyword matching.
3. CONCEPTUAL QUERY: A high-level conceptual question focused on the 'how' or 'why' of a mechanism described in the text.
4. AMBIGUOUS QUERY: A short, vague keyword search consisting of only 1-3 keywords from the text (e.g., 'route table' or 'traffic filtering').

Text to analyze:
\"\"\"{chunk['text']}\"\"\""""

            try:
                # Esegue la chiamata strutturata
                queries: SingleChunkQueries = self._generate_with_retry(prompt, SingleChunkQueries)
                
                # Mappa ciascuna query ad un record del ground truth
                categories = {
                    "needle": queries.needle_query,
                    "paraphrase": queries.paraphrase_query,
                    "conceptual": queries.conceptual_query,
                    "ambiguous": queries.ambiguous_query
                }

                for cat, text in categories.items():
                    if text and text.strip():
                        record = {
                            "query_id": f"Q_{query_counter:04d}",
                            "query": text.strip(),
                            "category": cat,
                            "relevant_doc_ids": [chunk["chunk_id"]],
                            "source_metadata": {
                                "source_file": chunk.get("source_file"),
                                "chunk_id": chunk["chunk_id"]
                            }
                        }
                        ground_truth_records.append(record)
                        query_counter += 1

            except Exception as e:
                logger.error(f"Salto il chunk {chunk['chunk_id']} a causa di errori ripetuti dell'API: {e}")
                # Continua con il chunk successivo anziché fallire del tutto
                continue

        # 3. Generazione query Multi-Hop
        logger.info("Avvio generazione query multi-hop...")
        for j, (c1, c2) in enumerate(multihop_pairs):
            pair_key = tuple(sorted([c1['chunk_id'], c2['chunk_id']]))
            if pair_key in existing_multihop_queries:
                logger.info(f"[{j+1}/{len(multihop_pairs)}] Riutilizzo query multi-hop esistente per la coppia: {c1['chunk_id']} + {c2['chunk_id']}")
                rec = existing_multihop_queries[pair_key]
                record = {
                    "query_id": f"Q_{query_counter:04d}",
                    "query": rec["query"],
                    "category": "multi-hop",
                    "relevant_doc_ids": rec["relevant_doc_ids"],
                    "source_metadata": rec["source_metadata"]
                }
                ground_truth_records.append(record)
                query_counter += 1
                continue

            logger.info(f"[{j+1}/{len(multihop_pairs)}] Generazione query multi-hop per la coppia: {c1['chunk_id']} + {c2['chunk_id']}")
            
            prompt = f"""Analyze the following two related fragments of text (e.g., technical documentation) and generate exactly 1 complex query (MULTI-HOP QUERY) in English.
The query must be designed such that answering it requires connecting and combining information present in BOTH fragments. It must not be answerable using only one of the fragments.

Text 1:
\"\"\"{c1['text']}\"\"\"

Text 2:
\"\"\"{c2['text']}\"\"\""""

            try:
                queries_mh: MultiHopQueries = self._generate_with_retry(prompt, MultiHopQueries)
                
                if queries_mh.multi_hop_query and queries_mh.multi_hop_query.strip():
                    record = {
                        "query_id": f"Q_{query_counter:04d}",
                        "query": queries_mh.multi_hop_query.strip(),
                        "category": "multi-hop",
                        "relevant_doc_ids": [c1["chunk_id"], c2["chunk_id"]],
                        "source_metadata": {
                            "source_file": c1.get("source_file"),
                            "chunk_id_1": c1["chunk_id"],
                            "chunk_id_2": c2["chunk_id"]
                        }
                    }
                    ground_truth_records.append(record)
                    query_counter += 1

            except Exception as e:
                logger.error(f"Salto la coppia {c1['chunk_id']}+{c2['chunk_id']} a causa di errori: {e}")
                continue

        # 4. Scrittura su file JSONL
        with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
            for record in ground_truth_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Generazione completata! Salvare {len(ground_truth_records)} query in {GROUND_TRUTH_FILE}")
        return ground_truth_records
