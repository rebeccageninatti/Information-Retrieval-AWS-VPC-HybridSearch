"""
Script principale per eseguire la pipeline di Ingestion:
1. Pulisce i file grezzi in data/raw/vpc/ e li salva in data/processed/vpc/.
2. Genera i chunk di testo per le dimensioni 300, 800, 1500 e li salva in data/chunks/.
"""

import os
import json
import sys
from pathlib import Path
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHUNKS_DIR,
    CHUNK_SIZES,
    CHUNK_OVERLAP_PCT,
    DEFAULT_CHUNK_STRATEGY,
    DATASET_SUBDIR,
    DATASET_NAME
)
from src.utils.logging_utils import get_logger
from src.ingestion.cleaner import DocumentCleaner
from src.ingestion.chunker import DocumentChunker

logger = get_logger(__name__)

def clean_documents():
    """
    Scorre ricorsivamente data/raw/vpc/, pulisce i file markdown e li salva in data/processed/vpc/.
    """
    logger.info("Avvio pulizia documenti...")
    cleaner = DocumentCleaner()
    
    input_dir = RAW_DATA_DIR / "vpc"
    output_dir = PROCESSED_DATA_DIR / "vpc"
    
    if not input_dir.exists():
        logger.error(f"La cartella dei dati grezzi {input_dir} non esiste!")
        sys.exit(1)
        
    count = 0
    # Scorre ricorsivamente la cartella
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(input_dir)
                
                # Definisce il percorso di output
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                        
                    cleaned_text = cleaner.clean_markdown(raw_text)
                    
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)
                        
                    count += 1
                except Exception as e:
                    logger.error(f"Errore durante la pulizia di {file_path}: {e}")
                    
    logger.info(f"Pulizia completata! Elaborati {count} file markdown. Salvati in {output_dir}")


def generate_chunks(strategy: str = "recursive"):
    """
    Legge i documenti puliti in data/processed/vpc/, genera i chunk per le 3 dimensioni configurate
    e li salva in data/chunks/ come file JSONL.
    """
    logger.info(f"Avvio generazione chunk con strategia: {strategy}")
    chunker = DocumentChunker()
    processed_dir = PROCESSED_DATA_DIR / DATASET_SUBDIR
    
    if not processed_dir.exists():
        logger.error(f"La cartella dei documenti puliti {processed_dir} non esiste. Esegui prima la pulizia!")
        sys.exit(1)
        
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Per ciascuna dimensione dei chunk configurata
    for chunk_size in CHUNK_SIZES:
        chunk_overlap = int(chunk_size * CHUNK_OVERLAP_PCT)
        output_file = CHUNKS_DIR / f"chunks_{DATASET_NAME}_{strategy}_{chunk_size}.jsonl"
        
        logger.info(f"Generazione chunk per size={chunk_size}, overlap={chunk_overlap}...")
        
        chunks_list = []
        global_chunk_idx = 0
        
        # Rileva dinamicamente se ci sono sottocartelle già elaborate come dataset a sé stanti
        pre_existing_chunks = []
        skipped_subdirs = set()
        
        # Scansiona le sottocartelle dirette in processed_dir
        if processed_dir.exists():
            for item in processed_dir.iterdir():
                if item.is_dir():
                    subdir_name = item.name
                    # Cerca il file di chunk per questa sottocartella specifica
                    subdir_chunks_file = CHUNKS_DIR / f"chunks_{DATASET_NAME}_{subdir_name}_{strategy}_{chunk_size}.jsonl"
                    if subdir_chunks_file.exists():
                        logger.info(f"Trovati chunk pre-esistenti per la sottocartella '{subdir_name}' in {subdir_chunks_file.name}. Verranno riutilizzati.")
                        skipped_subdirs.add(subdir_name)
                        try:
                            with open(subdir_chunks_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    if line.strip():
                                        chunk = json.loads(line)
                                        # Aggiorna il percorso del file sorgente se non contiene già il prefisso della sottocartella
                                        if not chunk["source_file"].startswith(f"{subdir_name}/"):
                                            chunk["source_file"] = f"{subdir_name}/{chunk['source_file']}"
                                        pre_existing_chunks.append(chunk)
                        except Exception as e:
                            logger.error(f"Errore nel caricamento dei chunk per '{subdir_name}': {e}")
                            skipped_subdirs.remove(subdir_name)
                            
            if skipped_subdirs:
                logger.info(f"Sottocartelle saltate dal chunking ricorsivo perché già elaborate: {list(skipped_subdirs)}")
                logger.info(f"Caricati in totale {len(pre_existing_chunks)} chunk pre-elaborati.")

        for root, _, files in os.walk(processed_dir):
            # Calcola il percorso relativo di root rispetto a processed_dir per verificare se siamo dentro una sottocartella da saltare
            try:
                rel_root = Path(root).relative_to(processed_dir)
                if rel_root.parts and rel_root.parts[0] in skipped_subdirs:
                    continue
            except ValueError:
                pass

            for file in files:
                if file.endswith(".md"):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(processed_dir)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                            
                        # Salta i file vuoti dopo la pulizia
                        if not text.strip():
                            continue
                            
                        # Seleziona la strategia di chunking
                        if strategy == "recursive":
                            raw_chunks = chunker.split_recursive(text, chunk_size, chunk_overlap)
                            chunks = [{"text": c, "metadata": {}, "header_path": ""} for c in raw_chunks]
                        elif strategy == "token":
                            raw_chunks = chunker.split_tokens(text, chunk_size, chunk_overlap)
                            chunks = [{"text": c, "metadata": {}, "header_path": ""} for c in raw_chunks]
                        elif strategy == "markdown-two-stage":
                            chunks = chunker.split_markdown_two_stage(text, chunk_size, chunk_overlap)
                        else:
                            logger.error(f"Strategia di chunking sconosciuta: {strategy}")
                            sys.exit(1)
                            
                        # Salva ciascun chunk con metadati
                        for idx, chunk_data in enumerate(chunks):
                            chunk_text = chunk_data["text"]
                            # Salta eventuali chunk vuoti prodotti dallo splitter
                            if not chunk_text.strip():
                                continue
                                
                            char_count = len(chunk_text)
                            word_count = len(chunk_text.split())
                            
                            chunk_record = {
                                "chunk_id": f"{rel_path.stem}_{idx}",
                                "source_file": str(rel_path),
                                "chunk_index": idx,
                                "text": chunk_text,
                                "char_count": char_count,
                                "word_count": word_count,
                                "chunk_size_config": chunk_size,
                                "chunk_overlap_config": chunk_overlap,
                                "strategy": strategy,
                                "headers": chunk_data.get("metadata", {}),
                                "header_path": chunk_data.get("header_path", "")
                            }
                            chunks_list.append(chunk_record)
                            global_chunk_idx += 1
                            
                    except Exception as e:
                        logger.error(f"Errore durante il chunking di {file_path}: {e}")
                        
        # Aggiungiamo i chunk pre-esistenti caricati
        chunks_list.extend(pre_existing_chunks)
        global_chunk_idx += len(pre_existing_chunks)
        
        # Scrive il file JSONL di output
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for chunk in chunks_list:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            logger.info(f"Salvati {global_chunk_idx} chunk in {output_file}")
        except Exception as e:
            logger.error(f"Errore durante la scrittura del file di output {output_file}: {e}")

def main():
    # Esegue la pulizia e la generazione dei chunk utilizzando la configurazione predefinita
    clean_documents()
    generate_chunks(strategy=DEFAULT_CHUNK_STRATEGY)

if __name__ == "__main__":
    main()
