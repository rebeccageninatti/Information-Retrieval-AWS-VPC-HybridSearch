"""
Script di analisi per la lunghezza dei chunk.
Esegue il chunking ricorsivo (RecursiveCharacterTextSplitter) per le dimensioni 300, 800, 1500
e genera statistiche descrittive (caratteri e parole) e grafici di distribuzione.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    PROCESSED_DATA_DIR,
    CHUNK_OVERLAP_PCT,
    PROJECT_ROOT
)
from src.utils.logging_utils import get_logger
from src.ingestion.chunker import DocumentChunker

# Definiamo le dimensioni dei chunk da testare specificamente per l'analisi diagnostica
ANALYSIS_CHUNK_SIZES = [400, 800, 1200]

logger = get_logger(__name__)

def run_analysis():
    logger.info("Avvio analisi di chunking...")
    
    processed_dir = PROCESSED_DATA_DIR / "vpc"
    if not processed_dir.exists():
        logger.error(f"Cartella processed {processed_dir} non esiste! Esegui prima run_ingestion.py --clean-only")
        sys.exit(1)
        
    chunker = DocumentChunker()
    
    # Raccogliamo tutti i testi puliti
    documents = []
    for root, _, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if text:
                        documents.append(text)
                except Exception as e:
                    logger.error(f"Impossibile leggere {file_path}: {e}")
                    
    logger.info(f"Caricati {len(documents)} documenti per l'analisi.")
    
    all_data = []
    
    # Esegue il chunking per ciascuna dimensione e strategia configurata
    strategies = ["recursive", "markdown-two-stage"]
    
    for strategy in strategies:
        logger.info(f"Avvio analisi per strategia: {strategy}")
        for chunk_size in ANALYSIS_CHUNK_SIZES:
            chunk_overlap = int(chunk_size * CHUNK_OVERLAP_PCT)
            logger.info(f"Strategia: {strategy} | chunk_size={chunk_size} (overlap={chunk_overlap})...")
            
            total_chunks = 0
            for doc in documents:
                if strategy == "recursive":
                    chunks = chunker.split_recursive(doc, chunk_size, chunk_overlap)
                    for chunk in chunks:
                        if chunk.strip():
                            char_len = len(chunk)
                            word_len = len(chunk.split())
                            all_data.append({
                                "strategy": strategy,
                                "chunk_size_config": chunk_size,
                                "char_count": char_len,
                                "word_count": word_len
                            })
                            total_chunks += 1
                elif strategy == "markdown-two-stage":
                    chunks_dict = chunker.split_markdown_two_stage(doc, chunk_size, chunk_overlap, inject_headers=True)
                    for chunk_obj in chunks_dict:
                        text_val = chunk_obj["text"]
                        if text_val.strip():
                            char_len = len(text_val)
                            word_len = len(text_val.split())
                            all_data.append({
                                "strategy": strategy,
                                "chunk_size_config": chunk_size,
                                "char_count": char_len,
                                "word_count": word_len
                            })
                            total_chunks += 1
                        
            logger.info(f"Strategia: {strategy} | Generati {total_chunks} chunk per dimensione {chunk_size}.")
            
    # Crea un DataFrame pandas per l'analisi
    df = pd.DataFrame(all_data)
    
    # Calcola statistiche per ciascuna strategia
    for strategy in strategies:
        df_strat = df[df["strategy"] == strategy]
        summary_stats = []
        for chunk_size in ANALYSIS_CHUNK_SIZES:
            df_sub = df_strat[df_strat["chunk_size_config"] == chunk_size]
            
            char_stats = df_sub["char_count"].describe()
            word_stats = df_sub["word_count"].describe()
            
            summary_stats.append({
                "Config Size (Chars)": chunk_size,
                "Total Chunks": len(df_sub),
                "Mean Chars": f"{char_stats['mean']:.1f}",
                "Median Chars": f"{char_stats['50%']:.1f}",
                "Min Chars": int(char_stats['min']) if not pd.isna(char_stats['min']) else 0,
                "Max Chars": int(char_stats['max']) if not pd.isna(char_stats['max']) else 0,
                "Std Chars": f"{char_stats['std']:.1f}",
                "Mean Words": f"{word_stats['mean']:.1f}",
                "Median Words": f"{word_stats['50%']:.1f}",
                "Min Words": int(word_stats['min']) if not pd.isna(word_stats['min']) else 0,
                "Max Words": int(word_stats['max']) if not pd.isna(word_stats['max']) else 0,
                "Std Words": f"{word_stats['std']:.1f}"
            })
            
        df_summary = pd.DataFrame(summary_stats)
        
        # Stampa i risultati in formato tabella leggibile
        print("\n" + "="*80)
        print(f"STATISTICHE DI CHUNKING - STRATEGIA: {strategy.upper()}")
        print("="*80)
        print(df_summary.to_string(index=False))
        print("="*80 + "\n")
        
    # Genera i grafici di distribuzione (Disabilitato per ora, vedi report/figures/README.md)
    # generate_plots(df)

def generate_plots(df: pd.DataFrame):
    # Setup stile grafici
    sns.set_theme(style="whitegrid")
    
    # Crea la figura con due subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Grafico 1: Confronto Lunghezza in Caratteri per Strategia
    sns.boxplot(
        x="chunk_size_config", 
        y="char_count", 
        hue="strategy",
        data=df, 
        ax=axes[0],
        palette="muted"
    )
    axes[0].set_title("Distribuzione della Lunghezza in Caratteri per Strategia", fontsize=13, fontweight="bold", pad=15)
    axes[0].set_xlabel("Dimensione Configurate (Caratteri)", fontsize=11)
    axes[0].set_ylabel("Lunghezza Reale (Caratteri)", fontsize=11)
    axes[0].legend(title="Strategia")
    
    # Grafico 2: Confronto Lunghezza in Parole per Strategia
    sns.boxplot(
        x="chunk_size_config", 
        y="word_count", 
        hue="strategy",
        data=df, 
        ax=axes[1],
        palette="muted"
    )
    axes[1].set_title("Distribuzione della Lunghezza in Parole per Strategia", fontsize=13, fontweight="bold", pad=15)
    axes[1].set_xlabel("Dimensione Configurate (Caratteri)", fontsize=11)
    axes[1].set_ylabel("Lunghezza Reale (Parole)", fontsize=11)
    axes[1].legend(title="Strategia")
    
    # Ottimizza layout
    plt.tight_layout()
    
    # Crea cartella report/figures se non esiste
    figures_dir = PROJECT_ROOT / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = figures_dir / "chunk_length_distributions.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Grafico delle distribuzioni comparato salvato in: {plot_path}")
    plt.close()

if __name__ == "__main__":
    run_analysis()
