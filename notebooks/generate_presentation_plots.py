#!/usr/bin/env python3
"""
Script per la generazione di grafici in stile minimal con colori pastello per la presentazione.
Salva le immagini in report/presentazione/immagini/ per le seguenti slide:
1. Risultati Globali di Accuratezza (3 varianti)
2. L'Impatto del Reranking (3 varianti, con prevenzione sovrapposizione scritte)
3. Analisi per Categoria di Query (3 varianti, inclusa Radar chart)
4. Analisi di Sensibilità — Dimensione del Chunk (3 varianti)

Inoltre, stampa a schermo la tabella di ripartizione dei chunk.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE STRUTTURA E PATHS
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
IMAGINI_DIR = PROJECT_ROOT / "report" / "presentazione" / "immagini"
IMAGINI_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# STILE GLOBALE (Minimalist Pastel Design System)
# ──────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.color": "#CCCCCC",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "#CCCCCC",
    "patch.edgecolor": "white",
    "patch.linewidth": 0.5
})

# Palette pastello minimal
C_BM25 = "#FFAAA5"        # Rosa salmone pastello
C_BM25_DARK = "#E58B86"   # Rosa scuro (per Rerank)
C_VECTOR = "#A2C2E8"      # Azzurro pastello
C_VECTOR_DARK = "#85A3C7" # Azzurro scuro (per Rerank)
C_HYBRID = "#A8E6CF"      # Verde menta pastello
C_HYBRID_DARK = "#89C7B1" # Verde scuro (per Rerank)
C_TEXT = "#2D3748"        # Grigio scuro per testi
C_TEXT_LIGHT = "#718096"  # Grigio chiaro per sottotitoli/etichette
C_BG = "#FAFAFA"          # Sfondo off-white
C_BORDER = "#E2E8F0"      # Bordo grigio chiaro

# Palette degli Alpha (per visualizzazioni a gradiente)
PALETTE_ALPHAS = {
    "BM25-only": C_BM25,
    "Vector-only": C_VECTOR,
    "Hybrid-RRF (α=0.1)": "#F5D6D4",
    "Hybrid-RRF (α=0.3)": "#EBE2F5",
    "Hybrid-RRF (α=0.5)": C_HYBRID,
    "Hybrid-RRF (α=0.7)": "#D5EBE2",
    "Hybrid-RRF (α=0.9)": "#FAF1D6",
    "Vector-only + Rerank": C_VECTOR_DARK,
    "BM25-only + Rerank": C_BM25_DARK,
    "Hybrid-RRF (α=0.5) + Rerank": C_HYBRID_DARK
}

# ──────────────────────────────────────────────────────────────────────
# HELPERS PER IL CARICAMENTO DATI
# ──────────────────────────────────────────────────────────────────────
def load_json(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Risultati non trovati in: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_strategy_name(name):
    """Semplifica i nomi delle strategie per renderli più leggibili nei grafici."""
    if name == "Hybrid-RRF (α=0.5)":
        return "Hybrid (α=0.5)"
    if name == "Hybrid-RRF (α=0.9)":
        return "Hybrid (α=0.9)"
    if name == "Hybrid-RRF (α=0.5) + Rerank":
        return "Hybrid (α=0.5) + Rerank"
    return name

# ──────────────────────────────────────────────────────────────────────
# FUNZIONE DI ALGORITMO DI SPAZIATURA VERTICALE PER ETICHETTE (SLOPE CHART)
# ──────────────────────────────────────────────────────────────────────
def adjust_label_positions(y_vals, min_diff=0.015):
    """
    Algoritmo di spaziatura verticale per evitare sovrapposizioni di testi.
    y_vals: lista di tuple (original_y, text, color)
    ritorna una lista di y_adjusted nello stesso ordine.
    """
    # Ordina per y originale
    indexed = sorted(enumerate(y_vals), key=lambda x: x[1][0])
    n = len(indexed)
    adjusted = [0.0] * n
    
    # Primo passaggio (dal basso verso l'alto)
    for i in range(n):
        orig_idx, (y, text, col) = indexed[i]
        if i == 0:
            adjusted[orig_idx] = y
        else:
            prev_idx = indexed[i-1][0]
            if y - adjusted[prev_idx] < min_diff:
                adjusted[orig_idx] = adjusted[prev_idx] + min_diff
            else:
                adjusted[orig_idx] = y
                
    # Secondo passaggio (dall'alto verso il basso per bilanciare eventuali spinte eccessive)
    for i in range(n-2, -1, -1):
        orig_idx, (y, text, col) = indexed[i]
        next_idx = indexed[i+1][0]
        if adjusted[next_idx] - adjusted[orig_idx] < min_diff:
            # Spingi leggermente verso il basso se c'è spazio rispetto al valore originale
            candidate = adjusted[next_idx] - min_diff
            if candidate >= y:
                adjusted[orig_idx] = candidate
                
    return adjusted

# ──────────────────────────────────────────────────────────────────────
# SLIDE 1: Risultati Globali di Accuratezza
# ──────────────────────────────────────────────────────────────────────
def plot_slide1_global_accuracy(data_rerank, data_standard):
    print("Generazione grafici Slide 1 (Risultati Globali)...")
    results = data_standard["results"]
    
    # Filtriamo solo le strategie senza reranking per la slide di accuratezza globale
    target_strategies = [
        "BM25-only",
        "Vector-only",
        "Hybrid-RRF (α=0.1)",
        "Hybrid-RRF (α=0.3)",
        "Hybrid-RRF (α=0.5)",
        "Hybrid-RRF (α=0.7)",
        "Hybrid-RRF (α=0.9)"
    ]
    
    # ── VARIANTE 1: Barre orizzontali di NDCG@10 (Ordinato)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    plot_data = []
    for strat in target_strategies:
        if strat in results:
            val = results[strat]["metrics"]["overall"]["ndcg@10"]
            plot_data.append((strat, val))
    
    plot_data.sort(key=lambda x: x[1])
    strats, vals = zip(*plot_data)
    strats_clean = [format_strategy_name(s) for s in strats]
    colors = [PALETTE_ALPHAS.get(s, C_VECTOR) for s in strats]
    
    bars = ax.barh(strats_clean, vals, color=colors, height=0.6, edgecolor="white", zorder=3)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("nDCG@10", color=C_TEXT_LIGHT, fontweight="semibold")
    ax.set_title("Risultati Globali: Confronto nDCG@10 per Strategia\n(Dataset Completo vpc_latest · 4427 chunk)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.4f}", 
                va="center", ha="left", fontsize=9, color=C_TEXT, fontweight="semibold")
                
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="y")
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide1_global_accuracy_type1.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    
    # ── VARIANTE 1 (Recall): Barre orizzontali di Recall@10 (Ordinato)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    plot_data_recall = []
    for strat in target_strategies:
        if strat in results:
            val = results[strat]["metrics"]["overall"]["recall@10"]
            plot_data_recall.append((strat, val))
            
    plot_data_recall.sort(key=lambda x: x[1])
    strats_r, vals_r = zip(*plot_data_recall)
    strats_clean_r = [format_strategy_name(s) for s in strats_r]
    colors_r = [PALETTE_ALPHAS.get(s, C_VECTOR) for s in strats_r]
    
    bars = ax.barh(strats_clean_r, vals_r, color=colors_r, height=0.6, edgecolor="white", zorder=3)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Recall@10", color=C_TEXT_LIGHT, fontweight="semibold")
    ax.set_title("Risultati Globali: Confronto Recall@10 per Strategia\n(Dataset Completo vpc_latest · 4427 chunk)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.4f}", 
                va="center", ha="left", fontsize=9, color=C_TEXT, fontweight="semibold")
                
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="y")
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide1_global_accuracy_type1_recall.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    
    # ── VARIANTE 2: Barre raggruppate delle metriche chiave (NDCG@10, Recall@10, MRR)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    metrics = ["ndcg@10", "recall@10", "mrr"]
    labels = ["nDCG@10", "Recall@10", "MRR"]
    strategies_subset = ["BM25-only", "Vector-only", "Hybrid-RRF (α=0.5)", "Hybrid-RRF (α=0.9)"]
    
    x = np.arange(len(metrics))
    width = 0.18
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    
    for i, strat in enumerate(strategies_subset):
        vals = [results[strat]["metrics"]["overall"][m] for m in metrics]
        color = PALETTE_ALPHAS.get(strat, C_VECTOR)
        bars = ax.bar(x + offsets[i], vals, width, label=format_strategy_name(strat), 
                      color=color, edgecolor="white", zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", 
                    ha="center", va="bottom", fontsize=7.5, color=C_TEXT_LIGHT)
                    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=C_TEXT, fontweight="semibold")
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Punteggio", color=C_TEXT_LIGHT)
    ax.set_title("Metriche Chiave di Retrieval: BM25, Vector e Ibrido\n(Dataset Completo vpc_latest · 4427 chunk)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor=C_BG, edgecolor=C_BORDER)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide1_global_accuracy_type2.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    # ── VARIANTE 3: Scatter Plot Latenza vs Accuratezza (NDCG@10)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    # Carichiamo le latenze realistiche (simulate dal file standard non-fast)
    std_results = data_standard["results"]
    scatter_data = []
    for strat in target_strategies:
        if strat in std_results:
            acc = std_results[strat]["metrics"]["overall"]["ndcg@10"]
            lat = std_results[strat]["avg_latency_ms"]
            scatter_data.append((strat, lat, acc))
            
    for strat, lat, acc in scatter_data:
        color = PALETTE_ALPHAS.get(strat, C_VECTOR)
        clean_name = format_strategy_name(strat)
        ax.scatter(lat, acc, s=180, color=color, edgecolor="gray", linewidth=0.5, alpha=0.9, zorder=3, label=clean_name)
        # Etichetta vicino al punto
        offset_x = 12 if lat < 600 else -12
        ha = "left" if lat < 600 else "right"
        ax.text(lat + offset_x, acc - 0.002, clean_name, fontsize=8.5, ha=ha, va="center", color=C_TEXT, fontweight="semibold")
        
    ax.set_xlabel("Latenza Media di Risposta (ms)", color=C_TEXT_LIGHT, fontweight="semibold")
    ax.set_ylabel("Accuratezza (nDCG@10)", color=C_TEXT_LIGHT, fontweight="semibold")
    ax.set_xlim(-50, 950)
    ax.set_ylim(0.50, 0.65)
    ax.set_title("Trade-off Latenza vs. Accuratezza (nDCG@10)\n(Dataset Completo vpc_latest · Latenze di rete simulate)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide1_global_accuracy_type3.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# SLIDE 2: L'Impatto del Reranking
# ──────────────────────────────────────────────────────────────────────
def plot_slide2_reranking(data_rerank):
    print("Generazione grafici Slide 2 (Reranking)...")
    results = data_rerank["results"]
    
    # Definiamo le coppie base vs rerank
    pairs = [
        ("BM25-only", "BM25-only + Rerank", "BM25", C_BM25, C_BM25_DARK),
        ("Vector-only", "Vector-only + Rerank", "Vector", C_VECTOR, C_VECTOR_DARK),
        ("Hybrid-RRF (α=0.5)", "Hybrid-RRF (α=0.5) + Rerank", "Hybrid (α=0.5)", C_HYBRID, C_HYBRID_DARK)
    ]
    
    # ── VARIANTE 1: Barre raggruppate (Prima vs Dopo Rerank)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    x = np.arange(len(pairs))
    width = 0.35
    
    base_vals = [results[p[0]]["metrics"]["overall"]["ndcg@10"] for p in pairs]
    rerank_vals = [results[p[1]]["metrics"]["overall"]["ndcg@10"] for p in pairs]
    
    bars1 = ax.bar(x - width/2, base_vals, width, label="Base Retriever", 
                   color=[p[3] for p in pairs], edgecolor="white", zorder=3)
    bars2 = ax.bar(x + width/2, rerank_vals, width, label="+ Jina Reranker v3", 
                   color=[p[4] for p in pairs], edgecolor="white", zorder=3)
                   
    # Annotazione dei valori
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_TEXT_LIGHT)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_TEXT, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs], fontsize=10, color=C_TEXT, fontweight="semibold")
    ax.set_ylim(0.4, 0.82)
    ax.set_ylabel("nDCG@10", color=C_TEXT_LIGHT)
    ax.set_title("Impatto del Reranking su nDCG@10\n(Dataset Completo vpc_latest · 415 query)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor=C_BG, edgecolor=C_BORDER, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide2_reranking_type1.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    
    # ── Barre raggruppate per Recall@10 (Richiesto dall'utente)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    base_recall_vals = [results[p[0]]["metrics"]["overall"]["recall@10"] for p in pairs]
    rerank_recall_vals = [results[p[1]]["metrics"]["overall"]["recall@10"] for p in pairs]
    
    bars1 = ax.bar(x - width/2, base_recall_vals, width, label="Base Retriever", 
                   color=[p[3] for p in pairs], edgecolor="white", zorder=3)
    bars2 = ax.bar(x + width/2, rerank_recall_vals, width, label="+ Jina Reranker v3", 
                   color=[p[4] for p in pairs], edgecolor="white", zorder=3)
                   
    # Annotazione dei valori
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_TEXT_LIGHT)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_TEXT, fontweight="bold")
        
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs], fontsize=10, color=C_TEXT, fontweight="semibold")
    ax.set_ylim(0.6, 0.95)
    ax.set_ylabel("Recall@10", color=C_TEXT_LIGHT)
    ax.set_title("Impatto del Reranking su Recall@10\n(Dataset Completo vpc_latest · 415 query)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor=C_BG, edgecolor=C_BORDER, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide2_reranking_type1_recall.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    
    # ── VARIANTE 2: Incremento Netto (Delta)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    deltas_ndcg = [results[p[1]]["metrics"]["overall"]["ndcg@10"] - results[p[0]]["metrics"]["overall"]["ndcg@10"] for p in pairs]
    deltas_recall = [results[p[1]]["metrics"]["overall"]["recall@5"] - results[p[0]]["metrics"]["overall"]["recall@5"] for p in pairs]
    
    x = np.arange(len(pairs))
    width = 0.3
    
    bars1 = ax.bar(x - width/2, deltas_ndcg, width, label="Guadagno nDCG@10", color="#A2D9CE", edgecolor="white", zorder=3)
    bars2 = ax.bar(x + width/2, deltas_recall, width, label="Guadagno Recall@5", color="#D2B4DE", edgecolor="white", zorder=3)
    
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"+{h:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_TEXT, fontweight="semibold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"+{h:.3f}", ha="center", va="bottom", fontsize=8.5, color=C_TEXT, fontweight="semibold")
        
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in pairs], fontsize=10, color=C_TEXT, fontweight="semibold")
    ax.set_ylabel("Incremento Assoluto (Delta)", color=C_TEXT_LIGHT)
    ax.set_ylim(0, 0.22)
    ax.set_title("Incremento dell'Accuratezza grazie al Reranking (Delta)\n(Incremento netto sulle metriche nDCG@10 e Recall@5)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor=C_BG, edgecolor=C_BORDER)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide2_reranking_type2.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    # ── VARIANTE 3: Slope Graph (Con algoritmo di spaziatura per evitare sovrapposizioni delle scritte)
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    # Prepariamo i dati per la spaziatura verticale delle etichette
    left_labels = []
    right_labels = []
    
    for p in pairs:
        val_base = results[p[0]]["metrics"]["overall"]["ndcg@10"]
        val_rerank = results[p[1]]["metrics"]["overall"]["ndcg@10"]
        left_labels.append((val_base, f"{p[2]}: {val_base:.4f}", p[3]))
        right_labels.append((val_rerank, f"{p[2]} + Rerank: {val_rerank:.4f}", p[4]))
        
    # Spaziamo verticalmente
    left_adj = adjust_label_positions(left_labels, min_diff=0.018)
    right_adj = adjust_label_positions(right_labels, min_diff=0.018)
    
    # Plot linee di pendenza
    for i, p in enumerate(pairs):
        val_base = results[p[0]]["metrics"]["overall"]["ndcg@10"]
        val_rerank = results[p[1]]["metrics"]["overall"]["ndcg@10"]
        
        # Linea di collegamento
        ax.plot([0, 1], [val_base, val_rerank], marker="o", color=p[4], linewidth=2.5, markersize=8, zorder=3)
        
        # Etichetta sinistra (Base)
        y_adj_left = left_adj[i]
        ax.text(-0.05, y_adj_left, left_labels[i][1], ha="right", va="center", 
                fontsize=9.5, color=C_TEXT, fontweight="semibold")
        
        # Etichetta destra (Rerank)
        y_adj_right = right_adj[i]
        ax.text(1.05, y_adj_right, right_labels[i][1], ha="left", va="center", 
                fontsize=9.5, color=C_TEXT, fontweight="bold")
        
        # Linee tratteggiate di collegamento se le etichette sono state spostate
        if abs(y_adj_left - val_base) > 0.001:
            ax.plot([-0.04, -0.01], [y_adj_left, val_base], linestyle=":", color="gray", linewidth=0.8)
        if abs(y_adj_right - val_rerank) > 0.001:
            ax.plot([1.01, 1.04], [val_rerank, y_adj_right], linestyle=":", color="gray", linewidth=0.8)
            
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.48, 0.77)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Base Retriever", "+ Jina Reranker v3"], fontsize=11, color=C_TEXT, fontweight="bold")
    ax.set_ylabel("nDCG@10", color=C_TEXT_LIGHT)
    ax.set_title("Analisi Pendenza: Incremento dell'accuratezza nDCG@10\n(Le etichette sono auto-spaziate per evitare sovrapposizioni)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=20)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.grid(visible=False, axis="x")
    
    # Nascondi bordi per fare una visualizzazione pura delle linee
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide2_reranking_type3.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# SLIDE 3: Analisi per Categoria di Query
# ──────────────────────────────────────────────────────────────────────
def plot_slide3_categories(data_rerank):
    print("Generazione grafici Slide 3 (Categorie)...")
    results = data_rerank["results"]
    
    categories = ["needle", "paraphrase", "conceptual", "ambiguous", "multi-hop"]
    categories_clean = [c.capitalize() for c in categories]
    
    # Definiamo le 6 strategie per mostrare l'impatto del rerank per categoria
    all_strategies = [
        "BM25-only", "BM25-only + Rerank",
        "Vector-only", "Vector-only + Rerank",
        "Hybrid-RRF (α=0.5)", "Hybrid-RRF (α=0.5) + Rerank"
    ]
    strategy_labels = [
        "BM25", "BM25 + Rerank",
        "Vector", "Vector + Rerank",
        "Hybrid (α=0.5)", "Hybrid (α=0.5) + Rerank"
    ]
    colors = [C_BM25, C_BM25_DARK, C_VECTOR, C_VECTOR_DARK, C_HYBRID, C_HYBRID_DARK]
    
    # ── VARIANTE 1: Barre raggruppate per categoria (Tutte e 6 le strategie per vedere il Rerank)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    x = np.arange(len(categories))
    width = 0.12
    offsets = [-2.5*width, -1.5*width, -0.5*width, 0.5*width, 1.5*width, 2.5*width]
    
    for i, strat in enumerate(all_strategies):
        vals = [results[strat]["metrics"][cat]["ndcg@10"] for cat in categories]
        bars = ax.bar(x + offsets[i], vals, width, label=strategy_labels[i], 
                      color=colors[i], edgecolor="white", zorder=3)
        # Scriviamo i valori solo sulle strategie principali o con rerank per evitare sovraffollamento
        if "+ Rerank" in strategy_labels[i]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", 
                        ha="center", va="bottom", fontsize=6.5, color=C_TEXT, fontweight="bold")
                    
    ax.set_xticks(x)
    ax.set_xticklabels(categories_clean, fontsize=10, color=C_TEXT, fontweight="semibold")
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("nDCG@10", color=C_TEXT_LIGHT)
    ax.set_title("Analisi di Dettaglio: nDCG@10 per Categoria con Reranking\n(Confronto Prima/Dopo per BM25, Vector e Ibrido)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor=C_BG, edgecolor=C_BORDER, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide3_category_analysis_type1.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    # ── VARIANTE 2: Heatmap strategica delle categorie (Espansa a 6 strategie per includere il Rerank)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    fig.patch.set_facecolor(C_BG)
    
    # Prepariamo la matrice
    matrix_data = []
    for strat in all_strategies:
        row = [results[strat]["metrics"][cat]["ndcg@10"] for cat in categories]
        matrix_data.append(row)
        
    df_heatmap = pd.DataFrame(matrix_data, index=strategy_labels, columns=categories_clean)
    
    cmap = sns.light_palette("#2B4C7E", as_cmap=True)
    sns.heatmap(df_heatmap, annot=True, cmap=cmap, fmt=".4f", cbar=True, 
                linewidths=1.0, linecolor="white", ax=ax, annot_kws={"fontweight": "semibold"})
                
    ax.set_title("Matrice di Rilevanza (nDCG@10) per Categoria di Query con Reranking", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.tick_params(axis='x', labelsize=10, labelcolor=C_TEXT)
    ax.tick_params(axis='y', labelsize=10, labelcolor=C_TEXT)
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide3_category_analysis_type2.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    # ── VARIANTE 3: Radar Chart Subplots (Confronto Prima/Dopo Rerank per ciascun Retriever)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), subplot_kw=dict(projection='polar'), dpi=300)
    fig.patch.set_facecolor(C_BG)
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # Definiamo i tre subplots (BM25, Vector, Hybrid)
    retrievers = [
        {
            "name": "BM25",
            "base_strat": "BM25-only",
            "rerank_strat": "BM25-only + Rerank",
            "base_color": C_BM25,
            "rerank_color": C_BM25_DARK
        },
        {
            "name": "Vector",
            "base_strat": "Vector-only",
            "rerank_strat": "Vector-only + Rerank",
            "base_color": C_VECTOR,
            "rerank_color": C_VECTOR_DARK
        },
        {
            "name": "Hybrid (α=0.5)",
            "base_strat": "Hybrid-RRF (α=0.5)",
            "rerank_strat": "Hybrid-RRF (α=0.5) + Rerank",
            "base_color": C_HYBRID,
            "rerank_color": C_HYBRID_DARK
        }
    ]
    
    for i, ret in enumerate(retrievers):
        ax = axes[i]
        ax.set_facecolor(C_BG)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Etichette delle categorie
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_clean, color=C_TEXT, size=9, fontweight="semibold")
        
        # Cerchi radiali
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color=C_TEXT_LIGHT, size=7.5)
        ax.set_ylim(0, 0.95)
        
        # Dati base
        vals_base = [results[ret["base_strat"]]["metrics"][cat]["ndcg@10"] for cat in categories]
        vals_base += vals_base[:1]
        ax.plot(angles, vals_base, linewidth=1.8, linestyle='dashed', label="Base", color=ret["base_color"])
        ax.fill(angles, vals_base, color=ret["base_color"], alpha=0.08)
        
        # Dati rerank
        vals_rerank = [results[ret["rerank_strat"]]["metrics"][cat]["ndcg@10"] for cat in categories]
        vals_rerank += vals_rerank[:1]
        ax.plot(angles, vals_rerank, linewidth=2.2, linestyle='solid', label="+ Rerank", color=ret["rerank_color"])
        ax.fill(angles, vals_rerank, color=ret["rerank_color"], alpha=0.15)
        
        ax.set_title(ret["name"], fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=True, facecolor=C_BG, edgecolor=C_BORDER, ncol=2)
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide3_category_analysis_type3.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# SLIDE 4: Analisi di Sensibilità — Dimensione del Chunk
# ──────────────────────────────────────────────────────────────────────
def plot_slide4_chunk_sensitivity(data_400, data_800, data_1200):
    print("Generazione grafici Slide 4 (Sensibilità Dimensione Chunk)...")
    
    sizes = [400, 800, 1200]
    datasets = [data_400, data_800, data_1200]
    
    target_strategies = ["BM25-only", "Vector-only", "Hybrid-RRF (α=0.9)"]
    strategy_labels = ["BM25", "Vector", "Hybrid (α=0.9)"]
    colors = [C_BM25, C_VECTOR, C_HYBRID]
    
    # ── VARIANTE 1: Line Plot (Andamento Curve)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    for i, strat in enumerate(target_strategies):
        vals = []
        for data in datasets:
            vals.append(data["results"][strat]["metrics"]["overall"]["ndcg@10"])
            
        ax.plot(sizes, vals, marker="o", markersize=7, linewidth=2, 
                label=strategy_labels[i], color=colors[i], zorder=3)
        
        for x_sz, val in zip(sizes, vals):
            ax.text(x_sz, val + 0.008, f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, color=C_TEXT)
            
    ax.set_xticks(sizes)
    ax.set_xlabel("Dimensione del Chunk (Caratteri)", color=C_TEXT_LIGHT, fontweight="semibold")
    ax.set_ylabel("nDCG@10", color=C_TEXT_LIGHT, fontweight="semibold")
    ax.set_ylim(0.3, 0.67)
    
    # Evidenzia lo sweet spot (800 caratteri)
    ax.axvline(800, color="gray", linestyle=":", alpha=0.5, zorder=1)
    ax.text(800 + 20, 0.32, "★ Dimensione Ottimale (Sweet Spot)", color=C_TEXT_LIGHT, fontsize=8.5, fontstyle="italic")
    
    ax.set_title("Analisi di Sensibilità: Accuratezza nDCG@10 vs Dimensione Chunk\n(Sottogruppo userguide · 415 query)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor=C_BG, edgecolor=C_BORDER, loc="lower right")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide4_chunk_sensitivity_type1.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    # ── VARIANTE 2: Barre Raggruppate
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    
    x = np.arange(len(target_strategies))
    width = 0.24
    offsets = [-width, 0, width]
    
    for i, sz in enumerate(sizes):
        vals = [datasets[i]["results"][strat]["metrics"]["overall"]["ndcg@10"] for strat in target_strategies]
        alpha = 1.0 if sz == 800 else 0.6
        hatch = "" if sz == 800 else "//"
        bars = ax.bar(x + offsets[i], vals, width, label=f"Chunk: {sz} char", 
                      color=colors, alpha=alpha, hatch=hatch, edgecolor="white", zorder=3)
                      
        for bar in bars:
            h = bar.get_height()
            weight = "bold" if sz == 800 else "normal"
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", 
                    ha="center", va="bottom", fontsize=8, color=C_TEXT, fontweight=weight)
                    
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_labels, fontsize=10, color=C_TEXT, fontweight="semibold")
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("nDCG@10", color=C_TEXT_LIGHT)
    ax.set_title("Confronto Impatto Dimensione Chunk per Strategia\n(Sottogruppo userguide · Risultati preliminari)", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor=C_BG, edgecolor=C_BORDER, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(visible=False, axis="x")
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide4_chunk_sensitivity_type2.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    # ── VARIANTE 3: Heatmap di configurazione (Strategia vs Chunk Size)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    fig.patch.set_facecolor(C_BG)
    
    matrix_data = []
    for strat in target_strategies:
        row = [data["results"][strat]["metrics"]["overall"]["ndcg@10"] for data in datasets]
        matrix_data.append(row)
        
    df_heatmap = pd.DataFrame(matrix_data, index=strategy_labels, columns=[f"{sz} Caratteri" for sz in sizes])
    
    cmap = sns.light_palette("#6A5ACD", as_cmap=True)
    sns.heatmap(df_heatmap, annot=True, cmap=cmap, fmt=".4f", cbar=True, 
                linewidths=1.0, linecolor="white", ax=ax, annot_kws={"fontweight": "semibold"})
                
    ax.set_title("Mappa di Calore nDCG@10: Strategia vs Dimensione Chunk", 
                 fontsize=12, color=C_TEXT, fontweight="bold", pad=15)
    ax.tick_params(axis='x', labelsize=10, labelcolor=C_TEXT)
    ax.tick_params(axis='y', labelsize=10, labelcolor=C_TEXT)
    plt.tight_layout()
    fig.savefig(IMAGINI_DIR / "slide4_chunk_sensitivity_type3.png", bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# CONTEGGIO E RILEVAZIONE REALE DEI CHUNK E TABELLA RIASSUNTIVA
# ──────────────────────────────────────────────────────────────────────
def get_chunk_counts():
    files = {
        "chunks_vpc_latest_userguide_markdown-two-stage_400.jsonl": ("Sottogruppo userguide", "400"),
        "chunks_vpc_latest_userguide_markdown-two-stage_800.jsonl": ("Sottogruppo userguide", "800"),
        "chunks_vpc_latest_userguide_markdown-two-stage_1200.jsonl": ("Sottogruppo userguide", "1200"),
        "chunks_vpc_latest_markdown-two-stage_800.jsonl": ("Dataset completo vpc_latest", "800")
    }
    
    counts_table = []
    print("\n" + "="*80)
    print("RILEVAZIONE CONTEGGIO CHUNK NEI FILE DEL WORKSPACE")
    print("="*80)
    
    for fname, (desc, size) in files.items():
        path = CHUNKS_DIR / fname
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            counts_table.append({
                "Dataset/Configurazione": desc,
                "Dimensione Chunk (Caratteri)": size,
                "Strategia": "Markdown Two-Stage",
                "Numero Chunk": lines
            })
            print(f"- {fname:<60} : {lines} chunk")
        else:
            print(f"- [ERRORE] File non trovato: {fname}")
            
    print("="*80 + "\n")
    return pd.DataFrame(counts_table)

# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=== INIZIO GENERAZIONE GRAFICI E VERIFICA DATI PRESENTAZIONE ===")
    
    # 1. Carica i dati
    try:
        data_rerank = load_json("evaluation_results_vpc_latest_markdown-two-stage_800_rerank_fast.json")
        data_standard = load_json("evaluation_results_vpc_latest_markdown-two-stage_800.json")
        data_400 = load_json("evaluation_results_vpc_latest_userguide_markdown-two-stage_400.json")
        data_800 = load_json("evaluation_results_vpc_latest_userguide_markdown-two-stage_800.json")
        data_1200 = load_json("evaluation_results_vpc_latest_userguide_markdown-two-stage_1200.json")
    except FileNotFoundError as e:
        print(f"\n[ERRORE CRITICO] {e}")
        print("Assicurati che tutti i file JSON siano presenti in data/results/")
        return
        
    # 2. Genera grafici
    plot_slide1_global_accuracy(data_rerank, data_standard)
    plot_slide2_reranking(data_rerank)
    plot_slide3_categories(data_rerank)
    plot_slide4_chunk_sensitivity(data_400, data_800, data_1200)
    
    # 3. Tabella dei chunk
    df_chunks = get_chunk_counts()
    
    print("\nTABELLA FORMATTATA PER WALKTHROUGH (MARKDOWN):")
    print(df_chunks.to_markdown(index=False))
    print("\n=== GENERAZIONE COMPLETATA CON SUCCESSO! ===")
    print(f"I grafici sono stati salvati in: {IMAGINI_DIR}")

if __name__ == "__main__":
    main()
