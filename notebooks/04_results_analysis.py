"""
Script per l'analisi visiva dei risultati di valutazione (Fase 4: Risultati).
Legge evaluation_results.json, genera tabelle riassuntive e salva i grafici comparativi in report/figures/.
"""

import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import EVAL_RESULTS_FILE, PROJECT_ROOT
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Configurazione dello stile globale per LaTeX e colori pastello minimali
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Georgia", "Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Palette pastello personalizzata per garantire coerenza tra i vari grafici
PASTEL_PALETTE = {
    "Vector-only": "#7FB3D5",                  # Azzurro pastello soft
    "Vector-only + Rerank": "#2980B9",         # Blu pastello scuro
    "BM25-only": "#F1948A",                    # Rosa/corallo pastello soft
    "BM25-only + Rerank": "#C0392B",           # Rosso pastello scuro
    "Hybrid-RRF (α=0.1)": "#E8DAEF",           # Lilla pastello chiarissimo
    "Hybrid-RRF (α=0.3)": "#D2B4DE",           # Viola pastello chiaro
    "Hybrid-RRF (α=0.5)": "#82E0AA",           # Verde pastello chiaro
    "Hybrid-RRF (α=0.5) + Rerank": "#27AE60", # Verde pastello scuro
    "Hybrid-RRF (α=0.7)": "#A2D9CE",           # Tealo pastello chiaro
    "Hybrid-RRF (α=0.9)": "#F7DC6F"            # Giallo pastello chiaro
}



def load_results(is_fast: bool = False) -> dict:
    """
    Carica il file dei risultati della valutazione con reranking.
    """
    results_path = EVAL_RESULTS_FILE.with_name(EVAL_RESULTS_FILE.stem + "_rerank.json")
    if is_fast:
        results_path = EVAL_RESULTS_FILE.with_name(EVAL_RESULTS_FILE.stem + "_rerank_fast.json")

    if not results_path.exists():
        raise FileNotFoundError(
            f"File dei risultati non trovato: {results_path}. "
            f"Esegui prima lo script di benchmark opportuno per produrlo."
        )

    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataframe(report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara due DataFrame pandas a partire dai risultati nidificati:
    1. df_overall: metriche overall e latenze per ciascuna strategia.
    2. df_categories: metriche disaggregate per categoria e strategia.
    """
    results = report["results"]
    
    overall_rows = []
    category_rows = []

    for strategy, data in results.items():
        latency = data["avg_latency_ms"]
        metrics_by_cat = data["metrics"]

        # 1. Overall Row
        overall_metrics = metrics_by_cat.get("overall", {})
        overall_row = {
            "Strategy": strategy,
            "Latency (ms)": latency,
        }
        for metric, val in overall_metrics.items():
            overall_row[metric.upper()] = val
        overall_rows.append(overall_row)

        # 2. Category Rows
        for cat, cat_metrics in metrics_by_cat.items():
            if cat == "overall":
                continue
            cat_row = {
                "Strategy": strategy,
                "Category": cat.capitalize(),
                "Latency (ms)": latency,
            }
            for metric, val in cat_metrics.items():
                cat_row[metric.upper()] = val
            category_rows.append(cat_row)

    df_overall = pd.DataFrame(overall_rows)
    df_categories = pd.DataFrame(category_rows)
    
    return df_overall, df_categories


def generate_overall_heatmap(df_overall: pd.DataFrame, figures_dir: Path, is_fast: bool = False):
    """
    Genera una heatmap comparativa delle metriche complessive (overall) per ogni strategia.
    """
    # Seleziona le colonne per le metriche, escludendo Strategy e Latency
    metric_cols = [col for col in df_overall.columns if col not in ["Strategy", "Latency (ms)"]]
    
    # Imposta l'indice su Strategy
    df_plot = df_overall.set_index("Strategy")[metric_cols]

    plt.figure(figsize=(12, 6))
    # Colormap pastello/soft slate blue
    cmap = sns.light_palette("#2B4C7E", as_cmap=True)
    sns.heatmap(df_plot, annot=True, cmap=cmap, fmt=".4f", cbar=True, linewidths=0.5)
    plt.title("Comparison Heatmap of Retrieval Strategies (Overall Metrics)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Strategy", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    suffix = "_fast" if is_fast else ""
    plot_path = figures_dir / f"retrieval_metrics_heatmap{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Heatmap complessiva salvata in: {plot_path}")
    plt.close()


def generate_category_bar_plot(df_categories: pd.DataFrame, figures_dir: Path, is_fast: bool = False):
    """
    Genera un radar chart comparativo composto da tre sottografici (BM25, Vector, Hybrid α=0.5)
    mostrando NDCG@10 per ciascuna delle 5 categorie di query con e senza Rerank.
    """
    import numpy as np
    
    # 5 Categorie in ordine
    cat_order = ["Needle", "Paraphrase", "Conceptual", "Ambiguous", "Multi-hop"]
    num_vars = len(cat_order)

    # Calcola gli angoli del radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # chiude il cerchio

    # Configurazioni dei 3 sottografici
    configs = [
        {
            "title": "BM25 Retrieval",
            "base": "BM25-only",
            "rerank": "BM25-only + Rerank"
        },
        {
            "title": "Vector Retrieval",
            "base": "Vector-only",
            "rerank": "Vector-only + Rerank"
        },
        {
            "title": "Hybrid Retrieval (α=0.5)",
            "base": "Hybrid-RRF (α=0.5)",
            "rerank": "Hybrid-RRF (α=0.5) + Rerank"
        }
    ]

    # Inizializza la figura con coordinate polari
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), subplot_kw=dict(projection='polar'))

    # Colori pastello minimali e coordinati
    color_base_line = "#5DADE2"   # Azzurro pastello scuro per bordo base
    color_base_fill = "#AED6F1"   # Azzurro pastello chiaro per riempimento base
    color_rerank_line = "#EC7063" # Rosso/rosa pastello scuro per bordo rerank
    color_rerank_fill = "#F5B7B1" # Rosa pastello chiaro per riempimento rerank

    for i, cfg in enumerate(configs):
        ax = axes[i]
        
        # Estrai valori per Base
        base_df = df_categories[df_categories["Strategy"] == cfg["base"]]
        if not base_df.empty:
            val_base = []
            for cat in cat_order:
                val = base_df[base_df["Category"] == cat]["NDCG@10"].values
                val_base.append(val[0] if len(val) > 0 else 0.0)
            val_base += val_base[:1]
        else:
            val_base = [0.0] * (num_vars + 1)

        # Estrai valori per Rerank
        rerank_df = df_categories[df_categories["Strategy"] == cfg["rerank"]]
        if not rerank_df.empty:
            val_rerank = []
            for cat in cat_order:
                val = rerank_df[rerank_df["Category"] == cat]["NDCG@10"].values
                val_rerank.append(val[0] if len(val) > 0 else 0.0)
            val_rerank += val_rerank[:1]
        else:
            val_rerank = [0.0] * (num_vars + 1)

        # Disegna la linea ed il riempimento per Base
        ax.plot(angles, val_base, color=color_base_line, linewidth=1.5, linestyle="--", label="Senza Rerank")
        ax.fill(angles, val_base, color=color_base_fill, alpha=0.25)

        # Disegna la linea ed il riempimento per Rerank
        ax.plot(angles, val_rerank, color=color_rerank_line, linewidth=2.0, label="Con Rerank")
        ax.fill(angles, val_rerank, color=color_rerank_fill, alpha=0.35)

        # Imposta le etichette delle categorie sui 5 spigoli
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cat_order, fontsize=12, fontweight="bold")

        # Configura la griglia radiale (valori da 0 a 1)
        ax.set_ylim(0.0, 1.0)
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="gray", fontsize=10, fontweight="semibold")
        ax.grid(True, linestyle=":", alpha=0.6)

        # Titolo minimalista del sottografico
        ax.set_title(cfg["title"], fontsize=14, fontweight="bold", pad=15)

        # Mostra la legenda solo sul terzo grafico per non affollare la figura
        if i == 2:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), frameon=True, facecolor="white", edgecolor="none", fontsize=11)

    plt.tight_layout()

    suffix = "_fast" if is_fast else ""
    plot_path = figures_dir / f"ndcg_by_category{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Grafico radar delle categorie salvato in: {plot_path}")
    plt.close()


def generate_latency_accuracy_tradeoff(df_overall: pd.DataFrame, figures_dir: Path, is_fast: bool = False):
    """
    Genera uno scatter plot che illustra il compromesso tra latenza (ms) e accuratezza (NDCG@10).
    """
    plt.figure(figsize=(10, 6))
    
    # Disegna i punti basandoti sulla palette pastello coerente
    sns.scatterplot(
        x="Latency (ms)",
        y="NDCG@10",
        hue="Strategy",
        style="Strategy",
        data=df_overall,
        s=180,
        palette=PASTEL_PALETTE,
        edgecolor="gray",
        linewidth=0.5
    )

    # Aggiunge etichette di testo vicino ai punti
    for idx, row in df_overall.iterrows():
        plt.text(
            row["Latency (ms)"] + (df_overall["Latency (ms)"].max() * 0.015),
            row["NDCG@10"] - 0.005,
            row["Strategy"],
            fontsize=9,
            alpha=0.85,
            weight="semibold"
        )

    plt.title("Latency vs. Accuracy (NDCG@10) Trade-off", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Average Latency (ms)", fontsize=12)
    plt.ylabel("NDCG@10", fontsize=12)
    
    # Estende leggermente i limiti per far stare i testi
    plt.xlim(0, df_overall["Latency (ms)"].max() * 1.35)
    plt.ylim(df_overall["NDCG@10"].min() - 0.04, min(1.0, df_overall["NDCG@10"].max() + 0.04))
    
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    suffix = "_fast" if is_fast else ""
    plot_path = figures_dir / f"latency_vs_accuracy{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Scatter plot latenza vs accuratezza salvato in: {plot_path}")
    plt.close()


def generate_precision_recall_curves(df_overall: pd.DataFrame, figures_dir: Path, is_fast: bool = False):
    """
    Genera un grafico Precision-Recall a vari livelli di K (3, 5, 10).
    """
    plt.figure(figsize=(10, 6))
    
    # Seleziona le strategie di interesse per il grafico
    target_strategies = [
        "Vector-only",
        "Vector-only + Rerank",
        "BM25-only",
        "Hybrid-RRF (α=0.5)",
        "Hybrid-RRF (α=0.5) + Rerank"
    ]
    
    df_filtered = df_overall[df_overall["Strategy"].isin(target_strategies)]
    
    for idx, row in df_filtered.iterrows():
        strategy = row["Strategy"]
        # Estrae i punti (Recall, Precision) per K=3, 5, 10
        recalls = [row["RECALL@3"], row["RECALL@5"], row["RECALL@10"]]
        precisions = [row["PRECISION@3"], row["PRECISION@5"], row["PRECISION@10"]]
        k_values = [3, 5, 10]
        
        # Colore coerente dalla palette pastello
        color = PASTEL_PALETTE.get(strategy, "#85929E")
        
        # Disegna la linea per questa strategia
        plt.plot(recalls, precisions, marker='o', label=strategy, linewidth=2, markersize=8, color=color)
        
        # Aggiunge etichette di K vicino ai punti
        for r, p, k in zip(recalls, precisions, k_values):
            plt.text(r + 0.003, p + 0.003, f"K={k}", fontsize=8, alpha=0.8)
            
    plt.title("Precision-Recall Curve at Different K Levels (K=3, 5, 10)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Strategy", frameon=True)
    plt.tight_layout()
    
    suffix = "_fast" if is_fast else ""
    plot_path = figures_dir / f"precision_recall_curves{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Grafico Precision-Recall salvato in: {plot_path}")
    plt.close()


def generate_latency_comparison(df_overall: pd.DataFrame, figures_dir: Path, is_fast: bool = False):
    """
    Genera un bar plot comparativo per le latenze medie di ciascuna strategia.
    """
    plt.figure(figsize=(12, 6))
    
    # Ordina per latenza crescente
    df_sorted = df_overall.sort_values(by="Latency (ms)")
    
    sns.barplot(
        x="Strategy",
        y="Latency (ms)",
        data=df_sorted,
        palette=PASTEL_PALETTE
    )
    
    plt.title("Average Query Latency by Retrieval Strategy", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Retrieval Strategy", fontsize=12)
    plt.ylabel("Average Latency (ms)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    
    # Aggiunge i valori sopra le barre
    for i, val in enumerate(df_sorted["Latency (ms)"]):
        plt.text(i, val + (df_sorted["Latency (ms)"].max() * 0.01), f"{val:.1f} ms", ha='center', fontsize=9, weight="bold", alpha=0.85)

    plt.tight_layout()
    
    suffix = "_fast" if is_fast else ""
    plot_path = figures_dir / f"latency_comparison{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Grafico confronto latenze salvato in: {plot_path}")
    plt.close()


def run_analysis(is_fast: bool = False):
    logger.info(f"=== INIZIO ANALISI VISIVA DEI RISULTATI ({'FAST' if is_fast else 'STANDARD'}) ===")
    
    # 1. Carica i dati
    try:
        report = load_results(is_fast)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Crea la cartella delle figure se non esiste
    figures_dir = PROJECT_ROOT / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Prepara i DataFrame
    df_overall, df_categories = prepare_dataframe(report)

    # 3. Genera grafici
    generate_overall_heatmap(df_overall, figures_dir, is_fast)
    generate_category_bar_plot(df_categories, figures_dir, is_fast)
    generate_latency_accuracy_tradeoff(df_overall, figures_dir, is_fast)
    generate_precision_recall_curves(df_overall, figures_dir, is_fast)
    generate_latency_comparison(df_overall, figures_dir, is_fast)
    
    logger.info("=== ANALISI VISIVA COMPLETATA CON SUCCESSO ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analizza i risultati del benchmark.")
    parser.add_argument("--fast", action="store_true", help="Analizza i risultati della versione fast.")
    args = parser.parse_args()
    
    run_analysis(is_fast=args.fast)

