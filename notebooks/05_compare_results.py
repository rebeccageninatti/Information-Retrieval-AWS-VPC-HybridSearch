"""
Script per il confronto visivo dei risultati di valutazione (Sottogruppo vs Dataset Completo).
Legge i risultati di valutazione di 'vpc_latest_userguide' e 'vpc_latest' e genera grafici comparativi in report/figures/.
"""

import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

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
    "Vector-only": "#AED6F1",          # Azzurro pastello soft
    "BM25-only": "#F5B7B1",            # Rosa/corallo pastello soft
    "Hybrid-RRF (α=0.1)": "#F5EEF8",   # Lilla pastello chiarissimo
    "Hybrid-RRF (α=0.3)": "#EBDEF0",   # Viola pastello chiaro
    "Hybrid-RRF (α=0.5)": "#D4EFDF",   # Verde pastello chiaro
    "Hybrid-RRF (α=0.7)": "#D1F2EB",   # Tealo pastello chiaro
    "Hybrid-RRF (α=0.9)": "#FCF3CF"    # Giallo pastello chiaro
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_dataframes(data: dict, dataset_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = data["results"]
    overall_rows = []
    category_rows = []

    for strategy, strat_data in results.items():
        latency = strat_data["avg_latency_ms"]
        metrics_by_cat = strat_data["metrics"]

        # Overall Row
        overall_metrics = metrics_by_cat.get("overall", {})
        overall_row = {
            "Dataset": dataset_label,
            "Strategy": strategy,
            "Latency (ms)": latency,
        }
        for metric, val in overall_metrics.items():
            overall_row[metric.upper()] = val
        overall_rows.append(overall_row)

        # Category Rows
        for cat, cat_metrics in metrics_by_cat.items():
            if cat == "overall":
                continue
            cat_row = {
                "Dataset": dataset_label,
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

def generate_comparison_heatmap(df_sub: pd.DataFrame, df_full: pd.DataFrame):
    """
    Genera due heatmap affiancate (Subgroup vs Full) per le metriche overall.
    """
    metric_cols = [col for col in df_sub.columns if col not in ["Dataset", "Strategy", "Latency (ms)"]]
    
    # Ordiniamo le righe nello stesso modo per coerenza visiva
    strategies_order = [
        "BM25-only",
        "Vector-only",
        "Hybrid-RRF (α=0.1)",
        "Hybrid-RRF (α=0.3)",
        "Hybrid-RRF (α=0.5)",
        "Hybrid-RRF (α=0.7)",
        "Hybrid-RRF (α=0.9)"
    ]
    
    df_plot_sub = df_sub.set_index("Strategy").reindex(strategies_order)[metric_cols]
    df_plot_full = df_full.set_index("Strategy").reindex(strategies_order)[metric_cols]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    cmap = sns.light_palette("#2B4C7E", as_cmap=True)

    # Subgroup Heatmap
    sns.heatmap(df_plot_sub, annot=True, cmap=cmap, fmt=".4f", cbar=False, linewidths=0.5, ax=axes[0])
    axes[0].set_title("Sottogruppo (userguide - 1892 Chunks)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_ylabel("Strategia", fontsize=12)
    axes[0].set_xlabel("Metrica", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # Full Heatmap
    sns.heatmap(df_plot_full, annot=True, cmap=cmap, fmt=".4f", cbar=True, linewidths=0.5, ax=axes[1])
    axes[1].set_title("Dataset Completo (vpc_latest - 4427 Chunks)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Metrica", fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle("Heatmap Comparativa delle Metriche di Retrieval (Overall)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    plot_path = FIGURES_DIR / "retrieval_metrics_heatmap.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Heatmap comparativa salvata in: {plot_path}")
    plt.close()
def generate_comparison_category_bar_plot(df_categories_sub: pd.DataFrame, df_categories_full: pd.DataFrame):
    """
    Genera un singolo grafico a barre per il dataset completo confrontando NDCG@10 per categoria di query.
    """
    target_strategies = [
        "Vector-only",
        "BM25-only",
        "Hybrid-RRF (α=0.5)",
        "Hybrid-RRF (α=0.9)"
    ]
    df_full_filt = df_categories_full[df_categories_full["Strategy"].isin(target_strategies)]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Order of categories
    cat_order = ["Needle", "Paraphrase", "Conceptual", "Ambiguous", "Multi-hop"]

    sns.barplot(
        x="Category",
        y="NDCG@10",
        hue="Strategy",
        data=df_full_filt,
        order=cat_order,
        palette=PASTEL_PALETTE,
        ax=ax
    )
    
    ax.set_xlabel("Categoria Query", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Strategia", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    
    plt.tight_layout()

    plot_path = FIGURES_DIR / "ndcg_by_category.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Grafico categorie salvato in: {plot_path}")
    plt.close()


def generate_comparison_latency_accuracy_tradeoff(df_sub: pd.DataFrame, df_full: pd.DataFrame):
    """
    Genera uno scatter plot che mostra il trade-off latenza-accuratezza per entrambi i dataset,
    unendo con linee tratteggiate e frecce le stesse strategie per evidenziare il trend di scalabilità.
    """
    plt.figure(figsize=(12, 8))

    strategies = [
        "BM25-only",
        "Vector-only",
        "Hybrid-RRF (α=0.1)",
        "Hybrid-RRF (α=0.3)",
        "Hybrid-RRF (α=0.5)",
        "Hybrid-RRF (α=0.7)",
        "Hybrid-RRF (α=0.9)"
    ]

    # Plot lines/arrows showing shift from subgroup to full
    for strat in strategies:
        row_sub = df_sub[df_sub["Strategy"] == strat].iloc[0]
        row_full = df_full[df_full["Strategy"] == strat].iloc[0]
        
        x_sub, y_sub = row_sub["Latency (ms)"], row_sub["NDCG@10"]
        x_full, y_full = row_full["Latency (ms)"], row_full["NDCG@10"]
        
        # Disegna una freccia direzionale dal sottogruppo al completo
        plt.annotate(
            "",
            xy=(x_full, y_full),
            xytext=(x_sub, y_sub),
            arrowprops=dict(arrowstyle="->", color="gray", linestyle="--", alpha=0.6, lw=1.5, shrinkA=5, shrinkB=5)
        )

    # Disegna i punti del Sottogruppo (Cerchi)
    for strat in strategies:
        row = df_sub[df_sub["Strategy"] == strat].iloc[0]
        plt.scatter(
            row["Latency (ms)"],
            row["NDCG@10"],
            color=PASTEL_PALETTE[strat],
            marker='o',
            s=160,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.85,
            label=f"{strat} (Sub)" if strat == "Vector-only" or strat == "BM25-only" or strat == "Hybrid-RRF (α=0.5)" else ""
        )

    # Disegna i punti del Dataset Completo (Quadrati)
    for strat in strategies:
        row = df_full[df_full["Strategy"] == strat].iloc[0]
        plt.scatter(
            row["Latency (ms)"],
            row["NDCG@10"],
            color=PASTEL_PALETTE[strat],
            marker='s',
            s=160,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            label=f"{strat} (Full)" if strat == "Vector-only" or strat == "BM25-only" or strat == "Hybrid-RRF (α=0.5)" else ""
        )

    # Aggiungi etichette di testo vicino ai punti per chiarezza
    for strat in strategies:
        row_sub = df_sub[df_sub["Strategy"] == strat].iloc[0]
        row_full = df_full[df_full["Strategy"] == strat].iloc[0]
        
        # Etichetta per il sottogruppo
        plt.text(
            row_sub["Latency (ms)"] - 12,
            row_sub["NDCG@10"] + 0.005,
            f"{strat} (Sub)",
            fontsize=8,
            ha='right',
            alpha=0.7
        )
        # Etichetta per il dataset completo
        plt.text(
            row_full["Latency (ms)"] + 12,
            row_full["NDCG@10"] - 0.008,
            f"{strat} (Full)",
            fontsize=8,
            ha='left',
            alpha=0.85,
            fontweight="bold"
        )

    # Custom legend representing shapes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Sottogruppo (userguide - 1892 Ch.)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Dataset Completo (vpc_latest - 4427 Ch.)', markeredgecolor='black'),
        Line2D([0], [0], linestyle='--', color='gray', label='Direzione scalabilità (Sub -> Full)')
    ]
    
    # Aggiungiamo anche le strategie principali nella legenda
    for strat in ["Vector-only", "BM25-only", "Hybrid-RRF (α=0.5)", "Hybrid-RRF (α=0.9)"]:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=PASTEL_PALETTE[strat], markersize=10, label=strat))
        
    plt.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    plt.title("Trade-off Latenza vs. Accuratezza (NDCG@10): Effetto del Ridimensionamento dell'Indice", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Latenza Media di Query (ms)", fontsize=12)
    plt.ylabel("NDCG@10", fontsize=12)
    
    # Limiti x ed y
    plt.xlim(-50, df_full["Latency (ms)"].max() + 150)
    plt.ylim(df_sub["NDCG@10"].min() - 0.03, 0.65)
    
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plot_path = FIGURES_DIR / "latency_vs_accuracy.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Scatter plot latenza vs accuratezza salvato in: {plot_path}")
    plt.close()

def generate_comparison_latency_bar_plot(df_sub: pd.DataFrame, df_full: pd.DataFrame):
    """
    Genera un istogramma a barre raggruppate per confrontare le latenze medie.
    """
    strategies_order = [
        "BM25-only",
        "Vector-only",
        "Hybrid-RRF (α=0.1)",
        "Hybrid-RRF (α=0.3)",
        "Hybrid-RRF (α=0.5)",
        "Hybrid-RRF (α=0.7)",
        "Hybrid-RRF (α=0.9)"
    ]

    # Prepara un unico dataframe per seaborn
    df_latency_sub = df_sub[["Strategy", "Latency (ms)"]].copy()
    df_latency_sub["Dataset"] = "Sottogruppo (userguide)"
    
    df_latency_full = df_full[["Strategy", "Latency (ms)"]].copy()
    df_latency_full["Dataset"] = "Dataset Completo (vpc_latest)"
    
    df_plot = pd.concat([df_latency_sub, df_latency_full], ignore_index=True)
    df_plot["Strategy"] = pd.Categorical(df_plot["Strategy"], categories=strategies_order, ordered=True)

    plt.figure(figsize=(14, 7))
    
    # Custom palette per i due dataset
    dataset_colors = {
        "Sottogruppo (userguide)": "#A5D6A7",      # Verde pastello chiaro
        "Dataset Completo (vpc_latest)": "#EF9A9A"  # Rosso/rosa pastello chiaro
    }

    ax = sns.barplot(
        x="Strategy",
        y="Latency (ms)",
        hue="Dataset",
        data=df_plot,
        palette=dataset_colors,
        edgecolor="gray",
        linewidth=0.5
    )

    plt.title("Latenza Media delle Query: Sottogruppo vs. Dataset Completo", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Strategia di Retrieval", fontsize=12)
    plt.ylabel("Latenza Media (ms)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(title="Dataset", loc="upper left")

    # Aggiungi i valori sopra le barre
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f} ms",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=8.5,
                fontweight="semibold",
                alpha=0.85
            )

    plt.tight_layout()
    
    plot_path = FIGURES_DIR / "latency_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Grafico confronto latenze salvato in: {plot_path}")
    plt.close()

def main():
    print("=== INIZIO GENERAZIONE GRAFICI COMPARATIVI ===")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carica i dati
    path_sub = RESULTS_DIR / "evaluation_results_vpc_latest_userguide_markdown-two-stage_800.json"
    path_full = RESULTS_DIR / "evaluation_results_vpc_latest_markdown-two-stage_800.json"

    data_sub = load_json(path_sub)
    data_full = load_json(path_full)

    # 2. Prepara i DataFrame
    df_sub_overall, df_sub_cat = prepare_dataframes(data_sub, "Subgroup")
    df_full_overall, df_full_cat = prepare_dataframes(data_full, "Full")

    # 3. Genera e salva i grafici comparativi
    generate_comparison_heatmap(df_sub_overall, df_full_overall)
    generate_comparison_category_bar_plot(df_sub_cat, df_full_cat)
    generate_comparison_latency_accuracy_tradeoff(df_sub_overall, df_full_overall)
    generate_comparison_latency_bar_plot(df_sub_overall, df_full_overall)

    print("=== GENERAZIONE COMPLETATA CON SUCCESSO ===")

if __name__ == "__main__":
    main()
