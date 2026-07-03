import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_FILE = PROJECT_ROOT / "ndcg_by_category_full.csv"

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Carica i risultati del dataset completo (800 character chunk size)
    path_full = RESULTS_DIR / "evaluation_results_vpc_latest_markdown-two-stage_800.json"
    data = load_json(path_full)
    
    results = data["results"]
    category_rows = []

    target_strategies = [
        "Vector-only",
        "BM25-only",
        "Hybrid-RRF (α=0.5)",
        "Hybrid-RRF (α=0.9)"
    ]

    for strategy, strat_data in results.items():
        if strategy not in target_strategies:
            continue
        metrics_by_cat = strat_data["metrics"]

        for cat, cat_metrics in metrics_by_cat.items():
            if cat == "overall":
                continue
            cat_row = {
                "Strategy": strategy,
                "Category": cat.capitalize(),
                "NDCG@10": cat_metrics.get("ndcg@10", 0.0)
            }
            category_rows.append(cat_row)

    df = pd.DataFrame(category_rows)
    
    # Ordine delle categorie desiderato
    cat_order = ["Needle", "Paraphrase", "Conceptual", "Ambiguous", "Multi-hop"]
    
    # Pivot per avere le strategie come colonne ed i tipi di query come righe
    df_pivot = df.pivot(index="Category", columns="Strategy", values="NDCG@10")
    df_pivot = df_pivot.reindex(cat_order)
    
    # Resetta l'indice per avere la colonna Category
    df_pivot = df_pivot.reset_index()
    
    # Salva in CSV
    df_pivot.to_csv(OUTPUT_FILE, index=False)
    print(f"File CSV creato con successo in: {OUTPUT_FILE}")
    print(df_pivot)

if __name__ == "__main__":
    main()
