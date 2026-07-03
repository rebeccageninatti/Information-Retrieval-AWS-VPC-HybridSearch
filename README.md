# 🏗️ Vector DB vs Hybrid Search

Confronto empirico tra un database vettoriale puro (ChromaDB), un approccio di ricerca ibrida (vettoriale + BM25) e l'integrazione di un Reranker su documentazione tecnica AWS VPC.

---

## 📌 Descrizione del Progetto

Questo progetto analizza come diverse configurazioni di retrieval impattino su **accuratezza** (Precision@K, Recall@K, MRR, NDCG@K) e **latenza** nella ricerca di informazioni all'interno di documentazione tecnica non strutturata.

Il benchmark si basa sulla documentazione reale di **AWS VPC User Guide** (subset `vpc/latest/userguide`, suddiviso in **1892 chunk** da 800 caratteri ciascuno con la strategia `markdown-two-stage`). La valutazione è effettuata su un dataset sintetico di **415 query di test** generate tramite LLM, suddivise in categorie per evidenziare i comportamenti asimmetrici delle diverse strategie.

Le strategie confrontate sono:
*   **Vector-only**: ricerca semantica con ChromaDB (cosine similarity) basata su embeddings remoti Jina AI (`jina-embeddings-v5-text-small`).
*   **BM25-only**: ricerca lessicale locale basata sulla libreria `rank_bm25`.
*   **Hybrid (RRF)**: fusione dei risultati di entrambi i canali tramite l'algoritmo **Reciprocal Rank Fusion (RRF)** pesato, regolando l'equilibrio con il parametro $\alpha$.
*   **Reranking**: pipeline di recupero a due stadi che raccoglie i migliori candidati e applica il riordinamento fine-grained usando **Jina Reranker v3** (`jina-reranker-v3`).

---

## 🛠️ Stack Tecnologico

*   **Linguaggio**: Python 3.12+ gestito tramite [uv](https://github.com/astral-sh/uv) (ambiente virtuale e dipendenze deterministiche).
*   **Vector DB**: [ChromaDB](https://docs.trychroma.com/) (database vettoriale integrato/locale).
*   **Modelli di Embedding**: API Jina AI.
*   **Reranking**: API Jina Reranker v3 (`jina-reranker-v3`).
*   **Retrieval lessicale**: `rank_bm25` (algoritmo BM25 puro in Python).
*   **Evaluation Engine**: [ranx](https://github.com/AmenRa/ranx) (framework standard per la valutazione di sistemi di Information Retrieval).
*   **Grafica**: `matplotlib` + `seaborn` per la generazione di heatmap, grafici radar e plot comparativi dei risultati.

---

## 📂 Struttura del Progetto

```
archi dati/
├── pyproject.toml                   # Configurazione progetto e dipendenze (uv)
├── PLAN.md                          # Piano di sviluppo dettagliato delle attività
├── README.md                        # Questa guida
├── CHANGELOG.md                     # Cronologia delle modifiche per release
│
├── src/                             # Codice sorgente principale
│   ├── ingestion/                   # Scraper, cleaner e chunker
│   ├── indexing/                    # Embedder, vector store e BM25 index
│   ├── retrieval/                   # Vector, BM25, Hybrid e Rerank retrievers
│   ├── evaluation/                  # Ground truth, benchmark e metriche
│   └── utils/                       # Utility e logging centralizzato
│
├── scripts/                         # Script di automazione della pipeline
│   ├── download_aws_docs.py         # Download della documentazione AWS VPC
│   ├── run_ingestion.py             # Pulizia e chunking a due stadi
│   ├── run_indexing.py              # Generazione embeddings e indici
│   ├── generate_ground_truth.py     # Generazione query di test tramite LLM
│   ├── run_evaluation.py            # Esecuzione benchmark completo con simulazione latenze
│   └── run_evaluation_fast.py       # Esecuzione benchmark con cache abilitata (modalità veloce)
│
└── notebooks/                       # Notebook e script di analisi esplorativa e visualizzazione
    ├── 01_data_exploration.py       # Statistiche descrittive sul corpus
    ├── 02_chunking_analysis.py      # Confronto delle lunghezze dei chunk
    ├── 03_retrieval_qualitative.py  # Test qualitativi interattivi di query
    ├── 04_results_analysis.py       # Generazione di heatmap e radar chart dei risultati
    ├── 05_compare_results.py        # Confronto visivo tra Sottogruppo e Dataset Completo
    ├── generate_csv.py              # Esportazione in CSV delle metriche per categoria
    └── generate_presentation_plots.py # Generazione grafici minimal per la presentazione Canva
```

Per i dettagli formali sul piano di sviluppo, consulta il file [PLAN.md](file:///Users/mattianessi/Downloads/archi%20dati/PLAN.md).

---

## 🚀 Guida di Esecuzione (Quick Start)

Tutte le esecuzioni Python devono essere anticipate da `uv run` per garantire l'uso del virtual environment configurato in [pyproject.toml](file:///Users/mattianessi/Downloads/archi%20dati/pyproject.toml).

### 1. Setup e Installazione
Clona il repository ed esegui la sincronizzazione delle dipendenze:
```bash
git clone <repo-url>
cd "archi dati"
uv sync
```
*Nota: Assicurati di copiare il file `.env.example` in `.env` e inserire le tue chiavi API personali (`JINA_API_KEY`, `GEMINI_API_KEY`).*

### 2. Ingestione dei Dati
Esegui lo scaricamento dei file raw in Markdown e la generazione dei chunk pre-processati:
```bash
# Scarica la guida utente AWS VPC
uv run python scripts/download_aws_docs.py

# Avvia la pulizia e il chunking (Markdown a due stadi, chunk da 800 caratteri)
uv run python scripts/run_ingestion.py
```

### 3. Indicizzazione
Genera gli embeddings e popola il Vector DB e l'indice lessicale BM25:
```bash
uv run python scripts/run_indexing.py
```

### 4. Generazione Ground Truth & Valutazione
Crea il dataset di valutazione sintetico ed esegui il runner di test:
```bash
# Genera query e ground truth tramite LLM (Gemini API)
uv run python scripts/generate_ground_truth.py

# Esegui la suite di valutazione standard con ranx
uv run python scripts/run_evaluation.py

# Esegui la suite di valutazione con Reranking (modalità veloce/cached)
uv run python scripts/run_evaluation_fast.py --rerank
```

### 5. Analisi Grafica e Generazione Report Presentazione
Per analizzare i dati e produrre i grafici comparativi:
```bash
# Esegui l'analisi standard per generare le heatmap e i grafici radar per categoria
uv run python notebooks/04_results_analysis.py

# Confronta il sottogruppo con il dataset completo (crea grafici di confronto)
uv run python notebooks/05_compare_results.py

# Genera la suite completa di grafici pastello per la presentazione (salvati in report/presentazione/immagini/)
uv run python notebooks/generate_presentation_plots.py

# Genera il file CSV riassuntivo dei risultati per categoria
uv run python notebooks/generate_csv.py
```

---

## 📊 Risultati del Benchmark (First-Stage Retrieval)

Di seguito sono riportati i risultati complessivi ottenuti su un set di **415 query di test** senza l'uso del Reranker:

| Strategia | NDCG@10 | Recall@10 | Recall@5 | Precision@10 | MRR | Latenza Media (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25-only** | 0.4972 | 0.6795 | 0.5819 | 0.0704 | 0.4481 | **5.1** |
| **Vector-only** | 0.6173 | 0.8048 | 0.6880 | 0.0836 | 0.5639 | **461.9** |
| **Hybrid-RRF ($\alpha=0.1$)** | 0.5295 | 0.7084 | 0.6241 | 0.0735 | 0.4804 | 466.8 |
| **Hybrid-RRF ($\alpha=0.3$)** | 0.5718 | 0.7687 | 0.6663 | 0.0798 | 0.5169 | 466.7 |
| **Hybrid-RRF ($\alpha=0.5$)** | 0.5943 | 0.7964 | 0.6867 | 0.0827 | 0.5365 | 467.2 |
| **Hybrid-RRF ($\alpha=0.7$)** | 0.6052 | 0.8205 | **0.7084** | 0.0853 | 0.5432 | 467.3 |
| **Hybrid-RRF ($\alpha=0.9$)** | **0.6220** | **0.8253** | 0.6976 | **0.0858** | **0.5636** | 468.2 |

### Conclusioni Chiave (First-Stage)
1.  **Vettoriale vs Lessicale**: La ricerca vettoriale pura (`Vector-only`) surclassa nettamente BM25 su tutte le metriche principali ($0.6173$ vs $0.4972$ NDCG@10), catturando il significato semantico e gestendo sinonimi o parafrasi che BM25 fallisce ad agganciare.
2.  **Sinergia Ibrida**: La fusione **Hybrid-RRF con $\alpha=0.9$** ottiene la prestazione migliore in assoluto del benchmark ($0.6220$ NDCG@10 e $82.53\%$ Recall@10). Il contributo lessicale del 10% funge da booster per chunk contenenti codici o nomenclature tecniche esatte.
3.  **Latenza di Produzione**: Con la detrazione dello sleep accumulato per rate-limiting, la ricerca ibrida richiede solo **~6-7 ms in più** rispetto a quella vettoriale pura (~468.2 ms vs ~461.9 ms), rendendo l'approccio ibrido la scelta ideale per la produzione.

---

## 🏆 Impatto del Reranking (Second-Stage Retrieval)

Di seguito sono riportati i risultati complessivi nel confronto con il **Jina Reranker v3** (misurati in **Fast Mode** con embedding di query pre-calcolati per evidenziare il guadagno qualitativo delle metriche):

| Strategia | NDCG@10 | Recall@10 | Recall@5 | Precision@10 | MRR | Latenza Media (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25-only** | 0.5289 | 0.7325 | 0.6181 | 0.0764 | 0.4713 | 14.9 |
| **BM25-only + Rerank** | **0.6971** (+31.8%) | **0.8277** | **0.7651** | **0.0863** | **0.6583** | 1390.8 |
| **Vector-only** | 0.6001 | 0.7783 | 0.6988 | 0.0810 | 0.5488 | **2.6** |
| **Vector-only + Rerank** | **0.7255** (+20.9%) | **0.8446** | **0.7988** | **0.0877** | **0.6895** | 1452.2 |
| **Hybrid-RRF ($\alpha=0.5$)** | 0.6161 | 0.8120 | 0.7205 | 0.0846 | 0.5596 | 23.6 |
| **Hybrid-RRF ($\alpha=0.5$) + Rerank** | **0.7274** (+18.1%) | **0.8711** | **0.8120** | **0.0904** | **0.6844** | 1130.1 |

### Conclusioni Chiave (Reranking)
1. **Incremento delle Metriche**: L'applicazione di Jina Reranker v3 determina incrementi notevoli e trasversali su tutti i metodi di ricerca. L'accuratezza NDCG@10 migliora del **+31.8%** per BM25, del **+20.9%** per Vector-only e del **+18.1%** per l'approccio ibrido, che tocca il valore massimo di **0.7274**.
2. **Posizionamento dei Risultati (MRR)**: Anche l'MRR (Mean Reciprocal Rank) subisce un salto significativo, dimostrando che le risposte più pertinenti vengono spinte in cima alla graduatoria.
3. **Trade-off di Latenza**: A fronte del netto miglioramento qualitativo, l'overhead introdotto dalle chiamate API di rete al Reranker remoto porta i tempi di risposta a circa **1.1 - 1.4 secondi**. In scenari reali, la scelta della pipeline a due stadi dipende dai vincoli operativi del sistema (tempo reale vs precisione assoluta).
