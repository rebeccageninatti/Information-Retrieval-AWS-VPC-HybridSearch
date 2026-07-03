# AWS VPC Search Console - Flask Server

Questo modulo implementa un server web Flask che fornisce una console di ricerca per testare e confrontare le diverse strategie di Information Retrieval implementate nel progetto (**Vector DB**, **BM25**, ed **Hybrid Search** tramite RRF).

L'interfaccia utente è progettata con uno stile minimalista, caratterizzato da toni pastello (verde salvia, azzurro carta da zucchero, lavanda) e un layout chiaro.

## Funzionalità

1. **Selezione Dinamica del Database**: Permette di effettuare ricerche sui chunk di taglia **400**, **800** o **1200** caratteri. Gli indici corrispondenti vengono caricati dinamicamente in memoria al momento della prima query.
2. **Confronto Strategie**:
   - **Vettoriale pura (Semantica)**: Ricerca densa basata sulla similarità del coseno in ChromaDB (modello `jina` o `gemini` configurato nel file `.env`).
   - **Lessicale pura (BM25)**: Ricerca sparsa a match di parole chiave basata su `rank-bm25`.
   - **Ibrida (RRF)**: Fusione dei risultati di entrambi i motori usando l'algoritmo Reciprocal Rank Fusion pesato tramite il parametro $\alpha$.
3. **Parametro Alpha Regolabile**: Uno slider consente di controllare l'importanza relativa del motore semantico rispetto a quello lessicale per la ricerca ibrida ($\alpha = 0.5$ corrisponde al bilanciamento equo).
4. **Visualizzazione Dettagliata**: Ogni risultato mostra l'header path (il percorso del documento), il nome del file sorgente, il punteggio finale e, nel caso della ricerca ibrida, i singoli rank originali (es. `Vector Rank: #2, BM25 Rank: #5`).

## Avvio del Server

Assicurarsi di trovarsi nella root del progetto `archi dati`.

1. **Installazione delle Dipendenze**:
   La dipendenza `flask` è stata aggiunta al file `pyproject.toml`. Sincronizzare l'ambiente tramite `uv`:
   ```bash
   uv sync
   ```

2. **Verifica delle Variabili d'Ambiente**:
   Il file `.env` nella root del progetto deve contenere le chiavi API necessarie (es: `JINA_API_KEY` o `GEMINI_API_KEY`) e la configurazione dell'embedder di default (`DEFAULT_EMBEDDER`).

3. **Verifica degli Indici**:
   Assicurarsi che gli indici siano stati creati per le dimensioni di interesse eseguendo il caricamento e l'indicizzazione preventiva. Ad esempio:
   - Gli indici BM25 devono risiedere in `data/indices/bm25_index_*.pkl`
   - Il database ChromaDB deve trovarsi in `data/indices/chroma_db`

4. **Avvio dell'applicazione**:
   Avviare il server Flask tramite `uv`:
   ```bash
   uv run python flask_server/app.py
   ```

5. **Accedere all'Interfaccia**:
   Aprire il browser all'indirizzo:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)
