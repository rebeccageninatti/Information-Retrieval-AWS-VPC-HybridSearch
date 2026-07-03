// Stato globale dell'applicazione
let systemStats = null;
let isExampleSelected = false;

// Esegui all'avvio della pagina
document.addEventListener("DOMContentLoaded", () => {
    fetchSystemStats();
    toggleAlphaSlider();
    
    // Gestione visibilità delle domande di esempio alla digitazione
    const queryInput = document.getElementById("query-input");
    if (queryInput) {
        queryInput.addEventListener("input", handleQueryInput);
    }
});

// Gestore per la digitazione nella barra di ricerca
function handleQueryInput() {
    const queryInput = document.getElementById("query-input");
    const container = document.getElementById("example-queries-container");
    if (!queryInput || !container) return;
    
    // Se l'utente digita manualmente, reimpostiamo la flag isExampleSelected
    isExampleSelected = false;
    
    if (queryInput.value.trim() === "") {
        container.classList.remove("hidden");
    } else {
        container.classList.add("hidden");
    }
}

// Aggiorna l'etichetta del valore alpha
function updateAlphaLabel(val) {
    const floatVal = parseFloat(val);
    document.getElementById("alpha-value").innerText = floatVal.toFixed(2);
}

// Mostra o nasconde lo slider dell'alpha in base alla strategia scelta
function toggleAlphaSlider() {
    const strategy = document.getElementById("strategy-select").value;
    const alphaGroup = document.getElementById("alpha-group");
    
    if (strategy === "hybrid-rrf") {
        alphaGroup.classList.remove("hidden");
    } else {
        alphaGroup.classList.add("hidden");
    }
}

// Gestione del cambio del dataset
function onDatasetChange() {
    if (!systemStats) return;
    
    const selectedDataset = document.getElementById("dataset-select").value;
    
    // Aggiorna la select delle taglie dei chunk in base ai database presenti per questo dataset
    updateChunkSizeSelect(selectedDataset);
    
    // Aggiorna etichetta dataset nello stato del sistema
    document.getElementById("stat-dataset").innerText = formatDatasetName(selectedDataset);
    
    // Aggiorna lo stato dei database (ready/missing)
    checkDatabaseStatus();
    
    // Ricalcola lo stato della cache in RAM per questo specifico dataset
    updateCacheStatus(selectedDataset);
    
    // Aggiorna le query di esempio per il dataset corrente
    fetchExampleQueries();
}

// Gestione del cambio della taglia dei chunk
function onChunkSizeChange() {
    checkDatabaseStatus();
    fetchExampleQueries();
}

// Seleziona una query di esempio
function selectExample(queryText) {
    const queryInput = document.getElementById("query-input");
    const container = document.getElementById("example-queries-container");
    if (!queryInput || !container) return;
    
    queryInput.value = queryText;
    isExampleSelected = true;
    
    // Mantieni visibile il contenitore poiché è stato selezionato
    container.classList.remove("hidden");
    queryInput.focus();
}

// Recupera le query di esempio dal server
async function fetchExampleQueries() {
    const datasetSelect = document.getElementById("dataset-select");
    const chunkSizeSelect = document.getElementById("chunk-size-select");
    const container = document.getElementById("example-queries-container");
    const list = document.getElementById("example-queries-list");
    
    if (!datasetSelect || !chunkSizeSelect || !container || !list) return;
    
    const selectedDataset = datasetSelect.value;
    const chunkSize = chunkSizeSelect.value;
    
    if (!selectedDataset || !chunkSize) {
        container.classList.add("hidden");
        return;
    }
    
    try {
        const response = await fetch(`/api/example_queries?dataset_name=${selectedDataset}&chunk_size=${chunkSize}`);
        if (!response.ok) throw new Error("Errore nel recupero delle query d'esempio.");
        
        const data = await response.json();
        const queries = data.queries || [];
        
        if (queries.length === 0) {
            container.classList.add("hidden");
            return;
        }
        
        list.innerHTML = "";
        queries.forEach(q => {
            const btn = document.createElement("button");
            btn.className = "example-query-btn";
            btn.type = "button";
            btn.innerText = q;
            btn.onclick = () => selectExample(q);
            list.appendChild(btn);
        });
        
        // Se il campo di input è vuoto o è selezionata una query d'esempio, mostralo
        const queryInput = document.getElementById("query-input");
        if (queryInput && (queryInput.value.trim() === "" || isExampleSelected)) {
            container.classList.remove("hidden");
        } else {
            container.classList.add("hidden");
        }
    } catch (error) {
        console.error("Errore nel recuperare le query d'esempio:", error);
        container.classList.add("hidden");
    }
}

// Aggiorna il selettore delle taglie dei chunk
function updateChunkSizeSelect(datasetName) {
    if (!systemStats || !systemStats.details[datasetName]) return;
    
    const chunkSizeSelect = document.getElementById("chunk-size-select");
    const currentVal = chunkSizeSelect.value;
    chunkSizeSelect.innerHTML = "";
    
    const datasetDetails = systemStats.details[datasetName];
    let selectedSet = false;
    
    const sizes = systemStats.available_sizes || [400, 800, 1200];
    
    sizes.forEach(size => {
        const detail = datasetDetails[size];
        // Mostra solo le taglie per cui esiste effettivamente l'indice
        if (detail && detail.exists) {
            const opt = document.createElement("option");
            opt.value = size;
            
            let label = `${size} chars`;
            if (size == 800) label += " (Default)";
            else if (size == 400) label += " (Fine-grained)";
            else if (size == 1200) label += " (Coarse-grained)";
            
            opt.innerText = label;
            chunkSizeSelect.appendChild(opt);
            
            // Tenta di mantenere la selezione precedente se ancora valida
            if (size == currentVal) {
                opt.selected = true;
                selectedSet = true;
            }
        }
    });
    
    // Se la selezione precedente non è più valida, seleziona 800 se presente, altrimenti la prima opzione
    if (!selectedSet && chunkSizeSelect.options.length > 0) {
        let hasDefault = false;
        for (let i = 0; i < chunkSizeSelect.options.length; i++) {
            if (chunkSizeSelect.options[i].value == "800") {
                chunkSizeSelect.options[i].selected = true;
                hasDefault = true;
                break;
            }
        }
        if (!hasDefault) {
            chunkSizeSelect.options[0].selected = true;
        }
    }
}

// Aggiorna l'indicatore di cache RAM visibile
function updateCacheStatus(datasetName) {
    if (!systemStats || !systemStats.details[datasetName]) return;
    
    const loadedSizes = [];
    const details = systemStats.details[datasetName];
    for (const size in details) {
        if (details[size].loaded) {
            loadedSizes.push(size);
        }
    }
    
    if (loadedSizes.length === 0) {
        document.getElementById("stat-loaded").innerText = "None";
    } else {
        document.getElementById("stat-loaded").innerText = loadedSizes.join(", ") + " chars";
    }
}

// Recupera le statistiche di sistema dal server
async function fetchSystemStats() {
    try {
        const response = await fetch("/api/stats");
        if (!response.ok) throw new Error("Errore nel recupero delle statistiche.");
        
        systemStats = await response.json();
        
        // Popola la select del dataset se vuota
        const datasetSelect = document.getElementById("dataset-select");
        
        if (datasetSelect.options.length === 0) {
            datasetSelect.innerHTML = "";
            const datasets = systemStats.available_datasets || [];
            
            datasets.forEach(ds => {
                const opt = document.createElement("option");
                opt.value = ds;
                opt.innerText = formatDatasetName(ds);
                if (ds === systemStats.default_dataset) {
                    opt.selected = true;
                }
                datasetSelect.appendChild(opt);
            });
        }
        
        const activeDataset = datasetSelect.value || systemStats.default_dataset;
        
        // Popola il pannello statistiche superiore ed inizializza la taglia
        document.getElementById("stat-embedder").innerText = systemStats.default_embedder.toUpperCase();
        
        onDatasetChange();
        
    } catch (error) {
        console.error("Errore statistiche:", error);
        document.getElementById("stat-embedder").innerText = "Error";
        document.getElementById("stat-dataset").innerText = "N/A";
        document.getElementById("stat-loaded").innerText = "-";
    }
}

// Controlla lo stato del database (se i file pkl esistono o meno su disco)
function checkDatabaseStatus() {
    if (!systemStats) return;
    
    const selectedDataset = document.getElementById("dataset-select").value;
    const selectedSize = document.getElementById("chunk-size-select").value;
    const badge = document.getElementById("db-status-badge");
    
    if (!selectedSize) {
        badge.innerText = "No DB found";
        badge.className = "db-status-badge missing";
        return;
    }
    
    const datasetDetails = systemStats.details[selectedDataset];
    if (datasetDetails) {
        const detail = datasetDetails[selectedSize];
        if (detail) {
            if (detail.exists) {
                badge.innerText = detail.loaded ? "In Cache" : "Available";
                badge.className = "db-status-badge ready";
            } else {
                badge.innerText = "Unavailable";
                badge.className = "db-status-badge missing";
            }
            return;
        }
    }
    
    badge.innerText = "Unknown";
    badge.className = "db-status-badge";
}

// Formatta il nome del dataset per la visualizzazione
function formatDatasetName(name) {
    if (!name) return "-";
    return name.replace(/_/g, " ").replace(/-/g, " ").toUpperCase();
}

// Esegue la ricerca tramite API
async function executeSearch() {
    const queryInput = document.getElementById("query-input");
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Disattiva stato esempio e nascondi contenitore suggerimenti
    isExampleSelected = false;
    const exampleContainer = document.getElementById("example-queries-container");
    if (exampleContainer) {
        exampleContainer.classList.add("hidden");
    }
    
    const selectedDataset = document.getElementById("dataset-select").value;
    const chunkSize = document.getElementById("chunk-size-select").value;
    const strategy = document.getElementById("strategy-select").value;
    const alpha = document.getElementById("alpha-slider").value;
    const k = document.getElementById("k-select").value;
    const useRerank = document.getElementById("use-rerank-checkbox").checked;
    
    // Riferimenti agli elementi UI
    const loader = document.getElementById("loader");
    const loaderSub = document.getElementById("loader-sub");
    const errorCard = document.getElementById("error-card");
    const errorText = document.getElementById("error-message-text");
    const resultsContainer = document.getElementById("results-container");
    const resultsCount = document.getElementById("results-count");
    
    if (!chunkSize) {
        alert("Nessun database disponibile per questo dataset. Seleziona un altro dataset o indicizza i documenti.");
        return;
    }
    
    // Imposta lo stato di caricamento nell'interfaccia
    setFormDisabled(true);
    loader.classList.remove("hidden");
    errorCard.classList.add("hidden");
    resultsContainer.innerHTML = "";
    resultsCount.innerText = "Searching...";
    
    // Cambia il sottotesto del loader in base alla strategia
    let loaderText = "";
    if (strategy === "vector-only") {
        loaderText = "Calculating query embedding and querying ChromaDB...";
    } else if (strategy === "bm25-only") {
        loaderText = "Running term-frequency analysis with BM25 model...";
    } else {
        loaderText = "Running sparse + dense search and fusing ranks via RRF...";
    }
    
    if (useRerank) {
        loaderText += " (Then calling Jina Reranker v3 API...)";
    }
    loaderSub.innerText = loaderText;
    
    try {
        const response = await fetch("/api/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: query,
                dataset_name: selectedDataset,
                chunk_size: chunkSize,
                strategy: strategy,
                alpha: parseFloat(alpha),
                k: parseInt(k),
                use_rerank: useRerank
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Search execution failed.");
        }
        
        // Renderizza i risultati
        renderResults(data);
        
        // Aggiorna le statistiche in background (dato che un nuovo database potrebbe essere stato caricato)
        fetchSystemStats();
        
    } catch (error) {
        console.error("Errore durante la query:", error);
        errorText.innerText = error.message;
        errorCard.classList.remove("hidden");
        resultsCount.innerText = "Search error";
        
        // Ripristina lo stato vuoto se fallisce
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <p>Unable to retrieve results. Please verify that the index files for dataset "${formatDatasetName(selectedDataset)}" and chunk size ${chunkSize} exist in the data/indices folder.</p>
            </div>
        `;
    } finally {
        setFormDisabled(false);
        loader.classList.add("hidden");
    }
}

// Disabilita o abilita il form durante il caricamento
function setFormDisabled(disabled) {
    document.getElementById("query-input").disabled = disabled;
    document.getElementById("dataset-select").disabled = disabled;
    document.getElementById("chunk-size-select").disabled = disabled;
    document.getElementById("strategy-select").disabled = disabled;
    document.getElementById("alpha-slider").disabled = disabled;
    document.getElementById("k-select").disabled = disabled;
    document.getElementById("use-rerank-checkbox").disabled = disabled;
    document.getElementById("search-btn").disabled = disabled;
}

// Renderizza la lista dei risultati nella UI
function renderResults(data) {
    const container = document.getElementById("results-container");
    const countLabel = document.getElementById("results-count");
    
    const results = data.results || [];
    const isReranked = data.use_rerank || false;
    const isMocked = data.is_mocked || false;
    
    let statsText = `${results.length} results found in ${data.elapsed_ms} ms`;
    if (isMocked) {
        statsText += " (Ground Truth Match)";
    } else if (isReranked) {
        statsText += " (with Jina Rerank v3)";
    }
    countLabel.innerText = statsText;
    
    if (results.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <p>No results matched your query. Try adjusting your search term or strategy.</p>
            </div>
        `;
        return;
    }
    
    // Genera ciascun risultato card
    results.forEach((res, index) => {
        const card = document.createElement("div");
        card.className = "result-card";
        
        // Formatta il titolo (header path)
        let headerPath = res.metadata.header_path || "";
        if (!headerPath) {
            // Cerca chiavi piatte come headers_Header 1
            const h1 = res.metadata["headers_Header 1"] || "";
            const h2 = res.metadata["headers_Header 2"] || "";
            const h3 = res.metadata["headers_Header 3"] || "";
            
            const headers = [h1, h2, h3].filter(h => h.trim() !== "");
            headerPath = headers.length > 0 ? headers.join(" > ") : "Document";
        }
        
        // Nome del file sorgente
        const sourceFile = res.metadata.source_file || "N/A";
        const chunkId = res.id || "N/A";
        
        // Badges aggiuntivi
        let rankBadgesHtml = "";
        if (res.source_details && res.source_details.is_mocked) {
            rankBadgesHtml += `<span class="badge badge-groundtruth">Ground Truth Match</span> `;
        }
        if (isReranked) {
            rankBadgesHtml += `<span class="badge badge-reranked">Reranked</span> `;
            if (res.source_details && res.source_details.base_score !== undefined && res.source_details.base_score !== null) {
                rankBadgesHtml += `<span class="badge badge-base-score">Base Score: ${res.source_details.base_score.toFixed(4)}</span> `;
            }
        }
        if (res.source_details) {
            const vRank = res.source_details.vector_rank;
            const bRank = res.source_details.bm25_rank;
            
            if (vRank !== undefined && vRank !== null) rankBadgesHtml += `<span class="badge badge-v-rank">Vector Rank #${vRank}</span> `;
            if (bRank !== undefined && bRank !== null) rankBadgesHtml += `<span class="badge badge-b-rank">BM25 Rank #${bRank}</span> `;
        }
        
        const scoreLabel = isReranked ? "Relevance Score" : "Score";
        
        card.innerHTML = `
            <div class="result-top-row">
                <div class="result-index-title">
                    <span class="result-number">${index + 1}</span>
                    <h3 class="result-path">${headerPath}</h3>
                </div>
                <span class="result-score">${scoreLabel}: ${res.score.toFixed(5)}</span>
            </div>
            <div class="result-meta-row">
                <span class="badge badge-file">File: ${sourceFile}</span>
                <span class="badge">ID: ${chunkId}</span>
                ${rankBadgesHtml}
            </div>
            <div class="result-text">${escapeHtml(res.text)}</div>
        `;
        
        container.appendChild(card);
    });
}

// Utility per evitare XSS
function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
