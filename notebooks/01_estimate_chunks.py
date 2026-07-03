import os

# =====================================================================
# CONFIGURATION
# =====================================================================
# Root directory containing the raw markdown files
INPUT_DIR = "data/raw/vpc"

# Chunking settings
CHUNK_SIZE_WORDS = 500
OVERLAP_PERCENT = 10  # 10% overlap
# =====================================================================

def chunk_text(text, chunk_size, overlap):
    """
    Split text into chunks of `chunk_size` words with `overlap` words.
    """
    words = text.split()
    if not words:
        return []
    
    if len(words) <= chunk_size:
        return [words]
    
    chunks = []
    step = chunk_size - overlap
    # Prevent infinite loop if step is 0 or negative
    if step <= 0:
        step = 1
        
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
            
    return chunks

def main():
    overlap_words = int(CHUNK_SIZE_WORDS * (OVERLAP_PERCENT / 100))
    print(f"Parametri di Chunking:")
    print(f"- Dimensione chunk: {CHUNK_SIZE_WORDS} parole")
    print(f"- Sovrapposizione: {OVERLAP_PERCENT}% ({overlap_words} parole)")
    print(f"- Cartella di ricerca: {INPUT_DIR}\n")
    
    total_files = 0
    total_words = 0
    total_chunks = 0
    
    file_stats = []
    
    # Cammina ricorsivamente nella cartella
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, INPUT_DIR)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    words = content.split()
                    word_count = len(words)
                    chunks = chunk_text(content, CHUNK_SIZE_WORDS, overlap_words)
                    chunk_count = len(chunks)
                    
                    file_stats.append({
                        "path": rel_path,
                        "words": word_count,
                        "chunks": chunk_count
                    })
                    
                    total_files += 1
                    total_words += word_count
                    total_chunks += chunk_count
                    
                except Exception as e:
                    print(f"Errore nella lettura di {file_path}: {e}")
                    
    # Stampa i risultati in formato tabella leggibile
    print(f"{'File Path':<80} | {'Parole':<8} | {'Chunk':<6}")
    print("-" * 100)
    for stat in sorted(file_stats, key=lambda x: x["path"]):
        path_str = stat["path"]
        if len(path_str) > 77:
            path_str = "..." + path_str[-74:]
        print(f"{path_str:<80} | {stat['words']:<8} | {stat['chunks']:<6}")
        
    print("-" * 100)
    print("RIASSUNTO GENERALE:")
    print(f"- File elaborati: {total_files}")
    print(f"- Parole totali: {total_words}")
    print(f"- Chunk totali stimati: {total_chunks}")
    if total_files > 0:
        print(f"- Media parole per file: {total_words / total_files:.1f}")
        print(f"- Media chunk per file: {total_chunks / total_files:.2f}")
    else:
        print("- Nessun file markdown trovato.")

if __name__ == "__main__":
    main()
