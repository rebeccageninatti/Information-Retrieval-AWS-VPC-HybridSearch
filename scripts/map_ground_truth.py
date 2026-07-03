import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    DATASET_NAME,
    DEFAULT_CHUNK_STRATEGY,
    CHUNK_SIZE_DEFAULT,
    GROUND_TRUTH_DIR,
    CHUNKS_DIR
)

def clean_text(text: str) -> str:
    if text.startswith("Context:"):
        parts = text.split("\n\n", 1)
        if len(parts) > 1:
            text = parts[1]
    # Remove whitespace and lowercase for comparison
    return "".join(text.split()).lower()

def main():
    target_size = CHUNK_SIZE_DEFAULT
    print(f"Mapping ground truth from 800 to target size {target_size}...")
    
    ref_gt_path = GROUND_TRUTH_DIR / f"ground_truth_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_800.jsonl"
    ref_chunks_path = CHUNKS_DIR / f"chunks_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_800.jsonl"
    
    target_gt_path = GROUND_TRUTH_DIR / f"ground_truth_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_{target_size}.jsonl"
    target_chunks_path = CHUNKS_DIR / f"chunks_{DATASET_NAME}_{DEFAULT_CHUNK_STRATEGY}_{target_size}.jsonl"
    
    if not ref_gt_path.exists():
        print(f"Error: Reference ground truth not found at {ref_gt_path}")
        sys.exit(1)
    if not ref_chunks_path.exists():
        print(f"Error: Reference chunks not found at {ref_chunks_path}")
        sys.exit(1)
    if not target_chunks_path.exists():
        print(f"Error: Target chunks not found at {target_chunks_path}")
        sys.exit(1)

    # Load 800 chunks
    ref_chunks = {}
    with open(ref_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                ref_chunks[c["chunk_id"]] = {
                    "text": c["text"],
                    "clean": clean_text(c["text"]),
                    "source_file": c["source_file"]
                }
                
    # Load target chunks
    target_chunks_by_file = {}
    with open(target_chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                sf = c["source_file"]
                if sf not in target_chunks_by_file:
                    target_chunks_by_file[sf] = []
                target_chunks_by_file[sf].append({
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "clean": clean_text(c["text"])
                })
                
    # Load and map 800 ground truth queries
    mapped_queries = []
    unmapped_count = 0
    
    with open(ref_gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            
            new_relevant_doc_ids = []
            
            for ref_doc_id in q["relevant_doc_ids"]:
                if ref_doc_id not in ref_chunks:
                    print(f"Warning: Chunk {ref_doc_id} not found in 800 chunks")
                    continue
                ref_chunk = ref_chunks[ref_doc_id]
                source_file = ref_chunk["source_file"]
                ref_clean = ref_chunk["clean"]
                
                # Find matching target chunks
                candidates = target_chunks_by_file.get(source_file, [])
                best_candidates = []
                
                for cand in candidates:
                    cand_clean = cand["clean"]
                    # Substring check: since target size (400) is smaller than ref size (800),
                    # the target chunk is likely a substring of the reference chunk.
                    if cand_clean in ref_clean or ref_clean in cand_clean:
                        best_candidates.append(cand["chunk_id"])
                
                # If no substring match, fall back to word overlap
                if not best_candidates:
                    best_overlap = 0
                    best_cand_id = None
                    ref_words = set(ref_chunk["text"].lower().split())
                    for cand in candidates:
                        cand_words = set(cand["text"].lower().split())
                        overlap = len(ref_words & cand_words)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_cand_id = cand["chunk_id"]
                    if best_cand_id:
                        best_candidates.append(best_cand_id)
                        
                new_relevant_doc_ids.extend(best_candidates)
                
            # Deduplicate
            new_relevant_doc_ids = list(set(new_relevant_doc_ids))
            
            if not new_relevant_doc_ids:
                print(f"Warning: Query {q['query_id']} mapped to 0 documents!")
                unmapped_count += 1
                
            new_q = q.copy()
            new_q["relevant_doc_ids"] = new_relevant_doc_ids
            
            # Update source_metadata chunk_id if single chunk
            if "source_metadata" in new_q and "chunk_id" in new_q["source_metadata"]:
                if new_relevant_doc_ids:
                    new_q["source_metadata"]["chunk_id"] = new_relevant_doc_ids[0]
            
            mapped_queries.append(new_q)
            
    # Write target ground truth
    with open(target_gt_path, "w", encoding="utf-8") as f:
        for mq in mapped_queries:
            f.write(json.dumps(mq, ensure_ascii=False) + "\n")
            
    print(f"Mapped {len(mapped_queries)} queries. Unmapped: {unmapped_count}")
    print(f"Saved to {target_gt_path}")

if __name__ == "__main__":
    main()
