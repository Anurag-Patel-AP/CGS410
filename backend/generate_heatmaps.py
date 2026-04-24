"""
Attention Heatmap Generator.
============================

Extracts the full 144-dimensional attention vector for (Language, Relation) pairs
across a significant sample of sentences. It averages these values into 12x12 matrices
to visually demonstrate the attention distribution across layers and heads.

Uses specialist_data_{lang}.json to filter out rare relations (we only compute 
heatmaps for relations with a valid specialist).
"""

import json
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.model_loader import model_manager
from backend.nlp_utils import matrix_and_pool, MLState

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BACKEND_DIR = os.path.dirname(__file__)
LANGS = ["en", "hi", "ja", "de"]

# To constrain execution time to < 1 hour, we sample 2500 sentences max per language.
MAX_SENTENCES = 2500

def run():
    print("Loading BERT...")
    model_manager.load_model("bert-base-multilingual-cased")
    tokenizer = model_manager.get_tokenizer()
    model = model_manager.get_model()
    model.to(MLState.device)

    all_heatmaps = {}

    for lang in LANGS:
        print(f"\nProcessing {lang.upper()}...")
        spec_path = os.path.join(BACKEND_DIR, f"specialist_data_{lang}.json")
        data_path = os.path.join(DATA_DIR, f"{lang}.json")

        if not os.path.exists(spec_path) or not os.path.exists(data_path):
            print(f"  Missing data for {lang}, skipping.")
            continue

        with open(spec_path, encoding="utf-8") as f:
            specialist_data = json.load(f)
        
        with open(data_path, encoding="utf-8") as f:
            sentences = json.load(f)

        # Build allowed relations map: {rel: {"layer": L, "head": H}}
        # This naturally filters out rare relations since they were dropped during profiling.
        allowed_relations = {
            item["rel"]: {"layer": item["layer"], "head": item["head"]}
            for item in specialist_data
        }

        # Keep running total arrays for the relations: shape (144,)
        rel_sums  = {rel: np.zeros(144) for rel in allowed_relations}
        rel_counts = {rel: 0 for rel in allowed_relations}

        sents_to_process = sentences[:MAX_SENTENCES]
        total_sents = len(sents_to_process)

        for i, s in enumerate(sents_to_process):
            if i % 250 == 0:
                print(f"  ({lang}) sentence {i}/{total_sents}...")

            tokens = s["tokens"]
            raw_heads = s["head"]
            deprels = s["deprel"]

            try:
                pooled = matrix_and_pool(tokens, tokenizer, model)  # (144, seq_len, seq_len)
            except Exception:
                continue

            N = pooled.shape[1]
            for j in range(N):
                if j >= len(raw_heads): break
                h_raw = raw_heads[j]
                if h_raw is None or str(h_raw).lower() == 'none': continue
                gold_h = int(h_raw) - 1
                if gold_h < 0 or gold_h >= N: continue
                
                # Base relation grouping (e.g. nsubj:pass -> nsubj) to match specialist_data
                raw_rel = str(deprels[j])
                base_rel = raw_rel.split(":")[0]

                if base_rel in allowed_relations:
                    # pooled[:, j, gold_h] is the 144-dimensional vector of attention 
                    # from child (j) to parent (gold_h)
                    attn_vec = pooled[:, j, gold_h].clone().numpy()
                    rel_sums[base_rel] += attn_vec
                    rel_counts[base_rel] += 1
        
        lang_output = {}
        for rel in allowed_relations:
            if rel_counts[rel] > 0:
                avg_vec = rel_sums[rel] / rel_counts[rel]
                # Reshape to 12x12
                matrix_12x12 = avg_vec.reshape((12, 12)).tolist()
                
                lang_output[rel] = {
                    "count": rel_counts[rel],
                    "specialist": allowed_relations[rel],
                    "heatmap": matrix_12x12
                }
        
        all_heatmaps[lang] = lang_output
        print(f"  {lang.upper()} complete. Mapped {len(lang_output)} relations.")

    out_path = os.path.join(DATA_DIR, "relation_heatmaps.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_heatmaps, f, indent=2)
    print(f"\nAll heatmaps computed and saved to {out_path}.")

if __name__ == "__main__":
    run()
