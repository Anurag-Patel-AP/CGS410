"""
Zero-Shot Model Evaluation Suite.
=================================

Automated evaluation pipeline script. Benchmarks the derived specialist attention 
heads against the held-out 30% Unlabeled Dependency testing dataset splits.
It reconstructs Maximum Spanning Arborescences on completely unseen sentences and 
generates the 'test_results.json' file driving the frontend dashboard charts.
"""
import json
import os
import sys
import csv

# Setup paths so it can be run from either root or backend/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.model_loader import model_manager
from backend.nlp_utils import matrix_and_pool, MLState, gold, extract_mst_dict, compute_uas

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BACKEND_DIR = os.path.dirname(__file__)

LANGS = ["en", "hi", "ja", "de"]
LANG_LABELS = {"en": "English", "hi": "Hindi", "ja": "Japanese", "de": "German"}

def main():
    print("Loading BERT...")
    model_manager.load_model("bert-base-multilingual-cased")
    tokenizer = model_manager.get_tokenizer()
    model     = model_manager.get_model()
    model.to(MLState.device)

    results = []

    for lang in LANGS:
        test_path = os.path.join(DATA_DIR, f"{lang}_test.json")
        if not os.path.exists(test_path):
            print(f"Skipping {lang}, file not found: {test_path}")
            continue
        with open(test_path, encoding="utf-8") as f:
            test_sents = json.load(f)

        spec_path = os.path.join(BACKEND_DIR, f"specialist_data_{lang}.json")
        if not os.path.exists(spec_path):
            print(f"Skipping {lang}, specialist data not found: {spec_path}")
            continue
        with open(spec_path, encoding="utf-8") as f:
            lang_specs = json.load(f)

        specialist_map = {e["rel"]: e["layer"] * 12 + e["head"] for e in lang_specs}
        head_indices   = list(set(specialist_map.values()))
        print(f"\n  [{lang.upper()}] Evaluating {len(test_sents)} test sentences (using {len(head_indices)} unique heads)...")

        for s in test_sents:
            tokens    = s["tokens"]
            raw_heads = s["head"]
            length    = len(tokens)
            try:
                pooled    = matrix_and_pool(tokens, tokenizer, model)
                N         = pooled.shape[1]
                gold_tree = gold(raw_heads)
                if N == 0 or not head_indices:
                    uas = 0.0
                else:
                    special = pooled[head_indices]
                    A       = special.mean(dim=0).numpy()
                    pred    = extract_mst_dict(A)
                    uas     = compute_uas(pred, gold_tree)
            except Exception:
                uas = None

            results.append({
                "language":      lang,
                "language_name": LANG_LABELS[lang],
                "sentence_id":   s["id"],
                "length":        length,
                "uas":           uas
            })

    out_file = os.path.join(BACKEND_DIR, "test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nHeld-out UAS results:")
    for lang in LANGS:
        vals = [r["uas"] for r in results if r["language"] == lang and r["uas"] is not None]
        avg  = sum(vals)/len(vals) if vals else 0
        print(f"  {LANG_LABELS[lang]:<12}: {avg:.2f}%  (n={len(vals)})")

    # Save specialist CSV
    print("\nSaving specialist_dataframe.csv")
    all_specs = {}
    for lang in LANGS:
        p = os.path.join(BACKEND_DIR, f"specialist_data_{lang}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                all_specs[lang] = json.load(f)

    all_rels = sorted(set(e["rel"] for lang_entries in all_specs.values() for e in lang_entries))
    csv_path = os.path.join(BACKEND_DIR, "specialist_dataframe.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["relation"]
        for lang in LANGS:
            header += [f"{lang}_layer", f"{lang}_head", f"{lang}_accuracy_%", f"{lang}_support"]
        w.writerow(header)
        for rel in all_rels:
            row = [rel]
            for lang in LANGS:
                entry = next((e for e in all_specs.get(lang, []) if e["rel"] == rel), None)
                if entry:
                    row += [entry["layer"], entry["head"], entry["accuracy"], entry["support"]]
                else:
                    row += ["", "", "", ""]
            w.writerow(row)

    print(f"Saved {len(all_rels)} relations to specialist_dataframe.csv")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
