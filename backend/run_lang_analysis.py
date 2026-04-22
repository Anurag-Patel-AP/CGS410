"""
Specialist Head Profiling Engine.
=================================

Automated discovery pipeline script. Evaluates every single attention head across 
all layers on the 70% UD training splits. Computes classification accuracy against
the gold-standard Universal Dependencies attachments to identify the mathematically 
optimal "Specialist Head" for every core relationship per language.
Outputs profiling data to unique JSONs.
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict

# Setup paths so it can be run from either root or backend/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.model_loader import model_manager
from backend.nlp_utils import matrix_and_pool, MLState

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BACKEND_DIR = os.path.dirname(__file__)

LANGS = ["en", "hi", "ja", "de"]

# ── Adaptive min support fraction ─────────────────────────────────────────────
MIN_SUPPORT_FRAC = 0.005  # relation must cover >= 0.5% of total word tokens

def discover_and_save(lang, train_sents, tokenizer, model):
    relation_correct = defaultdict(lambda: np.zeros(144))
    relation_total   = defaultdict(int)
    total_tokens     = 0

    print(f"\n  [{lang.upper()}] Processing {len(train_sents)} sentences...")
    for s in train_sents:
        tokens    = s["tokens"]
        raw_heads = s["head"]
        deprels   = s["deprel"]
        try:
            pooled = matrix_and_pool(tokens, tokenizer, model)
        except Exception:
            continue

        N = pooled.shape[1]
        total_tokens += N
        for i in range(N):
            if i >= len(raw_heads): break
            h_raw = raw_heads[i]
            if h_raw is None or str(h_raw).lower() == 'none': continue
            gold_h = int(h_raw) - 1
            if gold_h < 0 or gold_h >= N: continue
            rel = str(deprels[i])
            relation_total[rel] += 1
            row = pooled[:, i, :].clone()
            row[:, i] = -1
            preds = row.argmax(dim=1)
            relation_correct[rel] += (preds == gold_h).numpy()

    # Adaptive support threshold
    min_support = max(30, int(total_tokens * MIN_SUPPORT_FRAC))
    print(f"  Total word tokens: {total_tokens:,}  |  Min support threshold: {min_support} (0.5% of total)")

    raw = []
    dropped = []
    for rel in sorted(relation_total.keys()):
        total = relation_total[rel]
        if total < min_support:
            dropped.append((rel, total))
            continue
        acc = relation_correct[rel] / total
        best_idx = int(acc.argmax())
        raw.append({
            "rel":      rel,
            "layer":    best_idx // 12,
            "head":     best_idx % 12,
            "accuracy": round(float(acc[best_idx]) * 100, 1),
            "support":  total
        })

    # Group subcategories (e.g. nsubj:pass + nsubj → nsubj)
    groups = {}
    for d in raw:
        base = d["rel"].split(":")[0]
        if base not in groups:
            groups[base] = {"rel": base, "totalSupport": 0, "weightedAcc": 0.0,
                            "bestAcc": -1, "bestLayer": None, "bestHead": None}
        g = groups[base]
        g["weightedAcc"]  += d["accuracy"] * d["support"]
        g["totalSupport"] += d["support"]
        if d["accuracy"] > g["bestAcc"]:
            g["bestAcc"]   = d["accuracy"]
            g["bestLayer"] = d["layer"]
            g["bestHead"]  = d["head"]

    filtered = sorted([
        {"rel": g["rel"], "layer": g["bestLayer"], "head": g["bestHead"],
         "accuracy": round(g["weightedAcc"] / g["totalSupport"], 1),
         "support":  g["totalSupport"]}
        for g in groups.values()
    ], key=lambda x: x["accuracy"], reverse=True)

    out_path = os.path.join(BACKEND_DIR, f"specialist_data_{lang}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    print(f"  Kept {len(filtered)} relations (dropped {len(dropped)}: {[r for r, _ in dropped]})")
    for d in filtered:
        print(f"    {d['rel']:<15} {d['accuracy']:5.1f}%  support={d['support']:5d}  L{d['layer']}H{d['head']}")

    return filtered

def main():
    print("Loading BERT...")
    model_manager.load_model("bert-base-multilingual-cased")
    tokenizer = model_manager.get_tokenizer()
    model     = model_manager.get_model()
    model.to(MLState.device)

    all_specialists = {}
    for lang in LANGS:
        data_path = os.path.join(DATA_DIR, f"{lang}.json")
        if not os.path.exists(data_path):
            print(f"Skipping {lang}, file not found: {data_path}")
            continue
        with open(data_path, encoding="utf-8") as f:
            train_sents = json.load(f)
        all_specialists[lang] = discover_and_save(lang, train_sents, tokenizer, model)

    # Combined global specialist file
    all_combined = {}
    for lang, entries in all_specialists.items():
        for e in entries:
            base = e["rel"]
            if base not in all_combined:
                all_combined[base] = {"rel": base, "totalSupport": 0, "weightedAcc": 0.0,
                                       "bestAcc": -1, "bestLayer": None, "bestHead": None}
            g = all_combined[base]
            g["weightedAcc"]  += e["accuracy"] * e["support"]
            g["totalSupport"] += e["support"]
            if e["accuracy"] > g["bestAcc"]:
                g["bestAcc"]   = e["accuracy"]
                g["bestLayer"] = e["layer"]
                g["bestHead"]  = e["head"]

    combined_list = sorted([
        {"rel": g["rel"], "layer": g["bestLayer"], "head": g["bestHead"],
         "accuracy": round(g["weightedAcc"] / g["totalSupport"], 1), "support": g["totalSupport"]}
        for g in all_combined.values()
    ], key=lambda x: x["accuracy"], reverse=True)

    global_out = os.path.join(BACKEND_DIR, "specialist_data.json")
    with open(global_out, "w", encoding="utf-8") as f:
        json.dump(combined_list, f, indent=2)
    print(f"\nCombined global: {len(combined_list)} relations saved to {global_out}.")

if __name__ == "__main__":
    main()
