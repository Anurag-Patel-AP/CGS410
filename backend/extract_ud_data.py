import json
import os
import sys
import math

# ── Split configuration ───────────────────────────────────────────────────────
# Fraction used for specialist head discovery (per language-wise profiling).
TRAIN_RATIO = 0.70          # first 70%  →  specialist discovery
# Remaining fraction is held-out for UAS evaluation (not seen during discovery).
TEST_RATIO  = 1 - TRAIN_RATIO  # last 30%

def extract_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets library: pip install datasets")
        sys.exit(1)

    # Mapping of our internal language codes to HuggingFace Universal Dependencies configs
    lang_map = {
        "en": "en_ewt",    # English
        "hi": "hi_hdtb",   # Hindi
        "ja": "ja_gsd",    # Japanese
        "de": "de_gsd"     # German
    }

    for lang, config in lang_map.items():
        print(f"\nLoading '{lang}' from HuggingFace config '{config}'...")
        try:
            # We load the entire train split of the UD corpus
            ds_all = load_dataset(
                "universal_dependencies", config,
                split="train",
                trust_remote_code=True
            )

            all_sentences = []
            for i, example in enumerate(ds_all):
                tokens = example.get("tokens", [])
                head   = example.get("head",   [])
                deprel = example.get("deprel", [])
                if not tokens or not head or not deprel:
                    continue
                try:
                    int_heads = [int(h) for h in head]
                    all_sentences.append(
                        {"id": i, "tokens": tokens, "head": int_heads, "deprel": deprel}
                    )
                except ValueError:
                    continue

            total = len(all_sentences)
            train_size = math.floor(total * TRAIN_RATIO)
            test_size = total - train_size

            print(f"  Extracted {total} valid sentences.")
            print(f"  Split: {train_size} train (70%) / {test_size} test (30%)")

            # ── 70 % split: specialist head discovery ────────────────────────
            train_sentences = all_sentences[:train_size]
            out_train = os.path.join(data_dir, f"{lang}.json")
            with open(out_train, "w", encoding="utf-8") as f:
                json.dump(train_sentences, f, ensure_ascii=False, indent=2)
            print(f"  [TRAIN] Saved to {out_train}")

            # ── 30 % split: held-out UAS evaluation ─────────────────────────
            test_sentences = all_sentences[train_size:]
            # Re-label IDs so they are globally unique and clearly in the test range
            for j, s in enumerate(test_sentences):
                s["id"] = train_size + j
            out_test = os.path.join(data_dir, f"{lang}_test.json")
            with open(out_test, "w", encoding="utf-8") as f:
                json.dump(test_sentences, f, ensure_ascii=False, indent=2)
            print(f"  [TEST]  Saved to {out_test}")

        except Exception as e:
            print(f"  Failed to load data for {lang}: {e}")

if __name__ == "__main__":
    extract_data()
