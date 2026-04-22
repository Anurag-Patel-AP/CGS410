import torch
import numpy as np
import networkx as nx
from collections import defaultdict
import json
import os

class MLState:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Global fallback specialists (all languages combined)
    best_heads = {}
    relations = []
    # Per-language specialists: {lang: {rel: head_idx (0-143)}}
    lang_best_heads = {}

def load_lang_specialists(data_dir: str):
    """
    Load pre-computed per-language specialist JSON files at startup.
    Populates MLState.lang_best_heads and MLState.best_heads (combined/all).
    """
    langs = ["en", "hi", "ja", "de"]
    for lang in langs:
        path = os.path.join(data_dir, f"specialist_data_{lang}.json")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, lang '{lang}' will use global fallback.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        # Convert to {rel: flat_head_idx} dict
        mapping = {}
        for e in entries:
            flat_idx = e["layer"] * 12 + e["head"]
            mapping[e["rel"]] = flat_idx
        MLState.lang_best_heads[lang] = mapping
        print(f"Loaded {len(mapping)} specialist heads for lang='{lang}'")

    # Load combined/global
    combined_path = os.path.join(data_dir, "specialist_data.json")
    if os.path.exists(combined_path):
        with open(combined_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        MLState.best_heads = {e["rel"]: e["layer"] * 12 + e["head"] for e in entries}
        MLState.relations = list(MLState.best_heads.keys())
        print(f"Loaded {len(MLState.best_heads)} global specialist heads (combined)")

def matrix_and_pool(tokens, tokenizer, model):
    """Returns pooled attention tensor of shape (144, num_words, num_words)."""
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=512)
    word_ids = enc.word_ids(batch_index=0)

    inputs_device = {k: v.to(MLState.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**inputs_device, output_attentions=True)

    attentions = torch.cat(out.attentions, dim=1).squeeze(0).cpu()  # (144, seq, seq)

    valid_word_ids = [w for w in word_ids if w is not None]
    if not valid_word_ids:
        return torch.zeros((attentions.shape[0], 0, 0))

    num_words = max(valid_word_ids) + 1
    word_to_subwords = defaultdict(list)
    for i, w in enumerate(word_ids):
        if w is not None:
            word_to_subwords[w].append(i)

    num_heads = attentions.shape[0]
    pooled = torch.zeros((num_heads, num_words, num_words))
    for i in range(num_words):
        for j in range(num_words):
            si = word_to_subwords.get(i, [])
            sj = word_to_subwords.get(j, [])
            if not si or not sj: continue
            region = attentions[:, si, :][:, :, sj]
            pooled[:, i, j] = region.mean(dim=1).sum(dim=1)

    return pooled

def discover_specialists(data_manager, model_manager):
    """Fallback: discover global specialists at runtime (used only if JSONs missing)."""
    print("Discovering global specialists at runtime...")
    langs = ["en", "hi", "ja", "de"]
    tokenizer = model_manager.get_tokenizer()
    model = model_manager.get_model()
    model.to(MLState.device)

    relation_correct = defaultdict(lambda: np.zeros(144))
    relation_total = defaultdict(int)

    for lang in langs:
        sentences = data_manager.get_sentences(lang)
        for idx in range(min(100, len(sentences))):
            example = sentences[idx]
            tokens = example["tokens"]
            try:
                pooled = matrix_and_pool(tokens, tokenizer, model)
            except Exception:
                continue

            num_words = pooled.shape[1]
            for i in range(num_words):
                if i >= len(example["head"]): break
                h_raw = example["head"][i]
                if h_raw is None or str(h_raw).lower() == 'none': continue
                gold_h = int(h_raw) - 1
                if gold_h < 0 or gold_h >= num_words: continue
                rel = str(example["deprel"][i])
                relation_total[rel] += 1
                preds = pooled[:, i, :].argmax(dim=1)
                relation_correct[rel] += (preds == gold_h).numpy()

    MLState.best_heads = {}
    for rel, total in relation_total.items():
        if total > 0:
            best = (relation_correct[rel] / total).argmax()
            MLState.best_heads[rel] = int(best)
    MLState.relations = list(MLState.best_heads.keys())
    print(f"Found global specialists for {len(MLState.relations)} relations.")

def gold(heads):
    return {i: (int(h) - 1 if str(h).lower() != 'none' else -1) for i, h in enumerate(heads)}

def compute_uas(pred_tree, gold_tree):
    if len(gold_tree) == 1:
        return 100.0 if pred_tree.get(0, -1) == gold_tree.get(0, -1) else 0.0
    non_root = [i for i, p in gold_tree.items() if p != -1]
    if not non_root: return 0.0
    correct = sum(pred_tree.get(i, -1) == gold_tree[i] for i in non_root)
    return round((correct / len(non_root)) * 100, 2)

def extract_mst_dict(adj_matrix):
    G = nx.DiGraph()
    N = adj_matrix.shape[0]
    if N == 0: return {}
    for i in range(N):
        for j in range(N):
            if i != j:
                G.add_edge(j, i, weight=float(adj_matrix[i, j]))
    pred = {i: -1 for i in range(N)}
    try:
        mst = nx.algorithms.tree.branchings.maximum_spanning_arborescence(G)
        for u, v in mst.edges():
            pred[v] = u
    except:
        best_parents = adj_matrix.argmax(axis=1)
        for i, p in enumerate(best_parents):
            pred[i] = int(p)
    return pred

def parse_sentence(tokens, tokenizer, model, raw_heads, lang: str = "all"):
    """
    Parse using language-specific specialist heads if available,
    otherwise fall back to global combined specialists.
    """
    pooled = matrix_and_pool(tokens, tokenizer, model)
    N = pooled.shape[1]
    gold_tree = gold(raw_heads)

    if N == 0:
        return gold_tree, {}, 0.0

    # Select lang-specific specialist head indices
    specialist_map = MLState.lang_best_heads.get(lang, MLState.best_heads)
    if not specialist_map:
        specialist_map = MLState.best_heads  # ultimate fallback

    head_indices = list(set(specialist_map.values()))
    if not head_indices:
        return gold_tree, {}, 0.0

    special = pooled[head_indices]
    A = special.mean(dim=0).numpy()
    pred_tree = extract_mst_dict(A)
    uas = compute_uas(pred_tree, gold_tree)
    return gold_tree, pred_tree, uas
