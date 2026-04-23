---
title: BERT Dependency Parser
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Do LLMs Develop Dependency Grammar?
**CGS410 Project — Kritin Khowala & Anurag Patel, IIT Kanpur**

> **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/Anurag-patel/CGS410)

---

## Research Question

Transformer language models are trained to predict the next token, yet they consistently produce grammatically well-formed sentences. Prior work (Voita et al., 2019) showed that certain attention heads appear to specialize in capturing syntactic relations between words. This project asks:

1. **Do specialized syntactic attention heads emerge consistently across different languages?**
2. **Is the model more accurate at identifying core grammatical relations (e.g., *nsubj*, *obj*) than peripheral modifiers?**

### Hypothesis
We hypothesize that BERT's self-attention weights implicitly encode dependency relations. Specifically:
- Core arguments like `nsubj` and `obj` will achieve higher Unlabeled Attachment Scores (UAS) than adjuncts like `advcl` or `nmod`.
- Language typology (e.g., head-final Japanese/Hindi vs head-initial English) will affect the layer/head at which syntactic structure emerges.

---

## Methodology

### Attention Extraction
For a sentence of length *n*, we extract attention weights produced by each attention head *h* at layer *l*:

$$\alpha_{i,j}^{(l,h)} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)$$

These weights represent how strongly token *i* attends to token *j*. Sub-word tokens (WordPiece) are collapsed back to whole words by averaging.

### Specialist Head Discovery (Training Split)
Every one of the **144 attention heads** (12 layers × 12 heads) in `bert-base-multilingual-cased` is evaluated individually on the 70% training split of the [Universal Dependencies](https://universaldependencies.org/) corpus. For each UD relation (e.g. `nsubj`, `amod`, `case`), the head whose attention weights most accurately predict the correct dependency attachment is designated the **Specialist Head**.

### Dependency Prediction
For each token *i*, the predicted parent is the token receiving the highest attention weight at the specialist head:

$$\text{Parent}(i) = \arg\max_j\ \alpha_{i,j}^{(l,h)}$$

### Tree Extraction — Chu-Liu/Edmonds (MST)
The attention matrix is treated as a fully-connected directed graph (edge weight = attention weight). The [Chu-Liu/Edmonds algorithm](https://en.wikipedia.org/wiki/Edmonds%27_algorithm) extracts the globally optimal Maximum Spanning Arborescence — guaranteeing a valid dependency tree with no cycles.

### Evaluation
Performance is measured using the **Unlabeled Attachment Score (UAS)** on the held-out 30% test split (never seen during specialist discovery):

$$\text{UAS} = \frac{\text{correct parent predictions}}{\text{total tokens}}$$

### Statistical Validation — Linear Mixed Model
To isolate the effect of syntactic relation type while controlling for confounding variables (sentence length, language variation), we estimate a **Linear Mixed Model**:

$$y \sim \beta_0 + \beta_1 T + \beta_2 L + (1|\text{Language}) + \varepsilon$$

where:
- **y** = attention weight of the word pair
- **T** = gold dependency type (fixed effect)
- **L** = sentence length (fixed confounder)
- **(1|Language)** = per-language random intercept (controls cross-lingual baseline differences)
- **ε** = residual error

The model's fixed-effect coefficients (β₁) reveal which syntactic relations are genuinely encoded in BERT's attention beyond what sentence length or language baseline predict alone.

---

## Repository Structure

```
Website/
├── backend/
│   ├── data/                      # UD corpus splits (train/test JSON + Word.csv)
│   ├── graphs/                    # Auto-generated research plots (.png)
│   ├── main.py                    # FastAPI server — routes, API endpoints, static serving
│   ├── model_loader.py            # Singleton BERT model/tokenizer loader
│   ├── data_loader.py             # UD JSON corpus loader (train/test splits)
│   ├── nlp_utils.py               # Core NLP — attention pooling, MST, UAS, specialist logic
│   ├── extract_ud_data.py         # Step 1: Download & split UD corpus (70/30)
│   ├── run_lang_analysis.py       # Step 2: Discover specialist heads per language
│   ├── run_evaluation.py          # Step 3: Benchmark on held-out test set
│   ├── run_lmm_analysis.py        # Step 4: Fit LMM, export coefficients & graphs
│   ├── generate_graphs.py         # Optional: Regenerate UAS graphs from test_results.json
│   ├── specialist_data*.json      # Pre-computed specialist head assignments
│   ├── test_results.json          # Pre-computed evaluation results
│   ├── lmm_results.json           # Pre-computed LMM coefficients & random effects
│   └── requirements.txt
├── frontend/
│   ├── index.html                 # Single-page research app
│   ├── css/style.css              # Glassmorphic design system
│   └── js/
│       ├── app_v7.js              # Analyzer UI — sentence selection, API calls, tree render
│       ├── tree_v7.js             # D3.js dependency tree renderer
│       ├── specialist_table.js    # Specialist head accuracy table renderer
│       ├── unique_heads.js        # Unique/common specialist heads across languages
│       ├── evaluation_charts.js   # UAS charts (avg by language, UAS vs length)
│       └── lmm_charts.js          # LMM forest plot + random effects chart
├── Dockerfile                     # HuggingFace Spaces deployment container
└── README.md
```

---

## Prerequisites

- Python 3.10+
- `pip` package manager
- ~4 GB RAM (for BERT model loading)

---

## How to Run Locally (Full Pipeline)

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Extract & Split UD Corpus
Downloads `universal_dependencies` from HuggingFace and splits 70% train / 30% test:
```bash
python backend/extract_ud_data.py
```
*Populates `backend/data/` with `en.json`, `hi.json`, `ja.json`, `de.json` (train) and `*_test.json` (held-out).*

### 3. Discover Specialist Heads
Evaluates all 144 BERT attention heads on the training split to find the best head per relation per language:
```bash
python backend/run_lang_analysis.py
```
*Outputs `backend/specialist_data_*.json` files.*

### 4. Run Evaluation
Benchmarks specialist heads on the 30% held-out test split:
```bash
python backend/run_evaluation.py
```
*Outputs `backend/test_results.json` with per-sentence UAS scores.*

### 5. Run LMM Statistical Analysis
Fits the Linear Mixed Model from the proposal and generates all research graphs:
```bash
python backend/run_lmm_analysis.py
```
This single command:
- Fits **Model 1**: `y ~ C(T) + L + (1|Language)` — the proposal formula
- Fits **Model 2**: `y ~ C(T) + L + head + layer + (1|Language)` — extended model
- Extracts fixed-effect coefficients (β₁T) and per-language random intercepts
- Saves `backend/lmm_results.json` (feeds the live web UI)
- Generates **two static graphs** in `backend/graphs/`:

| File | Description |
|---|---|
| `lmm_forest_plot.png` | Fixed-effect coefficients β₁ per dependency type (dots + 95% CI bars) |
| `lmm_random_effects.png` | Per-language (1\|Language) random intercept deviations |

> **Optional:** Regenerate only the UAS graphs (no BERT needed):
> ```bash
> python backend/generate_graphs.py
> ```

### 6. Start the Web Server
```bash
python -m uvicorn backend.main:app --reload
```
Open **http://localhost:8000** — the full interactive research dashboard.

---

## Key Results

| Language | Avg UAS (Test Split) | n |
|---|---|---|
| English | ~47% | 20,857 tokens |
| Hindi | ~30% | 16,521 tokens |
| Japanese | ~35% | 22,872 tokens |
| German | ~42% | 11,118 tokens |

The LMM (Model 1) shows that grammatical core arguments (`nsubj`, `obj`) have consistently positive fixed-effect coefficients, confirming the hypothesis that BERT's attention is non-randomly aligned with dependency structure — even after controlling for sentence length and language baseline variation.

---

## Tech Stack

| Layer | Technology |
|---|---|
| BERT | `bert-base-multilingual-cased` (HuggingFace Transformers) |
| UD Corpus | `universal_dependencies` (HuggingFace Datasets) |
| MST Algorithm | Chu-Liu/Edmonds (custom implementation in `nlp_utils.py`) |
| Statistical Model | `statsmodels` Mixed Linear Model |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla JS + D3.js v7 (glassmorphic design) |
| Deployment | Docker → HuggingFace Spaces |

---

## References

- Ryu & Lewis (2021)
- Voita et al. (2019) — *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned*
- Vaswani et al. (2017) — *Attention Is All You Need*
- Edmonds (1967) — *Optimum Branchings*
