"""
Main FastAPI Application Engine.
================================

This module acts as the core backend router for the BERT Dependency Analyzer.
It spins up the Uvicorn webserver, loads the pre-computed specialist ML state, 
initializes the BERT models via the lifespan generator, and serves both the 
REST API endpoints as well as the static frontend HTML.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import os
import json

from backend.model_loader import model_manager
from backend.data_loader import data_manager
from backend.nlp_utils import load_lang_specialists, discover_specialists, parse_sentence, MLState

DATA_DIR = os.path.join(os.path.dirname(__file__), "")

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load_model("bert-base-multilingual-cased")
    data_manager.load_all()

    # Load pre-computed per-language specialist head JSONs
    data_dir = os.path.dirname(__file__)
    load_lang_specialists(data_dir)

    # If global specialist data missing (no JSON), fall back to runtime discovery
    if not MLState.best_heads:
        discover_specialists(data_manager, model_manager)
    yield

app = FastAPI(title="BERT UD Dependency Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    language: str
    sentence_id: int

@app.get("/api/languages")
def get_languages():
    return [
        {"code": "en", "name": "English"},
        {"code": "hi", "name": "Hindi"},
        {"code": "ja", "name": "Japanese"},
        {"code": "de", "name": "German"}
    ]

@app.get("/api/sentences/{lang}")
def get_sentences(lang: str):
    sentences = data_manager.get_test_sentences(lang)
    if not sentences:
        raise HTTPException(status_code=404, detail="Language not found or no held-out sentences available")
    return [{"id": s["id"], "text": " ".join(s["tokens"])} for s in sentences]

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    # Use held-out test sentence (NOT seen during specialist discovery)
    sentence_data = data_manager.get_test_sentence(req.language, req.sentence_id)
    if not sentence_data:
        raise HTTPException(status_code=404, detail="Sentence not found")

    tokens  = sentence_data["tokens"]
    heads   = sentence_data["head"]
    deprels = sentence_data["deprel"]

    tokenizer = model_manager.get_tokenizer()
    model     = model_manager.get_model()

    # Parse using language-specific specialist heads
    gold_tree, pred_tree, uas = parse_sentence(
        tokens, tokenizer, model, heads, lang=req.language
    )

    return {
        "tokens":    tokens,
        "gold_tree": gold_tree,
        "pred_tree": pred_tree,
        "uas":       uas,
        "deprels":   deprels
    }

@app.get("/api/specialists")
def get_specialists():
    path = os.path.join(os.path.dirname(__file__), "specialist_data.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Specialist data not computed yet.")
    with open(path, "r") as f:
        return json.load(f)

@app.get("/api/evaluation")
def get_evaluation():
    path = os.path.join(os.path.dirname(__file__), "test_results.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Evaluation results not computed yet. Run run_evaluation.py first.")
    with open(path) as f:
        return json.load(f)

@app.get("/api/lmm_evaluation")
def get_lmm_evaluation():
    path = os.path.join(os.path.dirname(__file__), "lmm_results.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="LMM results not computed yet. Run run_lmm_analysis.py first.")
    with open(path) as f:
        return json.load(f)

@app.get("/api/specialists/{lang}")
def get_specialists_lang(lang: str):
    valid = {"en", "hi", "ja", "de", "all"}
    if lang not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown language '{lang}'.")
    filename = "specialist_data.json" if lang == "all" else f"specialist_data_{lang}.json"
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No specialist data for '{lang}'.")
    with open(path) as f:
        return json.load(f)

@app.get("/api/heatmaps")
def get_heatmaps():
    path = os.path.join(os.path.dirname(__file__), "data", "relation_heatmaps.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Heatmaps not computed yet. Run generate_heatmaps.py first.")
    with open(path) as f:
        return json.load(f)

# Serve backend graphs (must be mounted BEFORE the root "/" catch-all)
graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")
if os.path.exists(graphs_dir):
    app.mount("/graphs", StaticFiles(directory=graphs_dir), name="graphs")

# Serve frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
