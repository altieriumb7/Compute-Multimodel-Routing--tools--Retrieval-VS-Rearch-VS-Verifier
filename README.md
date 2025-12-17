# Budgeted Multi-Tool Router (MVP)

This repo is a **minimal, runnable** implementation of a **budgeted router** that chooses between:
- **BM25 retrieval** (local, CPU)
- **Dense vector retrieval** via **Qdrant** (fast, free tier friendly)
- (**STOP** action is included in the action space; by default the loop is 1-step retrieval → answer/eval)

It includes:
- corpus ingestion + indexes (BM25 + Qdrant)
- trace generation (simple "teacher": try tools and pick best under budget)
- supervised router training (sklearn)
- evaluation (retrieval-only; optional LLM answer generation supported)

## Data formats

### Corpus (JSONL)
Each line:
```json
{"doc_id":"...", "title":"...", "text":"..."}
```

### QA set (JSONL)
Each line:
```json
{"qid":"...", "question":"...", "answer":"..."}
```
`answer` can be a string; for multiple answers use a list.

## Quickstart

### 0) Install deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Start Qdrant locally (docker)
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2) Build indexes
```bash
# BM25 index
python scripts/build_bm25.py --corpus data/corpus.jsonl --out artifacts/bm25

# Qdrant dense ingestion (embeddings + upsert)
python scripts/ingest_qdrant.py --corpus data/corpus.jsonl --collection wiki_mini
```

### 3) Generate traces (teacher tries both tools)
```bash
python scripts/generate_traces.py   --qa data/qa.jsonl   --bm25 artifacts/bm25   --qdrant_collection wiki_mini   --out artifacts/traces.jsonl
```

### 4) Train router
```bash
python scripts/train_router.py --traces artifacts/traces.jsonl --out artifacts/router
```

### 5) Evaluate router
```bash
python scripts/eval_router.py   --qa data/qa.jsonl   --bm25 artifacts/bm25   --qdrant_collection wiki_mini   --router artifacts/router   --budget 1.0
```

## Notes
- The current MVP uses a **retrieval-only** success signal: answer substring appears in retrieved text.
- You can switch to LLM generation by adding `--llm_model <hf_model_or_path>` in eval.
- Qdrant URL defaults to `http://localhost:6333` (set `QDRANT_URL` to override).
