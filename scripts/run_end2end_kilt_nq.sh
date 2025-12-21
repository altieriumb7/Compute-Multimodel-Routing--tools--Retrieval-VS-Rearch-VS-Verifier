#!/usr/bin/env bash
set -euo pipefail

# --- repo root ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs data artifacts results

LOG="logs/run_end2end_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "== [run] START =="; date
echo "[run] root=$ROOT"
echo "[run] log=$LOG"

# --- load .env if present (recommended) ---
# .env example:
# QDRANT_URL="https://....:6333"
# QDRANT_API_KEY="...."
if [[ -f ".env" ]]; then
  echo "[run] loading .env"
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# ---- defaults (override via env if you want) ----
TASK="${TASK:-nq}"
SPLIT="${SPLIT:-validation}"
MAX_EXAMPLES="${MAX_EXAMPLES:-5000}"

WIKI_CONFIG="${WIKI_CONFIG:-2019-08-01}"
NEG_PAGES="${NEG_PAGES:-5000}"
MAX_PAGES="${MAX_PAGES:-20000}"
MAX_PASSAGES="${MAX_PASSAGES:-200000}"

COLLECTION="${COLLECTION:-wiki_mini}"
DENSE_MODEL="${DENSE_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUDGET="${BUDGET:-1.0}"

# device auto-detect (cuda if available)
DEVICE="${DEVICE:-auto}"
if [[ "$DEVICE" == "auto" ]]; then
  if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then DEVICE="cuda"; else DEVICE="cpu"; fi
fi
echo "[run] device=$DEVICE"

# --- install deps (don’t force-install torch; let your image provide CUDA torch) ---
echo "== [deps] install =="; date
python -m pip install -U pip
python -m pip install "datasets<4.0.0"
python -m pip install qdrant-client sentence-transformers transformers rank-bm25 scikit-learn tqdm orjson accelerate
python -m pip install -e .

echo "== [deps] GPU check =="; date
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

# helper: run step with banner
step() {
  echo
  echo "============================================================"
  echo "== $*"
  echo "============================================================"
  date
}

# --- 1) Prepare KILT QA ---
if [[ -s "data/kilt_qa.jsonl" && -s "data/needed_wiki_ids.txt" ]]; then
  step "[skip] prepare_kilt_qa (already exists)"
else
  step "prepare_kilt_qa"
  python -u scripts/prepare_kilt_qa.py \
    --task "$TASK" \
    --split "$SPLIT" \
    --max_examples "$MAX_EXAMPLES" \
    --out "data/kilt_qa.jsonl" \
    --needed_ids "data/needed_wiki_ids.txt"
fi

# --- 2) Build KILT Wikipedia subset corpus (LONG) ---
if [[ -s "data/kilt_corpus.jsonl" ]]; then
  step "[skip] build_kilt_corpus_subset (already exists)"
else
  step "build_kilt_corpus_subset (can take long)"
  python -u scripts/build_kilt_corpus_subset.py \
    --config "$WIKI_CONFIG" \
    --needed_ids "data/needed_wiki_ids.txt" \
    --out "data/kilt_corpus.jsonl" \
    --neg_pages "$NEG_PAGES" \
    --max_pages "$MAX_PAGES" \
    --max_passages "$MAX_PASSAGES"
fi

# --- 3) Build BM25 ---
if [[ -d "artifacts/bm25" ]]; then
  step "[skip] build_bm25 (already exists)"
else
  step "build_bm25"
  python -u scripts/build_bm25.py \
    --corpus "data/kilt_corpus.jsonl" \
    --out "artifacts/bm25"
fi

# --- 4) Ingest Qdrant (GPU embeddings if DEVICE=cuda) ---
step "check qdrant points_count"
POINTS="$(python - <<PY
import os
from qdrant_client import QdrantClient
url=os.environ["QDRANT_URL"]
key=os.environ.get("QDRANT_API_KEY")
col=os.environ.get("COLLECTION","wiki_mini")
c=QdrantClient(url=url, api_key=key)
try:
    print(c.get_collection(col).points_count)
except Exception:
    print(0)
PY
)"
echo "[qdrant] collection=$COLLECTION points_count=$POINTS target~=$MAX_PASSAGES"

if [[ "${POINTS}" -ge "${MAX_PASSAGES}" ]]; then
  step "[skip] ingest_qdrant (already full enough)"
else
  step "ingest_qdrant"
  INGEST_CMD=(python -u scripts/ingest_qdrant.py
    --corpus "data/kilt_corpus.jsonl"
    --collection "$COLLECTION"
    --batch_size "$BATCH_SIZE"
    --device "$DEVICE"
  )

  # pass dense_model only if your ingest script supports it
  if grep -q "dense_model" scripts/ingest_qdrant.py; then
    INGEST_CMD+=(--dense_model "$DENSE_MODEL")
  fi

  # If empty collection, recreate for a clean start; otherwise just upsert
  if [[ "${POINTS}" -eq 0 ]]; then
    INGEST_CMD+=(--recreate)
  fi

  "${INGEST_CMD[@]}"
fi

# --- 5) Generate traces ---
if [[ -s "artifacts/traces_kilt.jsonl" ]]; then
  step "[skip] generate_traces_kilt (already exists)"
else
  step "generate_traces_kilt"
  python -u scripts/generate_traces_kilt.py \
    --qa "data/kilt_qa.jsonl" \
    --bm25 "artifacts/bm25" \
    --qdrant_collection "$COLLECTION" \
    --out "artifacts/traces_kilt.jsonl" \
    --budget "$BUDGET"
fi

# --- 6) Train router ---
if [[ -d "artifacts/router" ]]; then
  step "[skip] train_router (already exists)"
else
  step "train_router"
  python -u scripts/train_router.py \
    --traces "artifacts/traces_kilt.jsonl" \
    --out "artifacts/router"
fi

# --- 7) Eval ---
step "eval_kilt"
python -u scripts/eval_kilt.py \
  --qa "data/kilt_qa.jsonl" \
  --bm25 "artifacts/bm25" \
  --qdrant_collection "$COLLECTION" \
  --router "artifacts/router" \
  --budget "$BUDGET" \
  --out_csv "results/kilt_metrics.csv"

echo
echo "== DONE =="; date
echo "[run] results:"
ls -lh results || true

echo "[run] log saved to: $LOG"
