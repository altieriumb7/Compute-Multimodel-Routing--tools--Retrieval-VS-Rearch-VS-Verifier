#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[patch] Writing router model (sklearn>=1.8 compatible + balanced)..."
cat > src/toolrouter/router/model.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .actions import Action, DEFAULT_ACTIONS


@dataclass
class RouterModel:
    pipeline: Pipeline
    actions: List[Action]

    def predict_action_idx(self, text: str) -> int:
        return int(self.pipeline.predict([text])[0])

    def predict_action(self, text: str) -> Action:
        idx = self.predict_action_idx(text)
        return self.actions[idx]


def train_router_model(texts: Sequence[str], labels: Sequence[int], actions: Sequence[Action] = DEFAULT_ACTIONS) -> RouterModel:
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=200_000)),
            # sklearn >= 1.8: avoid multi_class / n_jobs args; use balanced to fight label skew
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipe.fit(list(texts), list(labels))
    return RouterModel(pipeline=pipe, actions=list(actions))


def save_router(model: RouterModel, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "router.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_router(dir_path: str) -> RouterModel:
    path = os.path.join(dir_path, "router.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
PY

echo "[patch] Writing improved KILT trace generator (rank-aware oracle)..."
cat > scripts/generate_traces_kilt.py <<'PY'
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Set, Tuple

from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.io import read_jsonl, write_jsonl
from toolrouter.router.actions import DEFAULT_ACTIONS, Action
from toolrouter.tokenize import simple_tokenize
from toolrouter.tools.bm25 import load_bm25
from toolrouter.tools.dense import QdrantDenseTool


def wiki_id_from_doc_id(doc_id: str) -> str:
    # corpus doc_id is like "334:0" -> page id "334"
    return str(doc_id).split(":", 1)[0]


def best_rank(docs, gold: Set[str]) -> int:
    for i, d in enumerate(docs):
        if wiki_id_from_doc_id(d.doc_id) in gold:
            return i + 1
    return 10**9


def pick_best_action(
    actions: List[Action],
    results: Dict[Tuple[str, int], Tuple[List, float]],  # (tool,k)->(docs,cost)
    gold: Set[str],
    budget: float,
) -> int:
    # Score tuple: (hit, -rank, -cost). Budget filters candidates.
    best_idx = None
    best_score = None

    for idx, a in enumerate(actions):
        if a.tool == "stop":
            docs, cost = [], 0.0
        else:
            docs, cost = results[(a.tool, a.k)]

        if cost > budget:
            continue

        r = best_rank(docs, gold)
        hit = 1 if r < 10**9 else 0
        score = (hit, -r, -cost)

        if best_score is None or score > best_score:
            best_score = score
            best_idx = idx

    # stop is always within budget, but keep a fallback anyway
    return 0 if best_idx is None else best_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--qdrant_collection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=1.0)

    ap.add_argument("--qdrant_url", default=os.environ.get("QDRANT_URL", ""))
    ap.add_argument("--qdrant_api_key", default=os.environ.get("QDRANT_API_KEY", ""))

    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="auto")  # "cuda" / "cpu" / "auto"
    args = ap.parse_args()

    if not args.qdrant_url:
        raise SystemExit("Missing QDRANT_URL (env) or --qdrant_url")

    # tools
    bm25_store = load_bm25(args.bm25)

    # NOTE: we want to avoid recomputing dense embeddings multiple times per query.
    # We'll query Qdrant once with max_k and slice for k=5/20.
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    client = QdrantClient(url=args.qdrant_url, api_key=(args.qdrant_api_key or None))
    embedder = SentenceTransformer(args.dense_model, device=device)
    dense = QdrantDenseTool(client=client, collection=args.qdrant_collection, embedder=embedder)

    # action set
    actions = list(DEFAULT_ACTIONS)
    max_k = max((a.k for a in actions if a.tool != "stop"), default=20)

    qa = read_jsonl(args.qa)
    traces = []

    # small helpers to build docs like the tools do
    def bm25_topk(question: str, k: int):
        q = simple_tokenize(question)
        scores = bm25_store.bm25.get_scores(q)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        docs = []
        for i in idxs:
            d = bm25_store.docs[i]
            docs.append(
                type("DocLike", (), {
                    "doc_id": str(d.get("doc_id", i)),
                    "title": str(d.get("title", "")),
                    "text": str(d.get("text", "")),
                    "score": float(scores[i]),
                })()
            )
        cost = 0.20 + 0.01 * k
        return docs, float(cost)

    def dense_topk(question: str, k: int):
        vec = embedder.encode([question], normalize_embeddings=True)[0]
        res = dense._query(vec, int(k))  # uses compatibility logic inside tool
        docs = []
        for p in res:
            payload = getattr(p, "payload", None) or {}
            pid = getattr(p, "id", None)
            score = getattr(p, "score", 0.0)
            docs.append(
                type("DocLike", (), {
                    "doc_id": str(payload.get("doc_id", pid)),
                    "title": str(payload.get("title", "")),
                    "text": str(payload.get("text", "")),
                    "score": float(score),
                })()
            )
        cost = 0.60 + 0.01 * k
        return docs, float(cost)

    for ex in tqdm(qa, desc="Generating KILT traces"):
        question = ex["question"]
        gold = set(map(str, ex.get("gold_wikipedia_ids", [])))

        # compute once per query
        bm25_docs_max, _ = bm25_topk(question, max_k)
        dense_docs_max, _ = dense_topk(question, max_k)

        # build per-(tool,k) results by slicing (fast)
        results: Dict[Tuple[str, int], Tuple[List, float]] = {}
        for a in actions:
            if a.tool == "stop":
                continue
            if a.tool == "bm25":
                docs = bm25_docs_max[: a.k]
                cost = 0.20 + 0.01 * a.k
            elif a.tool == "dense":
                docs = dense_docs_max[: a.k]
                cost = 0.60 + 0.01 * a.k
            else:
                raise RuntimeError(f"Unknown tool: {a.tool}")
            results[(a.tool, a.k)] = (docs, float(cost))

        label_idx = pick_best_action(actions, results, gold, args.budget)

        traces.append(
            {
                "id": ex.get("id", ""),
                "question": question,
                "gold_wikipedia_ids": list(gold),
                "label_action_idx": int(label_idx),
            }
        )

    write_jsonl(traces, args.out)
    print(f"Wrote {len(traces)} traces -> {args.out}")


if __name__ == "__main__":
    main()
PY

echo "[patch] Done."
echo "[patch] Now re-run: generate_traces_kilt.py -> train_router.py -> eval_kilt.py"
