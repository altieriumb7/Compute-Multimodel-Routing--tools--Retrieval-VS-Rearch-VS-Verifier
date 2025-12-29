from __future__ import annotations
import json

def _load_done_ids(path: str) -> set[str]:
    done = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    if 'id' in ex:
                        done.add(str(ex['id']))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return done


import argparse

def load_done_ids(path):
    done = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    qid = ex.get("id")
                    if qid is not None:
                        done.add(str(qid))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return done

import os
from typing import Dict, List, Set, Tuple

from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.io import read_jsonl, write_jsonl
from toolrouter.router.actions import DEFAULT_ACTIONS, Action
from toolrouter.tokenize import simple_tokenize
from toolrouter.tools.bm25 import load_bm25
from toolrouter.tools.qdrant_dense import QdrantDenseTool


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
    done_ids = _load_done_ids(args.out)
    if done_ids:
        print(f"[resume] skipping {len(done_ids)} already-written traces")
    out_f = open(args.out, 'a', encoding='utf-8')


    done_ids = load_done_ids(args.out)
    if done_ids:
        print(f"[resume] found {len(done_ids)} existing traces in {args.out}, resuming...")

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
        if str(ex.get("id")) in done_ids:
            continue
        if str(ex.get("id")) in done_ids:
            continue
        # KILT uses 'input' for the question
        question = ex.get("input") or ex.get("question") or ex.get("query") or (ex.get("meta", {}) or {}).get("question")
        if not question:
            continue

        # Gold Wikipedia ids from KILT output/provenance
        gold = set()
        for o in ex.get("output", []):
            for prov in (o.get("provenance") or []):
                wid = prov.get("wikipedia_id")
                if wid is not None:
                    gold.add(str(wid))
        if not gold:
            continue

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

        trace_item = (
            {
                "id": ex.get("id", ""),
                "question": question,
                "gold_wikipedia_ids": list(gold),
                "label_action_idx": int(label_idx),
            }
        )

        traces.append(trace_item)
        out_f.write(json.dumps(trace_item, ensure_ascii=False) + "\n")
        done_ids.add(str(trace_item.get('id')))
        out_f.flush()
    write_jsonl(args.out, traces)
    f_out.close()
    out_f.close()
    print(f"Wrote {len(traces)} traces -> {args.out}")



if __name__ == "__main__":
    main()
