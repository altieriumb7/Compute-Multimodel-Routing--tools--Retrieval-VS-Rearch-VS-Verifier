"""Generate "teacher" routing traces for KILT QA.

This is a drop-in alternative to scripts/generate_traces_kilt.py that is more
robust for Colab/vast.ai sessions: it can append to an existing output file.

Typical use:
  python scripts/generate_traces_kilt_resume.py ... --out artifacts/traces_kilt.jsonl --resume

If the run is interrupted, rerun the same command; it will continue from the
last written line.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable

import orjson
from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.config import QdrantConfig
from toolrouter.io import read_jsonl
from toolrouter.tools.bm25 import load_bm25, BM25Tool
from toolrouter.tools.qdrant_dense import QdrantDenseTool
from toolrouter.router.actions import DEFAULT_ACTIONS


def wiki_id_from_doc_id(doc_id: str) -> str:
    s = str(doc_id)
    return s.split(":", 1)[0] if ":" in s else s


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--qdrant_collection", required=True)
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu", help="cpu|cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all")
    ap.add_argument("--flush_every", type=int, default=200)
    args = ap.parse_args()

    qa = read_jsonl(args.qa)
    if args.max_examples and args.max_examples > 0:
        qa = qa[: args.max_examples]

    bm25 = BM25Tool(load_bm25(args.bm25))

    cfg = QdrantConfig()
    # Allow explicit overrides (useful in Colab)
    url = os.environ.get("QDRANT_URL", cfg.url)
    api_key = os.environ.get("QDRANT_API_KEY", cfg.api_key)
    client = QdrantClient(url=url, api_key=api_key)

    embedder = SentenceTransformer(args.dense_model, device=args.device)
    dense = QdrantDenseTool(client=client, collection=args.qdrant_collection, embedder=embedder)

    tools: Dict[str, object] = {"bm25": bm25, "dense": dense}

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    start = 0
    if args.resume and os.path.exists(out_path):
        start = count_lines(out_path)
        if start > len(qa):
            print(f"[resume] {out_path} already has {start} lines, but QA has {len(qa)}; nothing to do.")
            return
        print(f"[resume] Found existing {out_path} with {start} lines. Resuming at index {start}/{len(qa)}")

    mode = "ab" if (args.resume and os.path.exists(out_path)) else "wb"
    written = 0

    with open(out_path, mode) as out_f:
        for ex in tqdm(qa[start:], desc="Generating KILT traces", total=len(qa) - start):
            q = ex["question"]
            gold = set(ex.get("gold_wikipedia_ids") or [])

            best = None  # (hit, -cost, action_idx, cost)

            for ai, a in enumerate(DEFAULT_ACTIONS):
                if a.tool == "stop":
                    continue

                res = tools[a.tool].retrieve(q, a.k)  # type: ignore[attr-defined]
                if float(res.cost) > float(args.budget):
                    continue

                hit = any(wiki_id_from_doc_id(d.doc_id) in gold for d in res.docs)
                score = (1 if hit else 0, -float(res.cost))
                if best is None or score > best[0]:
                    best = (score, ai, float(res.cost), bool(hit))

            if best is None:
                # fallback: choose the cheapest non-stop action
                cheapest = None
                for ai, a in enumerate(DEFAULT_ACTIONS):
                    if a.tool == "stop":
                        continue
                    tool = tools[a.tool]
                    c = float(tool.base_cost + tool.per_doc_cost * a.k)  # type: ignore[attr-defined]
                    if cheapest is None or c < cheapest[0]:
                        cheapest = (c, ai)
                assert cheapest is not None
                best = ((0, -cheapest[0]), int(cheapest[1]), float(cheapest[0]), False)

            rec = {
                "qid": ex.get("qid"),
                "question": q,
                "label_action_idx": int(best[1]),
                "budget": float(args.budget),
                "chosen_cost": float(best[2]),
                "teacher_hit": bool(best[3]),
            }
            out_f.write(orjson.dumps(rec) + b"\n")
            written += 1

            if written % int(args.flush_every) == 0:
                out_f.flush()

    print(f"Wrote {start + written} traces -> {out_path}")


if __name__ == "__main__":
    main()
