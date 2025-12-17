import argparse
from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.config import QdrantConfig
from toolrouter.io import read_jsonl, write_jsonl
from toolrouter.tools.bm25 import load_bm25, BM25Tool
from toolrouter.tools.qdrant_dense import QdrantDenseTool
from toolrouter.router.actions import DEFAULT_ACTIONS
import torch
from sentence_transformers import SentenceTransformer
def wiki_id_from_doc_id(doc_id: str) -> str:
    s = str(doc_id)
    return s.split(":", 1)[0] if ":" in s else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--qdrant_collection", required=True)
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=1.0)
    args = ap.parse_args()

    qa = read_jsonl(args.qa)
    bm25 = BM25Tool(load_bm25(args.bm25))

    cfg = QdrantConfig()
    client = QdrantClient(url=cfg.url, api_key=cfg.api_key)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(args.dense_model, device=device)
    print("SentenceTransformer device:", embedder.device)

    dense = QdrantDenseTool(client=client, collection=args.qdrant_collection, embedder=embedder)

    tools = {"bm25": bm25, "dense": dense}

    traces = []
    for ex in tqdm(qa, desc="Generating KILT traces"):
        q = ex["question"]
        gold = set(ex.get("gold_wikipedia_ids") or [])
        best = None  # (hit, -cost, action_idx, cost)

        for ai, a in enumerate(DEFAULT_ACTIONS):
            if a.tool == "stop":
                continue
            res = tools[a.tool].retrieve(q, a.k)
            if res.cost > args.budget:
                continue
            hit = any(wiki_id_from_doc_id(d.doc_id) in gold for d in res.docs)
            score = (1 if hit else 0, -float(res.cost))
            if best is None or score > best[0]:
                best = (score, ai, float(res.cost), bool(hit))

        if best is None:
            # fallback: pick cheapest action overall (non-stop)
            cand = []
            for ai, a in enumerate(DEFAULT_ACTIONS):
                if a.tool == "stop": continue
                c = tools[a.tool].base_cost + tools[a.tool].per_doc_cost * a.k
                cand.append((c, ai, a))
            c, ai, a = min(cand, key=lambda t: t[0])
            best = ((0, -c), ai, float(c), False)

        traces.append({
            "qid": ex.get("qid"),
            "question": q,
            "label_action_idx": int(best[1]),
            "budget": float(args.budget),
            "chosen_cost": float(best[2]),
            "teacher_hit": bool(best[3]),
        })

    write_jsonl(args.out, traces)
    print(f"Wrote {len(traces)} traces -> {args.out}")

if __name__ == "__main__":
    main()
