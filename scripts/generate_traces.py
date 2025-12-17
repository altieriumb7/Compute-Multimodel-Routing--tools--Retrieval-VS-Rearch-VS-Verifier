import argparse
from tqdm import tqdm
from dataclasses import asdict

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.config import QdrantConfig
from toolrouter.io import read_jsonl, write_jsonl
from toolrouter.metrics import retrieval_contains_answer
from toolrouter.tools.bm25 import load_bm25, BM25Tool
from toolrouter.tools.qdrant_dense import QdrantDenseTool
from toolrouter.router.actions import DEFAULT_ACTIONS, Action

def concat_docs(docs):
    return "\n\n".join([d.text for d in docs])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="QA JSONL: qid, question, answer")
    ap.add_argument("--bm25", required=True, help="BM25 artifact dir")
    ap.add_argument("--qdrant_collection", required=True)
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True, help="Output traces.jsonl")
    ap.add_argument("--budget", type=float, default=1.0, help="Max allowed tool cost")
    args = ap.parse_args()

    qa = read_jsonl(args.qa)

    bm25_store = load_bm25(args.bm25)
    bm25_tool = BM25Tool(bm25_store)

    cfg = QdrantConfig()
    client = QdrantClient(url=cfg.url, api_key=cfg.api_key)
    embedder = SentenceTransformer(args.dense_model)
    dense_tool = QdrantDenseTool(client=client, collection=args.qdrant_collection, embedder=embedder)

    tools = {"bm25": bm25_tool, "dense": dense_tool}

    traces = []
    for ex in tqdm(qa, desc="Generating traces"):
        q = ex["question"]
        ans = ex.get("answer")

        best = None  # (utility, -cost, action)
        for ai, action in enumerate(DEFAULT_ACTIONS):
            if action.tool == "stop":
                # STOP is useful if you want to let router abstain; here STOP never "wins" unless nothing else helps.
                continue
            res = tools[action.tool].retrieve(q, action.k)
            if res.cost > args.budget:
                continue
            ok = retrieval_contains_answer(concat_docs(res.docs), ans)
            utility = 1.0 if ok else 0.0
            score = (utility, -res.cost)
            if best is None or score > best[0]:
                best = (score, ai, res.cost, ok)

        if best is None:
            # fallback: choose cheapest action among allowed
            allowed = [(i,a) for i,a in enumerate(DEFAULT_ACTIONS) if a.tool != "stop"]
            i, a = min(allowed, key=lambda t: (tools[t[1].tool].base_cost + tools[t[1].tool].per_doc_cost*t[1].k))
            chosen_idx = i
            chosen_cost = tools[a.tool].retrieve(q, a.k).cost
            chosen_ok = False
        else:
            chosen_idx = best[1]
            chosen_cost = best[2]
            chosen_ok = best[3]

        traces.append({
            "qid": ex.get("qid"),
            "question": q,
            "answer": ans,
            "label_action_idx": int(chosen_idx),
            "budget": float(args.budget),
            "chosen_cost": float(chosen_cost),
            "teacher_retrieval_success": bool(chosen_ok),
        })

    write_jsonl(args.out, traces)
    print(f"Wrote {len(traces)} traces to {args.out}")

if __name__ == "__main__":
    main()
