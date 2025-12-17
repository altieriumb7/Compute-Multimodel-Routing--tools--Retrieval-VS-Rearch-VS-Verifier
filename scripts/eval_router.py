import argparse
from tqdm import tqdm
from dataclasses import asdict

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.config import QdrantConfig
from toolrouter.io import read_jsonl
from toolrouter.metrics import retrieval_contains_answer, exact_match
from toolrouter.tools.bm25 import load_bm25, BM25Tool
from toolrouter.tools.qdrant_dense import QdrantDenseTool
from toolrouter.router.model import load_router
from toolrouter.router.actions import Action
from toolrouter.llm import LLM

def concat_docs(docs):
    return "\n\n".join([d.text for d in docs])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--qdrant_collection", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--budget", type=float, default=1.0)

    ap.add_argument("--llm_model", default=None, help="Optional HF model for answer generation")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    qa = read_jsonl(args.qa)

    bm25_store = load_bm25(args.bm25)
    bm25_tool = BM25Tool(bm25_store)

    cfg = QdrantConfig()
    client = QdrantClient(url=cfg.url, api_key=cfg.api_key)
    embedder = SentenceTransformer(args.dense_model)
    dense_tool = QdrantDenseTool(client=client, collection=args.qdrant_collection, embedder=embedder)

    tools = {"bm25": bm25_tool, "dense": dense_tool}

    router = load_router(args.router)

    llm = LLM(args.llm_model, max_new_tokens=args.max_new_tokens) if args.llm_model else None

    n = 0
    retrieval_ok = 0
    em_ok = 0
    total_cost = 0.0
    tool_counts = {"bm25": 0, "dense": 0, "stop": 0}

    for ex in tqdm(qa, desc="Eval"):
        q = ex["question"]
        ans = ex.get("answer")
        action = router.predict_action(q)

        if action.tool == "stop":
            tool_counts["stop"] += 1
            # no retrieval
            docs = []
            cost = 0.0
        else:
            res = tools[action.tool].retrieve(q, action.k)
            docs = res.docs
            cost = float(res.cost)
            tool_counts[action.tool] += 1

        n += 1
        total_cost += cost

        retrieved_text = concat_docs(docs)
        ok_retr = retrieval_contains_answer(retrieved_text, ans)
        retrieval_ok += int(ok_retr)

        if llm:
            pred = llm.answer(q, [ {"title": d.title, "text": d.text} for d in docs ])
            em_ok += int(exact_match(pred, ans))

    print("=== Results ===")
    print(f"N: {n}")
    print(f"Avg tool cost: {total_cost / max(1,n):.4f} (budget={args.budget})")
    print(f"Retrieval success (answer substring in retrieved docs): {retrieval_ok/n:.3f}")
    if llm:
        print(f"Exact match (LLM answers): {em_ok/n:.3f}")
    print(f"Tool usage: {tool_counts}")

if __name__ == "__main__":
    main()
