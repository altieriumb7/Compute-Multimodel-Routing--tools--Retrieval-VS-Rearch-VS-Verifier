import argparse, csv, time
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Tuple

from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.config import QdrantConfig
from toolrouter.io import read_jsonl
from toolrouter.tools.bm25 import load_bm25, BM25Tool
from toolrouter.tools.qdrant_dense import QdrantDenseTool
from toolrouter.router.model import load_router
from toolrouter.router.actions import DEFAULT_ACTIONS

DENSE_COST = 0.65  # must match your cost model / dense avg_cost gating


_t0 = time.time()
def _ts(msg: str):
    print(f"[time] {msg}: {time.time() - _t0:.2f}s", flush=True)


def wiki_id_from_doc_id(doc_id: str) -> str:
    s = str(doc_id)
    return s.split(":", 1)[0] if ":" in s else s


def _docs_cost(res: Any) -> Tuple[List[Any], float]:
    """Normalize tool output -> (docs, cost)."""
    if res is None:
        return [], 0.0
    if isinstance(res, list):
        return res, 0.0
    docs = getattr(res, "docs", None)
    if not isinstance(docs, list):
        docs = []
    cost = getattr(res, "cost", 0.0)
    try:
        cost = float(cost or 0.0)
    except Exception:
        cost = 0.0
    return docs, cost


def hit_rate(tool, qa, k: int) -> Tuple[float, float]:
    hits = 0
    costs = 0.0

    for ex in qa:
        q = ex.get("question", "")
        gold = str(ex.get("wiki_id", ""))

        res = tool.retrieve(q, k=k)
        docs, cost = _docs_cost(res)
        costs += cost

        pred_ids = []
        for d in docs[:k]:
            if d is None:
                continue
            if isinstance(d, dict):
                wid = d.get("wiki_id") or d.get("id") or d.get("doc_id") or ""
            else:
                wid = getattr(d, "wiki_id", "") or getattr(d, "id", "") or getattr(d, "doc_id", "")
            if wid:
                pred_ids.append(wiki_id_from_doc_id(wid))

        if gold and gold in pred_ids:
            hits += 1

    n = max(1, len(qa))
    return hits / n, costs / n


def main():
    _ts("enter main")
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--qdrant_collection", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--out_csv", default="results/kilt_metrics.csv")
    args = ap.parse_args()

    qa = read_jsonl(args.qa)
    _ts(f"loaded qa n={len(qa)}")

    bm25 = BM25Tool(load_bm25(args.bm25))
    _ts("loaded bm25")

    dense = None
    if args.budget >= DENSE_COST:
        cfg = QdrantConfig()
        client = QdrantClient(url=cfg.url, api_key=cfg.api_key, timeout=30)
        embedder = SentenceTransformer(args.dense_model)
        dense = QdrantDenseTool(client=client, collection=args.qdrant_collection, embedder=embedder)
        _ts("built dense")
    else:
        _ts("skipped dense (budget too low)")

    router = load_router(args.router)
    _ts("loaded router")

    rows = []

    # Baselines
    for k in (5, 20):
        hr, ac = hit_rate(bm25, qa, k)
        rows.append(("bm25", k, hr, ac))

        if dense is not None:
            hr, ac = hit_rate(dense, qa, k)
            rows.append(("dense", k, hr, ac))

    # Router (budget-enforced)
    hits = 0
    total_cost = 0.0
    use = Counter()

    tools = {"bm25": bm25}
    if dense is not None:
        tools["dense"] = dense

    # cheapest non-stop among available tools
    cheapest = None
    for a in DEFAULT_ACTIONS:
        if a.tool == "stop":
            continue
        tool = tools.get(a.tool)
        if tool is None:
            continue
        c = tool.base_cost + tool.per_doc_cost * a.k
        if cheapest is None or c < cheapest[0]:
            cheapest = (c, a)

    for ex in tqdm(qa, desc="Eval router"):
        q = ex.get("question", "")
        gold = str(ex.get("wiki_id", ""))

        # choose action
        a = router.predict_action(q)

        # If action tool unavailable (e.g., dense is None), fallback to cheapest
        if a.tool != "stop" and tools.get(a.tool) is None:
            a = cheapest[1]

        if a.tool == "stop":
            use["stop"] += 1
            continue

        tool = tools[a.tool]
        # expected cost
        expected = tool.base_cost + tool.per_doc_cost * a.k
        if expected > args.budget:
            # fallback if chosen action too expensive
            a = cheapest[1]
            tool = tools[a.tool]
            expected = tool.base_cost + tool.per_doc_cost * a.k
            if expected > args.budget:
                use["stop"] += 1
                continue

        res = tool.retrieve(q, k=a.k)
        docs, cost = _docs_cost(res)
        total_cost += cost
        use[a.tool] += 1

        pred_ids = []
        for d in docs[: a.k]:
            if d is None:
                continue
            if isinstance(d, dict):
                wid = d.get("wiki_id") or d.get("id") or d.get("doc_id") or ""
            else:
                wid = getattr(d, "wiki_id", "") or getattr(d, "id", "") or getattr(d, "doc_id", "")
            if wid:
                pred_ids.append(wiki_id_from_doc_id(wid))

        if gold and gold in pred_ids:
            hits += 1

    n = max(1, len(qa))
    router_hr = hits / n
    router_ac = total_cost / n
    rows.append((f"router(budget={args.budget})", "-", router_hr, router_ac))

    print("=== KILT hit@k (page-level) ===")
    for name, k, hr, ac in rows:
        if k == "-":
            print(f"{name}: hit_rate={hr:.3f} avg_cost={ac:.3f} usage={dict(use)}")
        else:
            print(f"{name}@{k}: hit_rate={hr:.3f} avg_cost={ac:.3f}")

    # write csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "k", "hit_rate", "avg_cost"])
        for name, k, hr, ac in rows:
            w.writerow([name, k, hr, ac])

    print(f"Wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
