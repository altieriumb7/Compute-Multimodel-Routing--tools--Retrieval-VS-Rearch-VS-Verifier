from __future__ import annotations
from dataclasses import dataclass
import os
import pickle
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi

from ..tokenize import simple_tokenize
from .base import Doc, ToolResult

@dataclass
class BM25Store:
    bm25: BM25Okapi
    docs: List[Dict[str, Any]]  # each has doc_id, title, text
    tokenized: List[List[str]]

def build_bm25(docs: List[Dict[str, Any]]) -> BM25Store:
    tokenized = [simple_tokenize((d.get("title","") + " " + d.get("text","")).strip()) for d in docs]
    bm25 = BM25Okapi(tokenized)
    return BM25Store(bm25=bm25, docs=docs, tokenized=tokenized)

def save_bm25(store: BM25Store, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(store, f)

def load_bm25(dir_path: str) -> BM25Store:
    with open(os.path.join(dir_path, "bm25.pkl"), "rb") as f:
        return pickle.load(f)

class BM25Tool:
    name = "bm25"

    def __init__(self, store: BM25Store, base_cost: float = 0.20, per_doc_cost: float = 0.01):
        self.store = store
        self.base_cost = base_cost
        self.per_doc_cost = per_doc_cost

    def retrieve(self, query: str, k: int) -> ToolResult:
        q = simple_tokenize(query)
        scores = self.store.bm25.get_scores(q)
        # top-k
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        docs: List[Doc] = []
        for i in idxs:
            d = self.store.docs[i]
            docs.append(Doc(
                doc_id=str(d.get("doc_id", i)),
                title=str(d.get("title","")),
                text=str(d.get("text","")),
                score=float(scores[i]),
            ))
        cost = self.base_cost + self.per_doc_cost * max(0, k)
        return ToolResult(docs=docs, cost=cost, meta={"k": k})
