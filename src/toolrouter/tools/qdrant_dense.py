from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

from .base import Doc, ToolResult

@dataclass
class QdrantDenseTool:
    name: str = "dense"

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder,
        base_cost: float = 0.60,
        per_doc_cost: float = 0.01,
        query_filter: Optional[Filter] = None,
    ):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.base_cost = float(base_cost)
        self.per_doc_cost = float(per_doc_cost)
        self.query_filter = query_filter

    def _query(self, vec, k: int):
        if hasattr(self.client, "query_points"):
            try:
                res = self.client.query_points(
                    collection_name=self.collection,
                    query=vec,
                    limit=k,
                    query_filter=self.query_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            except TypeError:
                res = self.client.query_points(
                    collection_name=self.collection,
                    query=vec,
                    limit=k,
                    filter=self.query_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            return getattr(res, "points", res)

        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=k,
                query_filter=self.query_filter,
                with_payload=True,
                with_vectors=False,
            )

        raise RuntimeError("QdrantClient has neither query_points nor search.")

    def retrieve(self, query: str, k: int) -> ToolResult:
        vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        res = self._query(vec, int(k))

        docs: List[Doc] = []
        for p in res:
            payload = getattr(p, "payload", None) or {}
            pid = getattr(p, "id", None)
            score = getattr(p, "score", 0.0)
            docs.append(Doc(
                doc_id=str(payload.get("doc_id", pid)),
                title=str(payload.get("title","")),
                text=str(payload.get("text","")),
                score=float(score),
            ))
        cost = self.base_cost + self.per_doc_cost * max(0, int(k))
        return ToolResult(docs=docs, cost=cost, meta={"k": int(k)})
