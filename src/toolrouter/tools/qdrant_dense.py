from __future__ import annotations
import time
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


    def _query(self, vec, k):

        # Robust remote call with retries for transient Qdrant issues

        last_err = None

        for attempt in range(12):

            try:

                return self.client.query_points(

                    collection_name=self.collection,

                    query=vec,

                    limit=int(k),

                    with_payload=True,

                    with_vectors=False,

                )

            except Exception as e:

                last_err = e

                msg = str(e)

                transient = any(t in msg for t in [

                    "502", "Bad Gateway", "503", "Service Unavailable", "504",

                    "timed out", "ConnectTimeout", "ReadTimeout", "ResponseHandlingException",

                    "Connection reset", "RemoteProtocolError",
                    "ConnectError",
                    "[Errno 111]",
                    "Connection refused"

                ])

                if (not transient) or (attempt == 11):

                    raise

                sleep_s = min(60, 2 ** attempt)

                print(f"[qdrant_dense] transient error: {msg[:160]} ... retry in {sleep_s}s ({attempt+1}/12)")

                time.sleep(sleep_s)

        raise last_err
