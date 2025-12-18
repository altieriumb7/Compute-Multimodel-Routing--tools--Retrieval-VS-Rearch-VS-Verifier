from __future__ import annotations

import argparse
import os
import uuid
from typing import Any, Dict, Iterable, List

import orjson
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from sentence_transformers import SentenceTransformer


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def doc_uuid(doc_id: str) -> uuid.UUID:
    # Deterministic UUID so re-ingesting is idempotent
    return uuid.uuid5(uuid.NAMESPACE_URL, doc_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch_size", type=int, default=256)

    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None, help="cpu|cuda. Default: auto-detect")
    ap.add_argument("--max_text_len", type=int, default=1000, help="truncate payload text (0 = keep full)")

    ap.add_argument("--qdrant_url", default=os.environ.get("QDRANT_URL"))
    ap.add_argument("--qdrant_api_key", default=os.environ.get("QDRANT_API_KEY"))
    args = ap.parse_args()

    if not args.qdrant_url:
        raise SystemExit("Missing QDRANT_URL (env) or --qdrant_url")

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print(f"[ingest_qdrant] url={args.qdrant_url} collection={args.collection}")
    print(f"[ingest_qdrant] model={args.dense_model} device={device} batch_size={args.batch_size}")

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)

    embedder = SentenceTransformer(args.dense_model, device=device)
    dim = int(embedder.get_sentence_embedding_dimension())

    vectors_config = qm.VectorParams(size=dim, distance=qm.Distance.COSINE)

    if args.recreate:
        client.recreate_collection(collection_name=args.collection, vectors_config=vectors_config)
    else:
        # Create if missing
        try:
            exists = client.collection_exists(args.collection)
        except Exception:
            # fallback
            cols = client.get_collections().collections
            exists = any(c.name == args.collection for c in cols)
        if not exists:
            client.create_collection(collection_name=args.collection, vectors_config=vectors_config)

    batch: List[Dict[str, Any]] = []
    texts: List[str] = []

    def flush():
        nonlocal batch, texts
        if not batch:
            return

        vecs = embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        points: List[qm.PointStruct] = []
        for rec, v in zip(batch, vecs):
            doc_id = str(rec["doc_id"])

            title = str(rec.get("title", ""))
            text = str(rec.get("text", ""))

            if args.max_text_len and args.max_text_len > 0:
                text = text[: args.max_text_len]

            payload = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
            }

            points.append(
                qm.PointStruct(
                    id=doc_uuid(doc_id),
                    vector=v.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=args.collection, points=points, wait=True)

        batch = []
        texts = []

    pbar = tqdm(desc="Embedding+upsert (stream)", unit="docs")
    for rec in iter_jsonl(args.corpus):
        if "doc_id" not in rec:
            continue
        t = rec.get("text", "")
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue

        batch.append(rec)
        texts.append(t)

        if len(batch) >= args.batch_size:
            flush()
            pbar.update(args.batch_size)

    if batch:
        flush()
        pbar.update(len(batch))

    pbar.close()

    info = client.get_collection(args.collection)
    print(f"[ingest_qdrant] DONE points_count={info.points_count}")


if __name__ == "__main__":
    main()
