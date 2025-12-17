import argparse
from tqdm import tqdm
import orjson

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from sentence_transformers import SentenceTransformer

from toolrouter.config import QdrantConfig
import torch
from sentence_transformers import SentenceTransformer
def iter_jsonl(path: str):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--vector_size", type=int, default=384)
    args = ap.parse_args()

    cfg = QdrantConfig()
    client = QdrantClient(url=cfg.url, api_key=cfg.api_key)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(args.dense_model, device=device)
    print("SentenceTransformer device:", embedder.device)

    try:
        test = embedder.encode(["hello"], normalize_embeddings=True)
        vec_size = int(test.shape[-1])
    except Exception:
        vec_size = args.vector_size

    if args.recreate:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )

    batch = []
    texts = []
    count = 0
    point_id = 0

    for doc in tqdm(iter_jsonl(args.corpus), desc="Embedding+upsert (stream)"):
        title = str(doc.get("title",""))
        text = str(doc.get("text",""))
        payload = {
            "doc_id": str(doc.get("doc_id", point_id)),
            "title": title,
            "text": text,
            "wikipedia_id": doc.get("wikipedia_id"),
            "paragraph_id": doc.get("paragraph_id"),
            "url": doc.get("url",""),
        }
        batch.append((point_id, payload, (title + "\n" + text).strip()))
        point_id += 1

        if len(batch) >= args.batch_size:
            ids = [b[0] for b in batch]
            payloads = [b[1] for b in batch]
            ttxt = [b[2] for b in batch]
            vecs = embedder.encode(ttxt, batch_size=args.batch_size, normalize_embeddings=True)
            points = [
                PointStruct(id=i, vector=vecs[j].tolist(), payload=payloads[j])
                for j, i in enumerate(ids)
            ]
            client.upsert(collection_name=args.collection, points=points)
            count += len(batch)
            batch = []

    if batch:
        ids = [b[0] for b in batch]
        payloads = [b[1] for b in batch]
        ttxt = [b[2] for b in batch]
        vecs = embedder.encode(ttxt, batch_size=args.batch_size, normalize_embeddings=True)
        points = [
            PointStruct(id=i, vector=vecs[j].tolist(), payload=payloads[j])
            for j, i in enumerate(ids)
        ]
        client.upsert(collection_name=args.collection, points=points)
        count += len(batch)

    print(f"Ingested ~{count} docs into Qdrant collection '{args.collection}' @ {cfg.url}")

if __name__ == "__main__":
    main()
