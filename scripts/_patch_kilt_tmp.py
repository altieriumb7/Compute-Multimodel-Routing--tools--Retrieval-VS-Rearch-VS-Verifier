import pathlib, textwrap

root = pathlib.Path.cwd()

# 1) build_kilt_corpus_subset.py (fix: dataset id + split + trust_remote_code)
(root / "scripts").mkdir(parents=True, exist_ok=True)
build_corpus_path = root / "scripts" / "build_kilt_corpus_subset.py"
build_corpus_code = r"""
import argparse
import random
import sys
from typing import Iterable, Set

import orjson
from tqdm import tqdm

from datasets import load_dataset

def load_wiki_stream(config: str):
    last = None
    for ds_name in ("facebook/kilt_wikipedia", "kilt_wikipedia"):
        try:
            return load_dataset(ds_name, config, split="full", streaming=True)
        except TypeError:
            # vecchie versioni di datasets senza trust_remote_code
            return load_dataset(ds_name, config, split="full", streaming=True)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not load KILT Wikipedia ({config}). Last error: {last!r}")

def read_needed_ids(path: str) -> Set[int]:
    needed = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                needed.add(int(s))
            except ValueError:
                pass
    return needed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="2019-08-01")
    ap.add_argument("--needed_ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg_pages", type=int, default=5000)
    ap.add_argument("--max_pages", type=int, default=20000)
    ap.add_argument("--max_passages", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    needed = read_needed_ids(args.needed_ids)
    if not needed:
        raise RuntimeError(f"needed_ids vuoto: {args.needed_ids}")

    rnd = random.Random(args.seed)

    ds = load_wiki_stream(args.config)

    kept_pages = 0
    kept_passages = 0
    neg_kept_pages = 0
    needed_found = set()

    out_path = args.out
    out_f = open(out_path, "wb")

    def write_obj(obj):
        out_f.write(orjson.dumps(obj))
        out_f.write(b"\n")

    for ex in tqdm(ds, desc="Scanning KILT Wikipedia (streaming)"):
        if kept_pages >= args.max_pages or kept_passages >= args.max_passages:
            break

        wid_raw = ex.get("wikipedia_id", None)
        if wid_raw is None:
            continue
        try:
            wid = int(wid_raw)
        except Exception:
            continue

        title = ex.get("wikipedia_title", "") or ""
        paragraphs = []
        t = ex.get("text", None)
        if isinstance(t, dict):
            paragraphs = t.get("paragraph", []) or []
        if not isinstance(paragraphs, list):
            continue

        is_needed = wid in needed

        # includi tutte le pagine "needed" che incontri
        take = is_needed

        # prendi anche un certo numero di pagine negative (distrattori)
        if (not take) and (neg_kept_pages < args.neg_pages):
            # campiona un po' per non riempire subito di negative
            # (se vuoi piÃ¹ negative, alza neg_pages oppure aumenta la probabilitÃ )
            if rnd.random() < 0.15:
                take = True

        if not take:
            continue

        kept_pages += 1
        if is_needed:
            needed_found.add(wid)
        else:
            neg_kept_pages += 1

        for pid, ptxt in enumerate(paragraphs):
            if kept_passages >= args.max_passages:
                break
            if not isinstance(ptxt, str):
                continue
            ptxt = ptxt.strip()
            if not ptxt:
                continue

            doc = {
                "doc_id": f"{wid}:{pid}",
                "title": str(title),
                "text": ptxt,
                "wikipedia_id": str(wid),
                "paragraph_id": int(pid),
            }
            write_obj(doc)
            kept_passages += 1

    out_f.close()

    print(f"Wrote corpus: {out_path}")
    print(f"Kept pages: {kept_pages} (neg_pages={neg_kept_pages})")
    print(f"Kept passages: {kept_passages}")
    print(f"Needed ids found: {len(needed_found)}/{len(needed)}")
    if len(needed_found) < len(needed):
        missing = len(needed) - len(needed_found)
        print(f"WARNING: missing {missing} needed wikipedia_ids in this subset. "
              f"Per coprire tutto, aumenta --max_pages/--max_passages oppure lascia che lo scan vada piÃ¹ avanti.")

if __name__ == "__main__":
    main()
"""
build_corpus_path.write_text(textwrap.dedent(build_corpus_code), encoding="utf-8")

# 2) Patch qdrant_dense.py: compat con client che non ha .search
qdrant_dense_path = root / "src" / "toolrouter" / "tools" / "qdrant_dense.py"
if qdrant_dense_path.exists():
    qdrant_dense_code = r"""
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
        embedder,
        collection: str,
        base_cost: float = 0.6,
        per_doc_cost: float = 0.15,
        query_filter: Optional[Filter] = None,
    ):
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.base_cost = float(base_cost)
        self.per_doc_cost = float(per_doc_cost)
        self.query_filter = query_filter

    def _call_qdrant(self, vec, k: int):
        # Preferisci API moderne se disponibili
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
                # alcune versioni usano 'filter' al posto di 'query_filter'
                res = self.client.query_points(
                    collection_name=self.collection,
                    query=vec,
                    limit=k,
                    filter=self.query_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            return getattr(res, "points", res)

        # fallback legacy
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=k,
                query_filter=self.query_filter,
                with_payload=True,
                with_vectors=False,
            )

        raise RuntimeError("QdrantClient non espone nÃ© query_points nÃ© search. Aggiorna qdrant-client o controlla l'import.")

    def retrieve(self, query: str, k: int) -> ToolResult:
        vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        res = self._call_qdrant(vec, int(k))

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
"""
    qdrant_dense_path.write_text(textwrap.dedent(qdrant_dense_code), encoding="utf-8")

print("Patched build_kilt_corpus_subset.py and (if present) qdrant_dense.py")
