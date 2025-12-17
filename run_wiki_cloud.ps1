param(
  [string]$WikiConfig = "20231101.en",
  [int]$MaxArticles = 50000,
  [int]$MaxPassages = 250000,
  [int]$PassageWords = 120,
  [int]$OverlapWords = 20,
  [string]$Collection = "wiki_mini",
  [string]$DenseModel = "sentence-transformers/all-MiniLM-L6-v2",
  [int]$IngestBatchSize = 128,
  [double]$Budget = 1.0
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONUTF8 = "1"

if ([string]::IsNullOrWhiteSpace($env:QDRANT_URL) -or [string]::IsNullOrWhiteSpace($env:QDRANT_API_KEY)) {
  throw "Set QDRANT_URL and QDRANT_API_KEY in this PowerShell session before running."
}

New-Item -ItemType Directory -Force -Path logs, artifacts, data | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs\run_$ts.log"

function Run-Step([string]$name, [string]$cmd) {
  "`n=== $name ===`n$cmd`n" | Tee-Object -FilePath $log -Append
  & powershell -NoProfile -Command $cmd 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) { throw "Step failed: $name (exit=$LASTEXITCODE). See $log" }
}

# 0) Ensure deps + editable install
Run-Step "Install requirements" "pip install -r requirements.txt"
Run-Step "Install package editable" "pip install -e ."

# 1) Patch Qdrant tool: use query_points() (new qdrant-client API)
$qdrantDensePath = "src\toolrouter\tools\qdrant_dense.py"
@'
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
        embedder,  # SentenceTransformer-like
        base_cost: float = 1.00,
        per_doc_cost: float = 0.01,
        query_filter: Optional[Filter] = None,
    ):
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.base_cost = base_cost
        self.per_doc_cost = per_doc_cost
        self.query_filter = query_filter

    def retrieve(self, query: str, k: int) -> ToolResult:
        vec = self.embedder.encode([query], normalize_embeddings=True)[0]

        # qdrant-client >= ~1.10: unified query API
        if hasattr(self.client, "query_points"):
            hits = self.client.query_points(
                collection_name=self.collection,
                query=vec,
                limit=k,
                query_filter=self.query_filter,
                with_payload=True,
                with_vectors=False,
            )
            res = hits.points
        else:
            # fallback for older clients
            res = self.client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=k,
                query_filter=self.query_filter,
                with_payload=True,
                with_vectors=False,
            )

        docs: List[Doc] = []
        for p in res:
            payload = p.payload or {}
            docs.append(Doc(
                doc_id=str(payload.get("doc_id", getattr(p, "id", ""))),
                title=str(payload.get("title","")),
                text=str(payload.get("text","")),
                score=float(getattr(p, "score", 0.0)),
            ))

        cost = self.base_cost + self.per_doc_cost * max(0, k)
        return ToolResult(docs=docs, cost=cost, meta={"k": k})
'@ | Set-Content -Encoding UTF8 $qdrantDensePath

# 2) Add Wikipedia builder script (HF dataset wikimedia/wikipedia -> corpus.jsonl + qa.jsonl)
$wikiBuilderPath = "scripts\build_wikipedia_corpus.py"
@'
import argparse, re
import orjson
from tqdm import tqdm
from datasets import load_dataset

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def iter_passages(text: str, title: str, max_words: int, overlap_words: int):
    words = norm_ws(text).split(" ")
    if not words:
        return
    step = max(1, max_words - overlap_words)
    for i in range(0, len(words), step):
        chunk = words[i:i+max_words]
        if len(chunk) < max(20, max_words // 4):
            break
        chunk_text = " ".join(chunk).strip()
        # include title in the chunk so it’s retrievable / auditable
        yield f"{title}\n{chunk_text}"

def make_definitional_qa(title: str, text: str):
    # Try to extract "Title is ..." from the first sentence.
    t = norm_ws(text)
    first = t.split(". ", 1)[0].strip()
    # Common Wikipedia style: "April is the fourth month..."
    m = re.match(rf"^{re.escape(title)}\s+is\s+(an?\s+|the\s+)?(.+)$", first, flags=re.IGNORECASE)
    if not m:
        return None
    rhs = m.group(2)
    rhs = re.sub(r"\s+", " ", rhs).strip()
    rhs = rhs.strip(" .;:,()[]")
    if len(rhs) < 5 or len(rhs) > 120:
        return None
    q = f"What is {title}?"
    a = rhs
    return q, a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="20231101.en")
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--qa_out", default="data/qa.jsonl")
    ap.add_argument("--max_articles", type=int, default=50000)
    ap.add_argument("--max_passages", type=int, default=250000)
    ap.add_argument("--passage_words", type=int, default=120)
    ap.add_argument("--overlap_words", type=int, default=20)
    ap.add_argument("--qa_size", type=int, default=20000)
    ap.add_argument("--streaming", action="store_true", default=True)
    args = ap.parse_args()

    ds = load_dataset("wikimedia/wikipedia", args.config, split="train", streaming=args.streaming)

    n_articles = 0
    n_passages = 0
    n_qa = 0

    with open(args.out, "wb") as fcorpus, open(args.qa_out, "wb") as fqa:
        for ex in tqdm(ds, desc=f"Reading {args.config}"):
            if n_articles >= args.max_articles or n_passages >= args.max_passages:
                break
            title = norm_ws(ex.get("title",""))
            text = ex.get("text","") or ""
            url = ex.get("url","")
            aid = str(ex.get("id",""))

            if not title or not text:
                continue

            # QA (definitional) if possible
            if n_qa < args.qa_size:
                qa = make_definitional_qa(title, text)
                if qa:
                    q, a = qa
                    row = {"qid": f"{aid}:def", "question": q, "answer": a}
                    fqa.write(orjson.dumps(row)); fqa.write(b"\n")
                    n_qa += 1

            # Passages
            chunk_idx = 0
            for chunk in iter_passages(text, title, args.passage_words, args.overlap_words):
                if n_passages >= args.max_passages:
                    break
                doc = {
                    "doc_id": f"{aid}:{chunk_idx}",
                    "article_id": aid,
                    "chunk_id": chunk_idx,
                    "title": title,
                    "url": url,
                    "text": chunk,
                }
                fcorpus.write(orjson.dumps(doc)); fcorpus.write(b"\n")
                n_passages += 1
                chunk_idx += 1

            n_articles += 1

    print(f"Wrote corpus: {args.out} | passages={n_passages}, articles={n_articles}")
    print(f"Wrote QA:     {args.qa_out} | qa={n_qa}")

if __name__ == "__main__":
    main()
'@ | Set-Content -Encoding UTF8 $wikiBuilderPath

# 3) Build Wikipedia subset -> data/corpus.jsonl + data/qa.jsonl
Run-Step "Build Wikipedia corpus+QA" ("python scripts\build_wikipedia_corpus.py " +
  "--config $WikiConfig --out data\corpus.jsonl --qa_out data\qa.jsonl " +
  "--max_articles $MaxArticles --max_passages $MaxPassages --passage_words $PassageWords --overlap_words $OverlapWords")

# 4) BM25
Run-Step "Build BM25" "python scripts\build_bm25.py --corpus data\corpus.jsonl --out artifacts\bm25"

# 5) Qdrant ingest (cloud)
Run-Step "Ingest Qdrant" ("python scripts\ingest_qdrant.py --corpus data\corpus.jsonl --collection $Collection " +
  "--model $DenseModel --batch_size $IngestBatchSize --recreate")

# 6) Traces -> Train -> Eval
Run-Step "Generate traces" ("python scripts\generate_traces.py --qa data\qa.jsonl --bm25 artifacts\bm25 --qdrant_collection $Collection --out artifacts\traces.jsonl --budget $Budget")
Run-Step "Train router" "python scripts\train_router.py --traces artifacts\traces.jsonl --out artifacts\router"
Run-Step "Eval router" ("python scripts\eval_router.py --qa data\qa.jsonl --bm25 artifacts\bm25 --qdrant_collection $Collection --router artifacts\router --budget $Budget")

"`nDONE. See log: $log`nArtifacts: artifacts\bm25, artifacts\router, artifacts\traces.jsonl, data\corpus.jsonl, data\qa.jsonl`n" | Tee-Object -FilePath $log -Append
