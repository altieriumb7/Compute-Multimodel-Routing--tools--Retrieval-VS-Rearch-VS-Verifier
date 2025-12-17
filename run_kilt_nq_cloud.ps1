Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# UTF-8 (evita progress bar "strana")
try { chcp 65001 | Out-Null } catch {}

$Root = $PSScriptRoot
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (!(Test-Path $Py)) { throw "Non trovo $Py. Crea la venv in .\.venv prima." }

# dirs
New-Item -ItemType Directory -Force -Path "logs","data","artifacts","results","scripts" | Out-Null
$Log = Join-Path "logs" ("kilt_run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
function Run-Step {
  param([string]$Name, [string]$Exe, [Parameter(ValueFromRemainingArguments=$true)][string[]]$CmdArgs)

  "`n=== $Name ===" | Tee-Object -FilePath $Log -Append | Out-Null
  ($Exe + " " + ($CmdArgs -join " ")) | Tee-Object -FilePath $Log -Append | Out-Null

  $tmpOut = New-TemporaryFile
  $tmpErr = New-TemporaryFile

  $p = Start-Process -FilePath $Exe -ArgumentList $CmdArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $tmpOut.FullName -RedirectStandardError $tmpErr.FullName

  Get-Content $tmpOut.FullName, $tmpErr.FullName | Tee-Object -FilePath $Log -Append | Out-Null
  Remove-Item $tmpOut.FullName, $tmpErr.FullName -ErrorAction SilentlyContinue

  if ($p.ExitCode -ne 0) { throw "Step failed: $Name (exit=$($p.ExitCode)). See $Log" }
}


# --- Pin datasets (<4) perchÃ© KILT Wikipedia usa loading script ---
Run-Step "Pin datasets (<4.0.0)" $Py "-m" "pip" "install" "--upgrade" "datasets<4.0.0"

# install deps + editable (aggiungo anche constraint datasets<4 per non ri-upgradare)
Run-Step "Install requirements (with datasets<4 constraint)" $Py "-m" "pip" "install" "-r" "requirements.txt" "datasets<4.0.0"
Run-Step "Install package editable" $Py "-m" "pip" "install" "-e" "."

# --- Write/patch scripts (NO python -c) ---

@'
import argparse
from datasets import load_dataset
import orjson

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="nq")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max_examples", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--needed_ids", required=True)
    args = ap.parse_args()

    ds = load_dataset("facebook/kilt_tasks", args.task, split=args.split, trust_remote_code=True)

    out_f = open(args.out, "wb")
    needed = set()

    n = 0
    for ex in ds:
        if n >= args.max_examples:
            break

        qid = ex.get("id")
        question = ex.get("input")
        outputs = ex.get("output") or []

        answers = []
        gold_ids = set()
        for o in outputs:
            ans = o.get("answer")
            if isinstance(ans, list):
                answers.extend([a for a in ans if isinstance(a, str)])
            elif isinstance(ans, str):
                answers.append(ans)

            for prov in (o.get("provenance") or []):
                wid = prov.get("wikipedia_id")
                if wid is None:
                    continue
                gold_ids.add(str(wid))

        if not question or not gold_ids:
            continue

        for wid in gold_ids:
            needed.add(wid)

        rec = {
            "qid": qid,
            "question": question,
            "answers": list(dict.fromkeys(answers))[:10],
            "gold_wikipedia_ids": sorted(gold_ids),
        }
        out_f.write(orjson.dumps(rec) + b"\n")
        n += 1

    out_f.close()

    with open(args.needed_ids, "w", encoding="utf-8") as f:
        for wid in sorted(needed, key=lambda x: int(x)):
            f.write(wid + "\n")

    print(f"Wrote QA: {args.out} (n={n})")
    print(f"Wrote needed wiki ids: {args.needed_ids} (pages={len(needed)})")

if __name__ == "__main__":
    main()
'@ | Set-Content -Encoding UTF8 "scripts\prepare_kilt_qa.py"

@'
import argparse
import random
import orjson
from tqdm import tqdm
from datasets import load_dataset

def read_needed(path: str):
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t)
    return s

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

    needed = read_needed(args.needed_ids)
    rnd = random.Random(args.seed)

    # NOTE: split corretto per KILT Wikipedia Ã¨ "full"
    ds = load_dataset("facebook/kilt_wikipedia", args.config, split="full", streaming=True, trust_remote_code=True)

    out_f = open(args.out, "wb")
    kept_pages = 0
    kept_passages = 0
    neg_pages = 0
    found = set()

    def write(doc):
        out_f.write(orjson.dumps(doc) + b"\n")

    for ex in tqdm(ds, desc="Scan KILT Wikipedia (streaming)"):
        if kept_pages >= args.max_pages or kept_passages >= args.max_passages:
            break

        wid = ex.get("wikipedia_id")
        if wid is None:
            continue
        wid = str(wid)

        title = ex.get("wikipedia_title") or ""
        text = ex.get("text") or {}
        paras = text.get("paragraph") if isinstance(text, dict) else None
        if not isinstance(paras, list) or not paras:
            continue

        is_needed = wid in needed
        take = is_needed

        if (not take) and (neg_pages < args.neg_pages):
            if rnd.random() < 0.15:
                take = True

        if not take:
            continue

        kept_pages += 1
        if is_needed:
            found.add(wid)
        else:
            neg_pages += 1

        for pid, ptxt in enumerate(paras):
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
            }
            write(doc)
            kept_passages += 1

    out_f.close()
    print(f"Wrote corpus: {args.out}")
    print(f"Kept pages: {kept_pages} (neg_pages={neg_pages})")
    print(f"Kept passages: {kept_passages}")
    print(f"Needed ids found: {len(found)}/{len(needed)}")

if __name__ == "__main__":
    main()
'@ | Set-Content -Encoding UTF8 "scripts\build_kilt_corpus_subset.py"

# Patch qdrant_dense: support query_points (client.search puÃ² non esserci)
@'
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
'@ | Set-Content -Encoding UTF8 "src\toolrouter\tools\qdrant_dense.py"

# (opzionale ma utile) Action space un po' piÃ¹ paper-friendly
@'
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

ToolName = Literal["bm25", "dense", "stop"]

@dataclass(frozen=True)
class Action:
    tool: ToolName
    k: int = 0
    def is_stop(self) -> bool:
        return self.tool == "stop"

DEFAULT_ACTIONS = [
    Action("bm25", 5),
    Action("bm25", 20),
    Action("dense", 5),
    Action("dense", 20),
    Action("stop", 0),
]
'@ | Set-Content -Encoding UTF8 "src\toolrouter\router\actions.py"

# KILT traces
@'
import argparse
from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.config import QdrantConfig
from toolrouter.io import read_jsonl, write_jsonl
from toolrouter.tools.bm25 import load_bm25, BM25Tool
from toolrouter.tools.qdrant_dense import QdrantDenseTool
from toolrouter.router.actions import DEFAULT_ACTIONS

def wiki_id_from_doc_id(doc_id: str) -> str:
    s = str(doc_id)
    return s.split(":", 1)[0] if ":" in s else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--qdrant_collection", required=True)
    ap.add_argument("--dense_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=1.0)
    args = ap.parse_args()

    qa = read_jsonl(args.qa)
    bm25 = BM25Tool(load_bm25(args.bm25))

    cfg = QdrantConfig()
    client = QdrantClient(url=cfg.url, api_key=cfg.api_key)
    embedder = SentenceTransformer(args.dense_model)
    dense = QdrantDenseTool(client=client, collection=args.qdrant_collection, embedder=embedder)

    tools = {"bm25": bm25, "dense": dense}

    traces = []
    for ex in tqdm(qa, desc="Generating KILT traces"):
        q = ex["question"]
        gold = set(ex.get("gold_wikipedia_ids") or [])
        best = None  # (hit, -cost, action_idx, cost)

        for ai, a in enumerate(DEFAULT_ACTIONS):
            if a.tool == "stop":
                continue
            res = tools[a.tool].retrieve(q, a.k)
            if res.cost > args.budget:
                continue
            hit = any(wiki_id_from_doc_id(d.doc_id) in gold for d in res.docs)
            score = (1 if hit else 0, -float(res.cost))
            if best is None or score > best[0]:
                best = (score, ai, float(res.cost), bool(hit))

        if best is None:
            # fallback: pick cheapest action overall (non-stop)
            cand = []
            for ai, a in enumerate(DEFAULT_ACTIONS):
                if a.tool == "stop": continue
                c = tools[a.tool].base_cost + tools[a.tool].per_doc_cost * a.k
                cand.append((c, ai, a))
            c, ai, a = min(cand, key=lambda t: t[0])
            best = ((0, -c), ai, float(c), False)

        traces.append({
            "qid": ex.get("qid"),
            "question": q,
            "label_action_idx": int(best[1]),
            "budget": float(args.budget),
            "chosen_cost": float(best[2]),
            "teacher_hit": bool(best[3]),
        })

    write_jsonl(args.out, traces)
    print(f"Wrote {len(traces)} traces -> {args.out}")

if __name__ == "__main__":
    main()
'@ | Set-Content -Encoding UTF8 "scripts\generate_traces_kilt.py"

# Simple eval (hit@k) + router
@'
import argparse, csv
from collections import Counter
from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from toolrouter.config import QdrantConfig
from toolrouter.io import read_jsonl
from toolrouter.tools.bm25 import load_bm25, BM25Tool
from toolrouter.tools.qdrant_dense import QdrantDenseTool
from toolrouter.router.model import load_router
from toolrouter.router.actions import DEFAULT_ACTIONS

def wiki_id_from_doc_id(doc_id: str) -> str:
    s = str(doc_id)
    return s.split(":", 1)[0] if ":" in s else s

def hit_rate(tool, qa, k: int):
    hits = 0
    costs = 0.0
    for ex in qa:
        gold = set(ex.get("gold_wikipedia_ids") or [])
        res = tool.retrieve(ex["question"], k)
        costs += float(res.cost)
        hit = any(wiki_id_from_doc_id(d.doc_id) in gold for d in res.docs)
        hits += 1 if hit else 0
    return hits/len(qa), costs/len(qa)

def main():
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

    bm25 = BM25Tool(load_bm25(args.bm25))
    cfg = QdrantConfig()
    client = QdrantClient(url=cfg.url, api_key=cfg.api_key)
    embedder = SentenceTransformer(args.dense_model)
    dense = QdrantDenseTool(client=client, collection=args.qdrant_collection, embedder=embedder)

    router = load_router(args.router)

    rows = []

    for k in (5, 20):
        hr, ac = hit_rate(bm25, qa, k)
        rows.append(("bm25", k, hr, ac))
        hr, ac = hit_rate(dense, qa, k)
        rows.append(("dense", k, hr, ac))

    # Router (budget-enforced)
    hits = 0
    total_cost = 0.0
    use = Counter()

    # precompute cheapest non-stop action as fallback
    tools = {"bm25": bm25, "dense": dense}
    cheapest = None
    for ai, a in enumerate(DEFAULT_ACTIONS):
        if a.tool == "stop": continue
        c = tools[a.tool].base_cost + tools[a.tool].per_doc_cost * a.k
        if cheapest is None or c < cheapest[0]:
            cheapest = (c, a)

    for ex in tqdm(qa, desc="Eval router"):
        gold = set(ex.get("gold_wikipedia_ids") or [])
        a = router.predict_action(ex["question"])

        if a.tool == "stop":
            use["stop"] += 1
            continue

        res = tools[a.tool].retrieve(ex["question"], a.k)
        if float(res.cost) > args.budget:
            # fallback to cheapest allowed
            a = cheapest[1]
            res = tools[a.tool].retrieve(ex["question"], a.k)

        use[a.tool] += 1
        total_cost += float(res.cost)
        hit = any(wiki_id_from_doc_id(d.doc_id) in gold for d in res.docs)
        hits += 1 if hit else 0

    router_hr = hits / len(qa)
    router_ac = total_cost / len(qa)
    rows.append(("router", -1, router_hr, router_ac))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system","k","hit_rate","avg_cost"])
        for r in rows:
            w.writerow(r)

    print("=== KILT hit@k (page-level) ===")
    for system,k,hr,ac in rows:
        if system != "router":
            print(f"{system}@{k}: hit_rate={hr:.3f} avg_cost={ac:.3f}")
    print(f"router(budget={args.budget}): hit_rate={router_hr:.3f} avg_cost={router_ac:.3f} usage={dict(use)}")
    print(f"Wrote CSV: {args.out_csv}")

if __name__ == "__main__":
    main()
'@ | Set-Content -Encoding UTF8 "scripts\eval_kilt.py"

# --- Run pipeline ---
# (usa le tue env QDRANT_URL e QDRANT_API_KEY giÃ  settate)
Run-Step "Prepare KILT QA + needed ids" $Py "scripts\prepare_kilt_qa.py" "--task" "nq" "--split" "validation" "--max_examples" "5000" "--out" "data/kilt_qa.jsonl" "--needed_ids" "data/needed_wiki_ids.txt"
Run-Step "Build KILT Wikipedia subset corpus" $Py "scripts\build_kilt_corpus_subset.py" "--config" "2019-08-01" "--needed_ids" "data/needed_wiki_ids.txt" "--out" "data/kilt_corpus.jsonl" "--neg_pages" "5000" "--max_pages" "20000" "--max_passages" "200000"
Run-Step "Build BM25" $Py "scripts\build_bm25.py" "--corpus" "data/kilt_corpus.jsonl" "--out" "artifacts/bm25"
Run-Step "Ingest Qdrant" $Py "scripts\ingest_qdrant.py" "--corpus" "data/kilt_corpus.jsonl" "--collection" "wiki_mini" "--recreate"
Run-Step "Generate KILT traces" $Py "scripts\generate_traces_kilt.py" "--qa" "data/kilt_qa.jsonl" "--bm25" "artifacts/bm25" "--qdrant_collection" "wiki_mini" "--out" "artifacts\traces_kilt.jsonl" "--budget" "1.0"
Run-Step "Train router" $Py "scripts\train_router.py" "--traces" "artifacts\traces_kilt.jsonl" "--out" "artifacts\router"
Run-Step "Eval KILT" $Py "scripts\eval_kilt.py" "--qa" "data/kilt_qa.jsonl" "--bm25" "artifacts/bm25" "--qdrant_collection" "wiki_mini" "--router" "artifacts\router" "--budget" "1.0" "--out_csv" "results\kilt_metrics.csv"

"`nALL DONE. Log: $Log" | Tee-Object -FilePath $Log -Append | Out-Null
