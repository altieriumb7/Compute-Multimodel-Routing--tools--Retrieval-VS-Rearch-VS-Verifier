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
