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
