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
        # include title in the chunk so itâ€™s retrievable / auditable
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
