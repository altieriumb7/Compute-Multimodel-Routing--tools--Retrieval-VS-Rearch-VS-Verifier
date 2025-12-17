import argparse
from toolrouter.io import read_jsonl
from toolrouter.tools.bm25 import build_bm25, save_bm25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="JSONL corpus with doc_id/title/text")
    ap.add_argument("--out", required=True, help="Output directory for BM25 pickle")
    args = ap.parse_args()

    docs = read_jsonl(args.corpus)
    store = build_bm25(docs)
    save_bm25(store, args.out)
    print(f"Saved BM25 to {args.out} (docs={len(docs)})")

if __name__ == "__main__":
    main()
