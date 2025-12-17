import os
import orjson

QA = "data/kilt_qa.jsonl"
NEEDED = "data/needed_wiki_ids.txt"
CORPUS = "data/kilt_corpus.jsonl"

def file_info(p):
    if not os.path.exists(p):
        print(f"{p}: MISSING")
        return
    sz = os.path.getsize(p)/1e6
    print(f"{p}: {sz:.2f} MB")

def read_jsonl(path):
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)

def validate_jsonl(path):
    ok = 0
    bad = 0
    last_good = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                orjson.loads(line)
                ok += 1
                last_good = f.tell()
            except Exception:
                bad += 1
                break
    return ok, bad, last_good

def wiki_id_from_doc_id(doc_id):
    s = str(doc_id)
    return s.split(":", 1)[0] if ":" in s else s

for p in [QA, NEEDED, CORPUS]:
    file_info(p)

if os.path.exists(CORPUS):
    ok, bad, last_good = validate_jsonl(CORPUS)
    print(f"\nCORPUS valid_lines={ok} bad={bad} last_good_offset={last_good}")

    # build set of wikipedia_ids in corpus (cheap memory: <= max_pages)
    corpus_wids = set()
    with open(CORPUS, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ex = orjson.loads(line)
            except Exception:
                break
            doc_id = ex.get("doc_id")
            if doc_id is not None:
                corpus_wids.add(wiki_id_from_doc_id(doc_id))

    print(f"CORPUS unique_wikipedia_pages={len(corpus_wids)}")

    if os.path.exists(QA):
        qa = list(read_jsonl(QA))
        covered = 0
        for ex in qa:
            gold = set(ex.get("gold_wikipedia_ids") or [])
            if any(w in corpus_wids for w in gold):
                covered += 1
        print(f"QA examples={len(qa)} covered={covered} coverage={covered/len(qa):.3f}")
