import argparse, random
from toolrouter.io import write_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--n_docs", type=int, default=200)
    ap.add_argument("--n_qa", type=int, default=100)
    args = ap.parse_args()

    random.seed(7)

    docs = []
    facts = []
    for i in range(args.n_docs):
        city = f"City{i}"
        country = f"Country{i%10}"
        pop = 100000 + i
        text = f"{city} is a city in {country}. Its population is {pop}."
        docs.append({"doc_id": f"d{i}", "title": city, "text": text})
        facts.append((city, country, pop))

    qa = []
    for j in range(args.n_qa):
        city, country, pop = random.choice(facts)
        if j % 2 == 0:
            q = f"Which country is {city} in?"
            a = country
        else:
            q = f"What is the population of {city}?"
            a = str(pop)
        qa.append({"qid": f"q{j}", "question": q, "answer": a})

    write_jsonl(f"{args.out_dir}/corpus.jsonl", docs)
    write_jsonl(f"{args.out_dir}/qa.jsonl", qa)
    print(f"Wrote {len(docs)} docs and {len(qa)} QA to {args.out_dir}/")

if __name__ == "__main__":
    main()
