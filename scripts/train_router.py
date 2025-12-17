import argparse
from toolrouter.io import read_jsonl
from toolrouter.router.model import train_router_model, save_router
from toolrouter.router.actions import DEFAULT_ACTIONS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.traces)
    texts = [r["question"] for r in rows]
    labels = [int(r["label_action_idx"]) for r in rows]
    model = train_router_model(texts, labels, actions=DEFAULT_ACTIONS)
    save_router(model, args.out)
    print(f"Saved router to {args.out} (n={len(rows)})")

if __name__ == "__main__":
    main()
