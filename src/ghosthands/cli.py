from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="GhostHands reproducible research tools")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--data", type=Path, default=Path("data/synthetic"))
    gen.add_argument("--train-count", type=int, default=240)
    gen.add_argument("--val-count", type=int, default=40)
    tr = sub.add_parser("train")
    tr.add_argument("--data", type=Path, default=Path("data/synthetic"))
    tr.add_argument("--output", type=Path, default=Path("runs/default"))
    tr.add_argument("--model", choices=["all", "absolute", "relative", "recurrent"], default="all")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--source", choices=["synthetic_teacher", "human", "all"], default="synthetic_teacher")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ev.add_argument("--checkpoints", type=Path, default=Path("runs/default"))
    ev.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    ev.add_argument("--count", type=int, default=100)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "generate":
        from .data import generate_episode, save_episode
        if not 2 <= args.train_count <= 10000 or not 1 <= args.val_count <= 10000:
            parser.error("Counts must preserve disjoint seed ranges")
        for split, count, base in [("train", args.train_count, 0), ("val", args.val_count, 10000)]:
            for seed in range(base, base + count):
                save_episode(generate_episode(seed, split), args.data)
            print(f"Saved {count} {split} synthetic episodes")
    elif args.command == "train":
        from .training import train
        for kind in ["absolute", "relative", "recurrent"] if args.model == "all" else [args.model]:
            train(args.data, args.output, kind, args.epochs, args.seed, args.source)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(args.data, args.checkpoints, args.output, args.count)
    else:
        import uvicorn
        uvicorn.run("ghosthands.server:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
