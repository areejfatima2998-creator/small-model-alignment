"""Create the fixed AlpacaEval subset: 300 prompts sampled with seed 42."""

from __future__ import annotations

import argparse
import json
import random

from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="eval_sets/alpaca_eval_300_seed42.json")
    args = p.parse_args()

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.n))
    subset = [
        {
            "id": f"alpaca-eval-{args.seed}-{i:04d}",
            "source_index": i,
            "instruction": rows[i]["instruction"],
            "dataset": "alpaca_eval",
            "source_subset": rows[i].get("dataset", ""),
        }
        for i in indices
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(subset)} prompts (of {len(rows)}) to {args.output}")


if __name__ == "__main__":
    main()
