"""Build blinded pairwise comparison files (candidate vs baseline).

Each pair randomly places the two outputs in positions A/B. The mapping from
position to model lives only under ``hidden``, which the judge never sees.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from alignment.data import load_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="generation JSON of the baseline (e.g. SFT)")
    p.add_argument("--candidate", required=True, help="generation JSON of the candidate (e.g. DPO)")
    p.add_argument("--output", required=True, help="pairs JSONL")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    baseline = {r["id"]: r for r in load_json(args.baseline)}
    candidate = {r["id"]: r for r in load_json(args.candidate)}
    common = [i for i in baseline if i in candidate]
    missing = len(baseline) - len(common)
    if missing:
        print(f"warning: {missing} baseline ids missing from candidate")

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_candidate_a = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for pair_id in common:
            b, c = baseline[pair_id], candidate[pair_id]
            assert b["instruction"] == c["instruction"], pair_id
            candidate_is_a = rng.random() < 0.5
            n_candidate_a += candidate_is_a
            first, second = (c, b) if candidate_is_a else (b, c)
            record = {
                "pair_id": pair_id,
                "instruction": b["instruction"],
                "response_a": first["output"],
                "response_b": second["output"],
                "hidden": {
                    "model_a": first["generator"],
                    "model_b": second["generator"],
                    "candidate": c["generator"],
                    "baseline": b["generator"],
                    "dataset": b.get("dataset", ""),
                    "category": b.get("category", ""),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(common)} pairs to {args.output} (candidate in position A: {n_candidate_a})")


if __name__ == "__main__":
    main()
