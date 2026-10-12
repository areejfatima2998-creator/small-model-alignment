"""Write a copy of a pairs file with positions A and B exchanged.

Judging both orders measures position bias and lets the analysis keep only
order-consistent verdicts (a response must win in both positions to count).
"""

from __future__ import annotations

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    n = 0
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            pair = json.loads(line)
            pair["response_a"], pair["response_b"] = pair["response_b"], pair["response_a"]
            hidden = pair["hidden"]
            hidden["model_a"], hidden["model_b"] = hidden["model_b"], hidden["model_a"]
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} swapped pairs to {args.output}")


if __name__ == "__main__":
    main()
