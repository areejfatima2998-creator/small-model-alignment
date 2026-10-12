"""Recover judgments from a Kaggle Benchmarks task log into per-comparison files.

Writes <name>_judged.jsonl, <name>_judged_failures.jsonl and
<name>_judged_summary.json for every comparison found in the log, appending to
any judgments already on disk (so resumed runs accumulate).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_pairwise_kaggle import read_jsonl, summarize  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pairs_dir", default=None, help="pairs files (<name>_pairs.jsonl) for total counts")
    args = p.parse_args()

    with open(args.log, encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = text.replace("\\t", "\t").replace('\\"', '"')
    os.makedirs(args.out_dir, exist_ok=True)

    found = {}
    for kind, name, payload in re.findall(r"(JUDGED|FAILED)\t([\w\-]+)\t(\{.*?\})(?=\\n|\n|\"|$)", text):
        try:
            found.setdefault(name, {"JUDGED": [], "FAILED": []})[kind].append(json.loads(payload))
        except json.JSONDecodeError:
            continue

    for name, groups in found.items():
        judged_path = os.path.join(args.out_dir, f"{name}_judged.jsonl")
        failure_path = os.path.join(args.out_dir, f"{name}_judged_failures.jsonl")
        existing = {r["pair_id"]: r for r in read_jsonl(judged_path)}
        for r in groups["JUDGED"]:
            existing[r["pair_id"]] = r
        with open(judged_path, "w", encoding="utf-8") as f:
            for r in existing.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        failures = read_jsonl(failure_path) + groups["FAILED"]
        with open(failure_path, "w", encoding="utf-8") as f:
            for r in failures:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        total = len(existing)
        if args.pairs_dir:
            total = len(read_jsonl(os.path.join(args.pairs_dir, f"{name}_pairs.jsonl"))) or total
        judged = list(existing.values())
        summary = summarize(judged, failures, total)
        summary.update(
            judge_model=judged[0]["judge_model"] if judged else None,
            rubric=judged[0]["rubric"] if judged else None,
        )
        with open(os.path.join(args.out_dir, f"{name}_judged_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(f"{name}: {len(judged)} judged, {len(failures)} failure records -> win {summary['win']} / "
              f"tie {summary['tie']} / loss {summary['loss']} / unknown {summary['unknown']}")


if __name__ == "__main__":
    main()
