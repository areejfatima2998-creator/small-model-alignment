"""Package saved pair files into a self-contained Kaggle Benchmarks task.

Local model-proxy credentials only expose a few judge models, so the judge runs
on Kaggle's side instead:

    python scripts/make_judge_task.py --out judge_task.py \
        --pairs dpo_uf_vs_sft:general:outputs/eval/dpo_uf_vs_sft_pairs.jsonl ...
    kaggle b t push pa2-pairwise-judge -f judge_task.py
    kaggle b t run pa2-pairwise-judge -m gemini-3.6-flash --wait
    kaggle b t log pa2-pairwise-judge -m gemini-3.6-flash > judge.log
    python scripts/collect_judgments.py --log judge.log --out_dir outputs/eval

The task reuses the batching/retry/blinding logic from judge_pairwise_kaggle.py
and prints one tagged JSON line per judgment so results can be recovered from
the run log. ``--skip`` excludes pairs already judged (resume support).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))

TASK_TEMPLATE = '''

# ---------------------------------------------------------------- benchmark task
import kaggle_benchmarks as kbench

JOBS = json.loads(zlib.decompress(base64.b64decode("{payload}")).decode("utf-8"))
JUDGE_MODEL_LABEL = "{judge_label}"


@kbench.task(name="pa2-pairwise-judge", description="Blinded pairwise judging of SFT vs DPO and LoRA vs base outputs.")
def pa2_pairwise_judge(llm) -> dict:
    totals = {{}}
    for job in JOBS:
        name, rubric = job["name"], job["rubric"]

        def checkpoint(records, failures, name=name):
            for r in records:
                print("JUDGED\\t" + name + "\\t" + json.dumps(r, ensure_ascii=False), flush=True)
            for f in failures:
                print("FAILED\\t" + name + "\\t" + json.dumps(f, ensure_ascii=False), flush=True)
            t = totals.setdefault(name, {{"judged": 0, "failed": 0}})
            t["judged"] += len(records)
            t["failed"] += len(failures)

        judge_pairs(lambda text: str(llm.prompt(text)), job["pairs"], rubric, JUDGE_MODEL_LABEL,
                    {batch_size}, {sleep_seconds}, {max_retries}, checkpoint)
    print("TOTALS\\t" + json.dumps(totals), flush=True)
    return totals


pa2_pairwise_judge.run(kbench.llm)
'''


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", required=True, help="name:rubric:path entries")
    p.add_argument("--skip", nargs="*", default=[], help="judged JSONL files whose pair_ids to skip, as name:path")
    p.add_argument("--out", required=True)
    p.add_argument("--judge_label", default="google/gemini-3.6-flash")
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--sleep_seconds", type=float, default=5.0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="pairs per job (dry run)")
    args = p.parse_args()

    skip = {}
    for entry in args.skip:
        name, path = entry.split(":", 1)
        skip.setdefault(name, set()).update(r["pair_id"] for r in read_jsonl(path))

    jobs = []
    for entry in args.pairs:
        name, rubric, path = entry.split(":", 2)
        pairs = [pr for pr in read_jsonl(path) if pr["pair_id"] not in skip.get(name, set())][: args.limit]
        jobs.append({"name": name, "rubric": rubric, "pairs": pairs})
        print(f"{name}: {len(pairs)} pairs to judge with rubric {rubric!r}")

    with open(os.path.join(HERE, "judge_pairwise_kaggle.py"), encoding="utf-8") as f:
        judge_src = f.read()
    judge_src = judge_src.split('\nif __name__ == "__main__":')[0]  # drop the CLI entry point
    # kaggle_benchmarks inspects real return annotations; postponed (string) annotations break it.
    judge_src = judge_src.replace("from __future__ import annotations\n", "")

    payload = base64.b64encode(zlib.compress(json.dumps(jobs).encode("utf-8"), 9)).decode("ascii")
    task = TASK_TEMPLATE.format(
        payload=payload, judge_label=args.judge_label, batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds, max_retries=args.max_retries,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(judge_src + "\nimport base64\nimport zlib\n" + task)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
