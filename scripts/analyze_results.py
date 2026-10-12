"""Aggregate judged comparisons for the report: overall, per-category, and length effects.

Reads results/eval/<name>_judged.jsonl plus the generation files and writes
results/analysis.md (tables) and results/analysis.json.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

COMPARISONS = {
    # name: (candidate generation file, baseline generation file, label)
    "dpo_hh_vs_sft": ("dpo_hh.json", "sft_hh.json", "HH-RLHF DPO vs SFT (HH synthetic eval, safety rubric)"),
    "dpo_uf_vs_sft": ("dpo_uf.json", "sft_alpaca.json", "UltraFeedback DPO vs SFT (AlpacaEval-300, general rubric)"),
    "lora_vs_base_style": ("persona_lora.json", "persona_base.json", "LoRA-DPO vs SmolLM2-Instruct (persona-style rubric)"),
    "lora_vs_base_usefulness": ("persona_lora.json", "persona_base.json", "LoRA-DPO vs SmolLM2-Instruct (usefulness-only rubric)"),
}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rates(records):
    c = defaultdict(int)
    for r in records:
        c[r["outcome"]] += 1
    decided = c["win"] + c["loss"]
    return {
        "n": len(records), "win": c["win"], "tie": c["tie"], "loss": c["loss"], "unknown": c["unknown"],
        "win_rate_excl_ties": c["win"] / decided if decided else None,
        "win_rate_ties_half": (c["win"] + 0.5 * c["tie"]) / (decided + c["tie"]) if decided + c["tie"] else None,
    }


def fmt(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    gen_dir, eval_dir = os.path.join(args.results, "generations"), os.path.join(args.results, "eval")

    out, md = {}, ["# Results analysis (auto-generated)\n"]
    for name, (cand_file, base_file, label) in COMPARISONS.items():
        judged = read_jsonl(os.path.join(eval_dir, f"{name}_judged.jsonl"))
        if not judged:
            continue
        failures = read_jsonl(os.path.join(eval_dir, f"{name}_judged_failures.jsonl"))
        summary_path = os.path.join(eval_dir, f"{name}_judged_summary.json")
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
        with open(os.path.join(gen_dir, cand_file), encoding="utf-8") as f:
            cand = {r["id"]: r for r in json.load(f)}
        with open(os.path.join(gen_dir, base_file), encoding="utf-8") as f:
            base = {r["id"]: r for r in json.load(f)}

        overall = rates(judged)
        by_cat = defaultdict(list)
        for r in judged:
            by_cat[r.get("category") or "all"].append(r)
        cats = {k: rates(v) for k, v in sorted(by_cat.items())}

        # length effect: does the judge prefer the longer response?
        longer_wins, decisive = 0, 0
        for r in judged:
            if r["outcome"] not in ("win", "loss"):
                continue
            c_len, b_len = len(cand[r["pair_id"]]["output"]), len(base[r["pair_id"]]["output"])
            if c_len == b_len:
                continue
            decisive += 1
            winner_is_cand = r["outcome"] == "win"
            longer_wins += (c_len > b_len) == winner_is_cand
        mean_len = {
            "candidate_chars": sum(len(cand[r["pair_id"]]["output"]) for r in judged) / len(judged),
            "baseline_chars": sum(len(base[r["pair_id"]]["output"]) for r in judged) / len(judged),
        }

        # order-swapped re-judgment: keep a win/loss only if it holds in both orders
        swapped = {r["pair_id"]: r for r in read_jsonl(os.path.join(eval_dir, f"{name}_swapped_judged.jsonl"))}
        debiased = None
        if swapped:
            combined, agree = [], 0
            for r in judged:
                s2 = swapped.get(r["pair_id"])
                if s2 is None:
                    continue
                same = r["outcome"] == s2["outcome"]
                agree += same
                outcome = r["outcome"] if same and r["outcome"] in ("win", "loss", "tie") else "tie"
                combined.append({**r, "outcome": outcome})
            debiased = {
                "overall": rates(combined),
                "consistency": agree / len(combined) if combined else None,
                "by_category": {k: rates([c for c in combined if (c.get("category") or "all") == k]) for k in cats},
                "swapped_overall": rates(list(swapped.values())),
            }

        out[name] = {
            "label": label, "overall": overall, "by_category": cats, "order_debiased": debiased,
            "unresolved_failures": summary.get("unresolved_failures", 0), "failure_records": len(failures),
            "position_A_preference_rate": summary.get("position_A_preference_rate"),
            "longer_response_wins_rate": longer_wins / decisive if decisive else None,
            "mean_length": mean_len,
            "judge_model": judged[0].get("judge_model"),
        }

        md.append(f"## {label}\n")
        md.append("| Subset | N | Wins | Ties | Losses | Unknown | Win rate (excl. ties) | Win rate (ties = ½) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for sub, r in [("**all**", overall)] + [(k, v) for k, v in cats.items() if len(cats) > 1]:
            md.append(f"| {sub} | {r['n']} | {r['win']} | {r['tie']} | {r['loss']} | {r['unknown']} | "
                      f"{fmt(r['win_rate_excl_ties'])} | {fmt(r['win_rate_ties_half'])} |")
        md.append("")
        md.append(f"- Judge: `{out[name]['judge_model']}`. Unresolved failures: {out[name]['unresolved_failures']}. "
                  f"Position-A preference: {fmt(out[name]['position_A_preference_rate'])}. "
                  f"Longer response wins: {fmt(out[name]['longer_response_wins_rate'])} of decisive pairs.")
        md.append(f"- Mean response length (chars): candidate {mean_len['candidate_chars']:.0f}, "
                  f"baseline {mean_len['baseline_chars']:.0f}.\n")
        if debiased:
            d = debiased
            md.append(f"**Order-swapped re-judgment.** Verdict agreement across the two orders: {fmt(d['consistency'])}. "
                      f"Swapped order alone: {d['swapped_overall']['win']}W / {d['swapped_overall']['tie']}T / "
                      f"{d['swapped_overall']['loss']}L. A win or loss counts only when it holds in both orders "
                      "(otherwise it is a tie):\n")
            md.append("| Subset | N | Wins | Ties | Losses | Win rate (excl. ties) | Win rate (ties = ½) |")
            md.append("|---|---|---|---|---|---|---|")
            for sub, r in [("**all**", d["overall"])] + [(k, v) for k, v in d["by_category"].items() if len(cats) > 1]:
                md.append(f"| {sub} | {r['n']} | {r['win']} | {r['tie']} | {r['loss']} | "
                          f"{fmt(r['win_rate_excl_ties'])} | {fmt(r['win_rate_ties_half'])} |")
            md.append("")

    with open(os.path.join(args.results, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(args.results, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
