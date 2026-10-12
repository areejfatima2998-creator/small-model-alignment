"""Plot training curves from metrics.jsonl files written by the training scripts."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def series(records, key, split=None):
    pts = [(r["step"], r[key]) for r in records if key in r and (split is None or r.get("split") == split)]
    return [s for s, _ in pts], [v for _, v in pts]


def smooth(values, k):
    if k <= 1:
        return values
    out = []
    for i in range(len(values)):
        window = values[max(0, i - k + 1) : i + 1]
        out.append(sum(window) / len(window))
    return out


def best_step(records):
    steps = [r["step"] for r in records if r.get("event") == "saved_best"]
    return steps[-1] if steps else None


def plot_sft(path, out, k):
    rec = read(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    x, y = series(rec, "train_loss", "train")
    ax.plot(x, y, alpha=0.3, color="C0")
    ax.plot(x, smooth(y, k), color="C0", label=f"train (moving avg, {k} logs)")
    x, y = series(rec, "val_loss", "val")
    ax.plot(x, y, "o-", color="C1", label="validation")
    b = best_step(rec)
    if b is not None:
        ax.axvline(b, ls="--", color="gray", label=f"best ckpt (step {b})")
    ax.set(xlabel="optimizer step", ylabel="response-token NLL", title="SFT loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_dpo(paths, labels, out, k):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (path, label) in enumerate(zip(paths, labels)):
        rec = read(path)
        color = f"C{i}"
        for ax, train_key, val_key, ylabel in (
            (axes[0], "train_loss", "val_loss", "DPO loss"),
            (axes[1], "train_reward_acc", "val_reward_acc", "reward accuracy"),
            (axes[2], "train_reward_margin", "val_reward_margin", "reward margin"),
        ):
            x, y = series(rec, train_key, "train")
            ax.plot(x, smooth(y, k), color=color, alpha=0.5, label=f"{label} train (avg {k})")
            x, y = series(rec, val_key, "val")
            ax.plot(x, y, "o-", color=color, label=f"{label} val")
            ax.set(xlabel="optimizer step", ylabel=ylabel)
        b = best_step(rec)
        if b is not None:
            axes[1].axvline(b, ls="--", color=color, alpha=0.6)
    axes[0].axhline(0.6931, ls=":", color="gray", label="log 2 (init)")
    axes[1].axhline(0.5, ls=":", color="gray")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("DPO training (dashed: best checkpoint by val reward accuracy)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", help="SFT metrics.jsonl")
    p.add_argument("--dpo", nargs="*", default=[], help="DPO metrics.jsonl files")
    p.add_argument("--dpo_labels", nargs="*", default=[])
    p.add_argument("--out_dir", default="outputs/plots")
    p.add_argument("--smooth", type=int, default=5)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.sft:
        plot_sft(args.sft, os.path.join(args.out_dir, "sft_loss.png"), args.smooth)
    if args.dpo:
        labels = args.dpo_labels or [os.path.basename(os.path.dirname(p)) for p in args.dpo]
        plot_dpo(args.dpo, labels, os.path.join(args.out_dir, "dpo_metrics.png"), args.smooth)
    print(f"plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
