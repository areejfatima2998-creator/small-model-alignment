"""From-scratch DPO (no TRL). Policy on one GPU, frozen fp16 reference on the other.

Same implementation for both tracks: ``--dataset hh`` or ``--dataset ultrafeedback``.
Best checkpoint is chosen by validation reward accuracy (strict r_w > r_l).
"""

from __future__ import annotations

import argparse
import math
import os

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import build_dpo_batch
from alignment.data import batches, load_examples, save_json
from alignment.losses import dpo_forward, dpo_rewards
from alignment.train_utils import MetricsLogger, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", required=True, help="SFT best checkpoint (policy init and reference)")
    p.add_argument("--dataset", choices=["hh", "ultrafeedback"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_train", type=int, default=15000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--optimizer", choices=["rmsprop", "adamw"], default="rmsprop")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps")
    p.add_argument("--log_every", type=int, default=5, help="optimizer steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy_device", default="cuda:0")
    p.add_argument("--reference_device", default="cuda:1")
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()


def make_batch(tokenizer, chunk, args):
    return build_dpo_batch(
        tokenizer,
        [e["prompt"] for e in chunk],
        [e["chosen"] for e in chunk],
        [e["rejected"] for e in chunk],
        args.max_length,
    )


@torch.no_grad()
def evaluate(policy, reference, tokenizer, examples, args):
    policy.eval()
    losses, chosen_rewards, rejected_rewards = [], [], []
    for chunk in batches(examples, args.eval_batch_size):
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            loss, logps = dpo_forward(policy, reference, make_batch(tokenizer, chunk, args), args.beta)
        rw, rl = dpo_rewards(*logps, args.beta)
        losses.append(loss.item() * len(chunk))
        chosen_rewards.append(rw.cpu())
        rejected_rewards.append(rl.cpu())
    policy.train()
    rw, rl = torch.cat(chosen_rewards), torch.cat(rejected_rewards)
    return {
        "val_loss": sum(losses) / len(examples),
        "val_reward_acc": (rw > rl).float().mean().item(),
        "val_reward_margin": (rw - rl).mean().item(),
        "val_chosen_reward": rw.mean().item(),
        "val_rejected_reward": rl.mean().item(),
    }


def main():
    args = parse_args()
    args.amp = args.policy_device.startswith("cuda")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = MetricsLogger(os.path.join(args.output_dir, "metrics.jsonl"))

    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    train = load_examples(args.dataset, "train", args.n_train, args.seed, tokenizer, args.max_prompt_length)
    val = load_examples(args.dataset, "val", args.n_val, args.seed, tokenizer, args.max_prompt_length)
    save_json({"n_train": len(train), "n_val": len(val), "args": vars(args)}, os.path.join(args.output_dir, "run_config.json"))

    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32).to(args.policy_device)
    policy.config.use_cache = False
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.train()

    ref_dtype = torch.float16 if args.reference_device.startswith("cuda") else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=ref_dtype).to(args.reference_device)
    reference.eval()
    reference.requires_grad_(False)

    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(policy.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    micro_total = math.ceil(len(train) / args.batch_size) * args.epochs
    total_steps = math.ceil(micro_total / args.grad_accum)

    metrics = evaluate(policy, reference, tokenizer, val, args)
    logger.log(step=0, split="val", **metrics)
    best_acc = -1.0

    step, micro = 0, 0
    window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    progress = tqdm(total=total_steps, desc=f"dpo-{args.dataset}")
    for epoch in range(args.epochs):
        for chunk in batches(train, args.batch_size, shuffle=True, seed=args.seed + epoch):
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                loss, logps = dpo_forward(policy, reference, make_batch(tokenizer, chunk, args), args.beta)
            scaler.scale(loss / args.grad_accum).backward()

            rw, rl = dpo_rewards(*logps, args.beta)
            window["loss"] += loss.item()
            window["acc"] += (rw > rl).float().mean().item()
            window["margin"] += (rw - rl).mean().item()
            window["n"] += 1
            micro += 1
            if micro % args.grad_accum != 0 and micro != micro_total:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)

            if step % args.log_every == 0:
                n = window["n"]
                logger.log(
                    step=step, split="train", train_loss=window["loss"] / n,
                    train_reward_acc=window["acc"] / n, train_reward_margin=window["margin"] / n,
                    grad_norm=float(grad_norm),
                )
                window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}

            if step % args.eval_every == 0 or step == total_steps:
                metrics = evaluate(policy, reference, tokenizer, val, args)
                logger.log(step=step, split="val", **metrics)
                if metrics["val_reward_acc"] > best_acc:
                    best_acc = metrics["val_reward_acc"]
                    save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "best"),
                                    extra={"step": step, **metrics, "reference": args.sft_path})
                    logger.log(step=step, event="saved_best", val_reward_acc=best_acc)

    progress.close()
    save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "final"),
                    extra={"step": step, "reference": args.sft_path})
    logger.log(step=step, event="done", best_val_reward_acc=best_acc)


if __name__ == "__main__":
    main()
