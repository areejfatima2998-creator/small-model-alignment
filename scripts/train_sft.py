"""Supervised fine-tuning with response-only loss (no HF Trainer).

fp32 master weights + fp16 autocast + GradScaler, gradient accumulation,
warmup-cosine LR, gradient clipping, periodic validation, best/final checkpoints.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import build_sft_batch
from alignment.data import batches, load_examples, save_json
from alignment.losses import batch_logprobs
from alignment.train_utils import MetricsLogger, param_groups, save_checkpoint, set_seed, warmup_cosine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dataset", default="ultrachat")
    p.add_argument("--output_dir", default="outputs/sft")
    p.add_argument("--n_train", type=int, default=30000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=150, help="optimizer steps")
    p.add_argument("--log_every", type=int, default=10, help="optimizer steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, tokenizer, examples, args):
    model.eval()
    total_logp, total_tokens = 0.0, 0
    for chunk in batches(examples, args.eval_batch_size):
        batch = build_sft_batch(
            tokenizer, [e["prompt"] for e in chunk], [e["response"] for e in chunk], args.max_length
        )
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            logps, n_tokens = batch_logprobs(model, batch)
        total_logp += logps.sum().item()
        total_tokens += n_tokens.sum().item()
    model.train()
    return -total_logp / max(1, total_tokens)


def main():
    args = parse_args()
    args.amp = args.device.startswith("cuda")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = MetricsLogger(os.path.join(args.output_dir, "metrics.jsonl"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train = load_examples(args.dataset, "train", args.n_train, args.seed, tokenizer, args.max_prompt_length)
    val = load_examples(args.dataset, "val", args.n_val, args.seed, tokenizer, args.max_prompt_length)
    save_json({"n_train": len(train), "n_val": len(val), "args": vars(args)}, os.path.join(args.output_dir, "run_config.json"))

    # Trainable weights stay fp32; autocast runs the matmuls in fp16.
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    micro_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = math.ceil(micro_per_epoch * args.epochs / args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: warmup_cosine(s, total_steps, warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    logger.log(step=0, split="val", val_loss=evaluate(model, tokenizer, val, args))
    best_val = float("inf")

    step, micro, running_loss, running_n = 0, 0, 0.0, 0
    progress = tqdm(total=total_steps, desc="sft")
    for epoch in range(args.epochs):
        for chunk in batches(train, args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch = build_sft_batch(
                tokenizer, [e["prompt"] for e in chunk], [e["response"] for e in chunk], args.max_length
            )
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                logps, n_tokens = batch_logprobs(model, batch)
            # Token-mean loss per micro-batch, averaged over the accumulation window.
            loss = -logps.sum() / n_tokens.sum().clamp_min(1)
            scaler.scale(loss / args.grad_accum).backward()
            running_loss += loss.item()
            running_n += 1
            micro += 1
            if micro % args.grad_accum != 0 and micro != micro_per_epoch * args.epochs:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            progress.update(1)

            if step % args.log_every == 0:
                logger.log(
                    step=step, split="train", train_loss=running_loss / running_n,
                    lr=scheduler.get_last_lr()[0], grad_norm=float(grad_norm),
                )
                running_loss, running_n = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                val_loss = evaluate(model, tokenizer, val, args)
                logger.log(step=step, split="val", val_loss=val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best"),
                                    extra={"step": step, "val_loss": val_loss})
                    logger.log(step=step, event="saved_best", val_loss=val_loss)

    progress.close()
    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final"), extra={"step": step})
    logger.log(step=step, event="done", best_val_loss=best_val)


if __name__ == "__main__":
    main()
