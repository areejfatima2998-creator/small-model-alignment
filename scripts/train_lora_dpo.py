"""Task 5: TRL DPOTrainer + PEFT LoRA persona tuning on SmolLM2-360M-Instruct.

This is the only part of the assignment where TRL is allowed.

LoRA choices (justify in the report):
- r=16, alpha=32: 220 short style pairs need only a low-rank update; alpha/r=2 is the usual scale.
- targets = all attention + MLP projections: style lives in word choice (MLP) as much as attention.
- lr 5e-5: LoRA adapters start at zero, so they tolerate a much higher lr than full DPO (1e-6).
- 3 epochs, effective batch 8: ~75 optimizer steps on 200 training pairs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--output_dir", default="outputs/lora_dpo")
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def to_conversational(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def filter_kwargs(cls, kwargs):
    """TRL renames arguments between versions; keep only the ones this version accepts."""
    accepted = inspect.signature(cls.__init__).parameters
    dropped = sorted(k for k in kwargs if k not in accepted)
    if dropped:
        print(f"[lora-dpo] {cls.__name__} ignoring unsupported args: {dropped}")
    return {k: v for k, v in kwargs.items() if k in accepted}


def main():
    args = parse_args()
    from datasets import Dataset

    with open(args.data_path, encoding="utf-8") as f:
        rows = json.load(f)
    random.Random(args.seed).shuffle(rows)
    eval_rows, train_rows = rows[: args.n_eval], rows[args.n_eval :]
    train_ds = Dataset.from_list([to_conversational(r) for r in train_rows])
    eval_ds = Dataset.from_list([to_conversational(r) for r in eval_rows])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = DPOConfig(**filter_kwargs(DPOConfig, dict(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )))

    trainer_kwargs = dict(model=model, ref_model=None, args=config, train_dataset=train_ds,
                          eval_dataset=eval_ds, peft_config=peft_config)
    params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    trainer.model.print_trainable_parameters()

    trainer.train()
    adapter_dir = os.path.join(args.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with open(os.path.join(args.output_dir, "metrics.jsonl"), "w") as f:
        for record in trainer.state.log_history:
            f.write(json.dumps(record) + "\n")
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "n_train": len(train_rows), "n_eval": len(eval_rows)}, f, indent=1)
    print(f"saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
