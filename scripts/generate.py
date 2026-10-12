"""Generate responses on a fixed eval set and save them as JSON.

``--prompt_format alpaca`` uses the SFT/DPO template; ``chat`` uses the
tokenizer's chat template (for the SmolLM2 instruct model in the LoRA task).
Use identical decoding settings for every model in a comparison.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.data import format_prompt, load_json, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--adapter_path", default=None, help="optional PEFT/LoRA adapter")
    p.add_argument("--eval_set", required=True)
    p.add_argument("--generator", required=True, help="name stored in the output (e.g. qwen2.5-0.5b-sft)")
    p.add_argument("--output", required=True)
    p.add_argument("--prompt_format", choices=["alpaca", "chat"], default="alpaca")
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--do_sample", action="store_true", help="default: greedy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="sanity check on the first N prompts")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_prompt(tokenizer, instruction, prompt_format):
    if prompt_format == "alpaca":
        return format_prompt(instruction)
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rows = load_json(args.eval_set)[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"  # decoder-only batching: pad on the left so new tokens line up
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(args.device)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.eval()

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.do_sample:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    results = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc=args.generator):
        chunk = rows[start : start + args.batch_size]
        prompts = [build_prompt(tokenizer, r["instruction"], args.prompt_format) for r in chunk]
        # Templates already contain any special tokens they need.
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(args.device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for row, ids in zip(chunk, new_tokens):
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            # Base models may keep going into a new "### Instruction:" block; cut it.
            text = text.split("### Instruction:")[0].strip()
            record = {
                "id": row["id"],
                "instruction": row["instruction"],
                "output": text,
                "generator": args.generator,
                "dataset": row.get("dataset", ""),
            }
            for key in ("category", "expected_behavior"):
                if key in row:
                    record[key] = row[key]
            results.append(record)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_json(results, args.output)
    settings = {k: v for k, v in vars(args).items() if k not in ("output",)}
    with open(args.output.replace(".json", "") + "_settings.json", "w") as f:
        json.dump(settings, f, indent=1)
    print(f"wrote {len(results)} generations to {args.output}")
    for r in results[:3]:
        print("\n### " + r["instruction"][:200] + "\n--> " + r["output"][:500])


if __name__ == "__main__":
    main()
