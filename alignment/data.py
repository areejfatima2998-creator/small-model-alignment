"""Prompt template and dataset loading for SFT and DPO.

One template is used everywhere (SFT, DPO, generation) so the model always sees
the same format. It is plain text, which suits a base model better than ChatML.
"""

from __future__ import annotations

import json
import random

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def _clean(text):
    return (text or "").strip()


# ---------------------------------------------------------------- SFT (UltraChat)

def parse_ultrachat(row):
    """First user turn -> first assistant turn, or None if malformed."""
    messages = row.get("messages") or []
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    instruction, response = _clean(messages[0]["content"]), _clean(messages[1]["content"])
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response}


# ---------------------------------------------------------------- DPO: HH-RLHF

HUMAN, ASSISTANT = "\n\nHuman:", "\n\nAssistant:"


def _split_hh(text):
    """Split a single-turn HH transcript into (prompt, response); None if multi-turn."""
    if text.count(HUMAN) != 1 or text.count(ASSISTANT) != 1:
        return None
    before, response = text.split(ASSISTANT)
    if HUMAN not in before:
        return None
    prompt = before.split(HUMAN, 1)[1]
    return _clean(prompt), _clean(response)


def parse_hh(row):
    chosen, rejected = _split_hh(row["chosen"]), _split_hh(row["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    prompt, chosen_text, rejected_text = chosen[0], chosen[1], rejected[1]
    if not prompt or not chosen_text or not rejected_text or chosen_text == rejected_text:
        return None
    return {"instruction": prompt, "chosen": chosen_text, "rejected": rejected_text}


# ---------------------------------------------------------------- DPO: UltraFeedback

def _last_assistant(messages):
    for message in reversed(messages or []):
        if message["role"] == "assistant":
            return _clean(message["content"])
    return ""


def parse_ultrafeedback(row):
    """Single-turn binarized pair; drops ties and identical responses."""
    if len(row.get("chosen") or []) != 2 or len(row.get("rejected") or []) != 2:
        return None
    if row.get("score_chosen") is not None and row.get("score_chosen") == row.get("score_rejected"):
        return None
    prompt = _clean(row["prompt"])
    chosen, rejected = _last_assistant(row["chosen"]), _last_assistant(row["rejected"])
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {"instruction": prompt, "chosen": chosen, "rejected": rejected}


DATASETS = {
    # name: (hub id, config, train split, val split, parser)
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "test_sft", parse_ultrachat),
    "hh": ("Anthropic/hh-rlhf", None, "train", "test", parse_hh),
    "ultrafeedback": ("HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", "test_prefs", parse_ultrafeedback),
}


def load_examples(name, split, n, seed, tokenizer=None, max_prompt_length=None):
    """Shuffle a split with ``seed``, parse rows, filter long prompts, keep ``n``."""
    from datasets import load_dataset

    hub_id, config, train_split, val_split, parser = DATASETS[name]
    ds = load_dataset(hub_id, config, split=train_split if split == "train" else val_split)
    ds = ds.shuffle(seed=seed)

    out = []
    for row in ds:
        example = parser(row)
        if example is None:
            continue
        example["prompt"] = format_prompt(example["instruction"])
        if tokenizer is not None and max_prompt_length is not None:
            if len(tokenizer.encode(example["prompt"], add_special_tokens=False)) > max_prompt_length:
                continue
        out.append(example)
        if len(out) >= n:
            break
    if len(out) < n:
        print(f"[data] warning: only {len(out)} usable {name}/{split} examples (wanted {n})")
    return out


def batches(items, batch_size, shuffle=False, seed=0):
    order = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [items[i] for i in order[start : start + batch_size]]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
