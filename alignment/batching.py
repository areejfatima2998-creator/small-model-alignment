"""Tokenization and batching for response-only language-model training.

Every training row is ``prompt + response + EOS``. Labels copy ``input_ids`` on
response positions and are ``-100`` on prompt and padding positions, so the
loss only scores the response. Masks are built from token *positions*, never by
comparing token values: Qwen uses the same id for pad and EOS, so a value-based
mask would silently drop the EOS label.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def _encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))


def resolve_pad_id(tokenizer):
    """Padding id to use; padded positions are masked, so any valid id works."""
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id
    return 0


def tokenize_example(tokenizer, prompt, response, max_length, max_prompt_length=None):
    """Return (input_ids, labels) lists for one prompt/response pair.

    EOS is appended to the response *before* truncation, so a response cut at
    ``max_length`` loses its EOS (as the assignment specifies). If
    ``max_prompt_length`` is given, the prompt keeps its last tokens.
    """
    prompt_ids = _encode(tokenizer, prompt)
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:] if max_prompt_length > 0 else []
    response_ids = _encode(tokenizer, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [eos_id]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def collate(rows, pad_id):
    """Right-pad (input_ids, labels) rows to the longest row in the batch."""
    width = max(1, max(len(ids) for ids, _ in rows))
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in rows:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_sft_batch(tokenizer, prompts, responses, max_length, max_prompt_length=None):
    """Batch of input_ids / attention_mask / labels, all shaped [B, T]."""
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    rows = [
        tokenize_example(tokenizer, p, r, max_length, max_prompt_length)
        for p, r in zip(prompts, responses)
    ]
    return collate(rows, resolve_pad_id(tokenizer))


def build_dpo_batch(
    tokenizer, prompts, chosen_responses, rejected_responses, max_length, max_prompt_length=None
):
    """Two LM batches with the identical prompt formatting for each side."""
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen and rejected must have the same length")
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length, max_prompt_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length, max_prompt_length),
    }
