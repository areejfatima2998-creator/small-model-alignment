"""Shared training helpers: seeding, logging, schedules, checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MetricsLogger:
    """Append-only JSONL so partial progress survives a Kaggle disconnect."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.start = time.time()

    def log(self, **record):
        record["elapsed_s"] = round(time.time() - self.start, 1)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def warmup_cosine(step, total_steps, warmup_steps):
    """LR multiplier: linear warmup to 1, then cosine decay to 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def param_groups(model, weight_decay):
    """No weight decay on biases and norm weights (1-D tensors)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(model, tokenizer, path, dtype=torch.float16, extra=None):
    """Save weights (cast to ``dtype`` to halve disk use) plus tokenizer."""
    os.makedirs(path, exist_ok=True)
    state = {k: v.detach().to(dtype) if v.is_floating_point() else v for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if extra is not None:
        with open(os.path.join(path, "checkpoint_info.json"), "w") as f:
            json.dump(extra, f, indent=1)
