"""Response-only log-probabilities, SFT loss, and the DPO objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sequence_logprobs(model, input_ids, attention_mask, labels):
    """Summed log p(label tokens) per row, plus the number of scored tokens.

    Position t's logits predict token t+1, so logits drop the last step and
    labels drop the first. Returns (sum_logps [B] float32, n_tokens [B]).
    """
    device = model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, T, V]
    logits = logits[:, :-1, :].float()  # [B, T-1, V]
    targets = labels[:, 1:]  # [B, T-1]
    mask = targets.ne(IGNORE_INDEX)

    safe_targets = targets.masked_fill(~mask, 0)
    # log_softmax(x)[y] = x[y] - logsumexp(x); avoids a second [B, T, V] tensor.
    token_logits = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logits - torch.logsumexp(logits, dim=-1)
    token_logps = token_logps * mask

    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def batch_logprobs(model, batch):
    return sequence_logprobs(model, batch["input_ids"], batch["attention_mask"], batch["labels"])


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logps, _ = batch_logprobs(model, batch)
    return logps


def sft_loss_from_batch(model, batch):
    """Token-mean NLL over all unmasked response tokens in the batch."""
    logps, n_tokens = batch_logprobs(model, batch)
    return -logps.sum() / n_tokens.sum().clamp_min(1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return sft_loss_from_batch(model, batch)


def dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    """Batch-mean -log sigmoid(beta * (policy gap - reference gap))."""
    policy_gap = policy_chosen - policy_rejected
    reference_gap = reference_chosen - reference_rejected
    logits = beta * (policy_gap - reference_gap)
    return -F.logsigmoid(logits).mean()


def dpo_rewards(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    """Implicit rewards r_w, r_l (manual eqs. 7-8), detached for logging."""
    chosen = beta * (policy_chosen - reference_chosen).detach()
    rejected = beta * (policy_rejected - reference_rejected).detach()
    return chosen, rejected


def concatenated_logprobs(model, dpo_batch):
    """Score chosen and rejected rows in one forward pass; returns (chosen, rejected)."""
    chosen, rejected = dpo_batch["chosen"], dpo_batch["rejected"]
    n = chosen["input_ids"].shape[0]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])

    def pad_to(t, value):
        return F.pad(t, (0, width - t.shape[1]), value=value)

    merged = {
        key: torch.cat([pad_to(chosen[key], fill), pad_to(rejected[key], fill)], dim=0)
        for key, fill in (("input_ids", 0), ("attention_mask", 0), ("labels", IGNORE_INDEX))
    }
    logps, _ = batch_logprobs(model, merged)
    return logps[:n], logps[n:]


def dpo_forward(policy_model, reference_model, dpo_batch, beta):
    """Loss and per-row log-probs for one DPO batch. Reference runs without grad."""
    policy_chosen, policy_rejected = concatenated_logprobs(policy_model, dpo_batch)
    with torch.no_grad():
        reference_chosen, reference_rejected = concatenated_logprobs(reference_model, dpo_batch)
    device = policy_chosen.device
    reference_chosen = reference_chosen.to(device).float()
    reference_rejected = reference_rejected.to(device).float()

    loss = dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    return loss, (policy_chosen, policy_rejected, reference_chosen, reference_rejected)


def compute_dpo_loss(
    policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length
):
    dpo_batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    loss, _ = dpo_forward(policy_model, reference_model, dpo_batch, beta)
    return loss
