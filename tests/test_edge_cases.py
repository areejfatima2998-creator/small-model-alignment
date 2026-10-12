"""Edge-case checks beyond the public smoke tests (offline, CPU only)."""

from __future__ import annotations

import math

import pytest
import torch

import submission_adapter as adapter
from alignment.losses import dpo_rewards
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class PadIsEosTokenizer(ToyTokenizer):
    """Mimics Qwen: pad and EOS share one id."""

    pad_token_id = 1


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


def brute_force_logp(model, tokenizer, prompt, response, max_length):
    """Independent per-token reference implementation (no batching, no padding)."""
    p = tokenizer.encode(prompt, add_special_tokens=False)
    r = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        r = r + [tokenizer.eos_token_id]
    ids = (p + r)[:max_length]
    logits = model(input_ids=torch.tensor([ids])).logits[0].float()
    total = 0.0
    for t in range(max(len(p), 1), len(ids)):
        total += torch.log_softmax(logits[t - 1], -1)[ids[t]].item()
    return total


def test_truncation_drops_eos_and_keeps_length():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab: "], ["hello"], max_length=6)
    assert batch["input_ids"].shape == (1, 6)
    assert tok.eos_token_id not in batch["labels"][0].tolist()
    assert batch["labels"][0, :4].eq(-100).all()
    assert batch["labels"][0, 4:].tolist() == tok.encode("he")


def test_exact_fit_keeps_eos():
    tok = ToyTokenizer()
    # 4 prompt + 2 response + 1 eos = 7
    batch = adapter.build_sft_batch(tok, ["ab: "], ["hi"], max_length=7)
    assert batch["labels"][0, -1] == tok.eos_token_id


def test_prompt_longer_than_max_length_gives_zero_logp():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["long prompt here"], ["x"], max_length=5)
    assert batch["labels"].eq(-100).all()
    logp = adapter.compute_response_logprobs(model, tok, ["long prompt here"], ["x"], max_length=5)
    assert logp.item() == 0.0


@pytest.mark.parametrize("tok_cls", [NoPadTokenizer, PadIsEosTokenizer])
def test_unusual_pad_configs_keep_eos_label(tok_cls):
    tok = tok_cls()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Longer Q: "], ["a", "bb"], max_length=32)
    for row, (p, r) in enumerate([("Q: ", "a"), ("Longer Q: ", "bb")]):
        end = len(p) + len(r) + 1
        assert batch["labels"][row, end - 1] == tok.eos_token_id
        assert batch["labels"][row, end:].eq(-100).all()
        assert batch["attention_mask"][row, :end].eq(1).all()
        assert batch["attention_mask"][row, end:].eq(0).all()


def test_no_eos_tokenizer_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["ab"], max_length=16)
    assert batch["labels"][0].ne(-100).sum() == 2


def test_logprobs_match_brute_force_and_row_order_with_mixed_lengths():
    tok = ToyTokenizer()
    torch.manual_seed(0)
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    with torch.no_grad():
        model.logit_bias.copy_(torch.randn(tok.vocab_size))
    prompts = ["Q: ", "A much longer question: ", "x", "mid: "]
    responses = ["long answer here", "y", "zz", "truncate me please"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=12)
    want = torch.tensor([brute_force_logp(model, tok, p, r, 12) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    order = [2, 0, 3, 1]
    shuffled = adapter.compute_response_logprobs(
        model, tok, [prompts[i] for i in order], [responses[i] for i in order], max_length=12
    )
    torch.testing.assert_close(shuffled, got[order])


def test_logprobs_on_tiny_qwen2_match_brute_force():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=ToyTokenizer().vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config).eval()
    tok = PadIsEosTokenizer()
    prompts = ["Q: hi ", "Longer prompt: "]
    responses = ["answer", "ok"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=20)
    want = torch.tensor([brute_force_logp(model, tok, p, r, 20) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=20)
    n_tokens = sum(len(r) + 1 for r in responses)
    torch.testing.assert_close(loss, -want.sum() / n_tokens, rtol=1e-4, atol=1e-4)


def test_dpo_loss_matches_hand_computation():
    pc = torch.tensor([-1.0, -3.0])
    pr = torch.tensor([-2.0, -1.0])
    rc = torch.tensor([-1.5, -2.0])
    rr = torch.tensor([-1.5, -2.0])
    beta = 0.5
    # logits = beta * ((pc - pr) - (rc - rr)) = [0.5, -1.0]
    expected = (-math.log(1 / (1 + math.exp(-0.5))) - math.log(1 / (1 + math.exp(1.0)))) / 2
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    assert loss.ndim == 0
    assert abs(loss.item() - expected) < 1e-6


def test_dpo_loss_is_stable_for_large_margins():
    big = torch.tensor([1000.0])
    zero = torch.tensor([0.0])
    assert torch.isfinite(adapter.compute_dpo_loss_from_logps(zero, big, zero, zero, 1.0))
    assert adapter.compute_dpo_loss_from_logps(big, zero, zero, zero, 1.0).item() < 1e-6


def test_dpo_rewards_tie_when_policy_equals_reference():
    v = torch.tensor([-3.0, -4.0])
    rw, rl = dpo_rewards(v, v - 1, v, v - 1, beta=0.1)
    assert (rw > rl).float().mean().item() == 0.0


def test_dpo_end_to_end_matches_tensor_level_and_prefers_chosen_after_steps():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size)
    reference = ToyCausalLM(tok.vocab_size)
    prompts, chosen, rejected = ["Pick: ", "Choose: "], ["a", "a"], ["b", "b"]
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.1, max_length=12)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))

    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
    for _ in range(5):
        optimizer.zero_grad()
        adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.1, max_length=12).backward()
        optimizer.step()
    a, b = tok.vocab["a"], tok.vocab["b"]
    assert policy.logit_bias[a] > policy.logit_bias[b]
    assert reference.logit_bias.grad is None


def test_dpo_batch_sides_share_prompt_masking():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["Task: "], ["good answer"], ["no"], max_length=32)
    for side in ("chosen", "rejected"):
        assert batch[side]["labels"][0, :6].eq(-100).all()
        assert batch[side]["input_ids"][0, :6].tolist() == tok.encode("Task: ")
