# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

> Draft. Sections marked **TODO** are filled in from the Kaggle outputs. Export to PDF (≤ 8 pages) for submission.

## 1. Repository, Models, and Data
- **Repository:** https://github.com/areejfatima2998-creator/small-model-alignment
- **Base model:** `Qwen/Qwen2.5-0.5B`. **Persona model:** `HuggingFaceTB/SmolLM2-360M-Instruct`.
- **SFT data:** `HuggingFaceH4/ultrachat_200k`. Train examples come from `train_sft` and validation from `test_sft`. Each split is shuffled with seed 42. Only the first user turn and first assistant reply are kept. Examples with a prompt longer than 256 tokens are dropped. Size: 30,000 train / 1,000 val.
- **HH-RLHF DPO:** `Anthropic/hh-rlhf` (all subsets; `train` / `test`). Only single-turn pairs are kept, meaning exactly one `Human:` and one `Assistant:` turn. Pairs are dropped if the two prompts differ, if the chosen and rejected responses are identical, or if either response is empty. Size: 15,000 / 1,000.
- **UltraFeedback DPO:** `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs` / `test_prefs`). Single-turn only, with score ties dropped. Size: 15,000 / 1,000.
- **LoRA persona:** `data/persona_lora/ramsay_style_preferences.json` (220 pairs). Split: 200 train / 20 eval (seed 42).
- **Evaluation sets:**
  - `eval_sets/alpaca_eval_300_seed42.json`: 300 AlpacaEval prompts sampled with `random.Random(42)` from `tatsu-lab/alpaca_eval`.
  - `eval_sets/hh_rlhf_eval_300_seed42.json`: provided.
  - `eval_sets/persona_style_prompts.json`: 24 prompts, 6 each for cooking, coding, general help and safety.
- **Seed:** 42 everywhere.
- **Prompt template** (used for SFT, DPO and all Qwen generation):
  ```
  ### Instruction:
  {instruction}

  ### Response:
  ```
  The response is followed by `<|endoftext|>` (Qwen's EOS). The persona model uses SmolLM2's own chat template, which is what TRL's conversational format applies.

## 2. Experimental Configuration
TODO: replace guide values with the values actually used if anything changed.

| | SFT | HH-RLHF DPO | UltraFeedback DPO | LoRA-DPO |
|---|---|---|---|---|
| Trainable params | full FT (fp32 master) | full FT | full FT | LoRA r=16, α=32, dropout 0.05; q,k,v,o,gate,up,down |
| Optimizer | AdamW (β=0.9,0.95) | RMSprop | RMSprop | AdamW (TRL default) |
| Peak LR / schedule | 2e-5, 3% warmup + cosine | 1e-6 constant | 1e-6 constant | 5e-5, 10% warmup + cosine |
| Weight decay | 0.1 (not on biases/norms) | 0 | 0 | 0 |
| Grad clip | 1.0 | 1.0 | 1.0 | 1.0 (HF default) |
| Batch × accum = eff. | 2 × 16 = 32 | 1 × 32 = 32 | 1 × 32 = 32 | 4 × 2 = 8 |
| Precision | fp16 autocast, fp32 weights, GradScaler | same; reference fp16 on cuda:1 | same | fp16 AMP |
| Max length / prompt | 512 / 256 | 512 / 256 | 512 / 256 | 512 / 256 |
| Epochs / steps | 1 / TODO | 1 / TODO | 1 / TODO | 3 / TODO |
| β | – | 0.1 | 0.1 | 0.1 |
| Reference | – | SFT best (`outputs/sft/best`) | SFT best | frozen base (adapter disabled) |
| Checkpoint selection | min val loss | max val reward acc | max val reward acc | final |

**Generation:**
- Qwen models: greedy, `max_new_tokens=384`, `repetition_penalty=1.1`, no sampling, the prompt template above, stop at EOS.
- Persona models: greedy, `max_new_tokens=256`, `repetition_penalty=1.1`, chat template.
- Settings are identical within each comparison and saved next to each generation file (`*_settings.json`).

**Loss reductions:**
- SFT loss is the **mean over all unmasked response tokens** in the batch. With gradient accumulation, the per-micro-batch token means are averaged.
- DPO uses response-token log-probabilities **summed** per sequence. The loss is the batch mean of −log σ(β(Δθ − Δref)).

## 3. Parameter Choices and Deviations
TODO from runs. Changes already made by design:

| Item | Guide | Used | Reason (resource vs performance) |
|---|---|---|---|
| Checkpoint save dtype | n/a | fp16 on disk | Resource: halves Kaggle disk use. Weights are loaded back to fp32 for training. |
| Prompt filtering | max prompt length 256 | prompts over 256 tokens dropped, not truncated | Truncating a prompt can cut out the instruction. Dropping keeps examples well-formed. |
| Decoding | greedy | greedy + repetition_penalty 1.1 | The 0.5B models loop under plain greedy decoding. Applied identically to every model. |

## 4. Training Behavior
TODO: insert `sft_loss.png` and `dpo_metrics.png` and discuss:
- whether SFT validation loss converged, and the step of the best checkpoint;
- DPO reward accuracy and margin over time. At step 0 these are exactly 0 (accuracy 0, loss log 2) because policy = reference;
- whether there is any sign of over-optimization, such as chosen rewards becoming negative while the margin grows.

## 5. DPO-versus-SFT Results
TODO from `*_summary.json`.

| Comparison | Eval set | N | Wins | Ties | Losses | Unknown | Failures | Win rate (excl. ties) | Win rate (ties = ½) | Judge |
|---|---|---|---|---|---|---|---|---|---|---|
| HH DPO vs SFT | HH synthetic (300) | | | | | | | | | |
| UF DPO vs SFT | AlpacaEval-300 seed 42 | | | | | | | | | |

**Persona (reported separately):**

| Comparison | Rubric | N | Wins | Ties | Losses |
|---|---|---|---|---|---|
| LoRA-DPO vs SmolLM2-Instruct | persona style | 24 | | | |
| LoRA-DPO vs SmolLM2-Instruct | usefulness/safety only | 24 | | | |

Invalid judgments: a batch whose reply is not parseable JSON is retried up to 3 times, with a 5 s sleep between attempts, and is then logged to `*_failures.jsonl`. Items within a parsed reply that have no valid winner count as "unknown" and are excluded from win rates. Position bias is reported as `position_A_preference_rate`.

## 6. Interpretation and Error Analysis
TODO. Cover:
- which model wins on each set and by how much;
- a per-category breakdown for HH (harmful vs. benign prompts) to look for over-refusal;
- length effects for UltraFeedback;
- 2–3 examples where DPO improves and 2–3 where it regresses or changes little.

## 7. Limitations and Reproducibility
- Single seed. 300-prompt eval sets. LLM-judge bias (position, length, self-preference).
- Kaggle T4 compute limits.
- Key paths: `outputs/*/run_config.json`, `outputs/*/metrics.jsonl`, `outputs/generations/*.json`, `outputs/eval/*_judged.jsonl`, `outputs/eval/*_summary.json`.
