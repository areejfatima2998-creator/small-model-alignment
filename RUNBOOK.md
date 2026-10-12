# Kaggle Runbook

All notebooks are in `notebooks/`. Each one is self-contained: it clones (or copies) the repo, installs dependencies without touching PyTorch, checks for 2×T4, and runs the tests.

## 0. Get the repo into Kaggle (once)
Choose one:
- **GitHub:** push this repo, then set `REPO_URL` in the first cell of each notebook.
- **Kaggle dataset:** zip this folder (without `outputs/`), upload it as a private Kaggle dataset, and attach it as an input. Leave `REPO_URL = ""`.

Notebook settings for 01–04: **Accelerator: GPU T4 x2**, **Internet: On**.

## 1. Run order

| Notebook | Inputs to attach | GPU time | Produces |
|---|---|---|---|
| `01_baseline_and_sft` | repo | ~2 h | `outputs/sft/{best,final}`, base + SFT generations (AlpacaEval, HH), `sft_loss.png` |
| `02_dpo_hh` | repo + **01 output** | ~2.5 h | `outputs/dpo_hh/{best,final}`, `dpo_hh.json`, `dpo_hh_vs_sft_pairs.jsonl` |
| `03_dpo_ultrafeedback` | repo + **01 output** | ~2.5 h | `outputs/dpo_uf/{best,final}`, `dpo_uf.json`, `dpo_uf_vs_sft_pairs.jsonl` |
| `04_lora_persona` | repo | ~0.3 h | LoRA adapter, persona generations, `lora_vs_base_pairs.jsonl` |
| `05_judge` (Benchmarks notebook) | repo + **02, 03, 04 outputs** | CPU, ~0.5–1 h | `*_judged.jsonl`, `*_summary.json`, `*_failures.jsonl` |

You can run 02, 03 and 04 **in parallel** once 01 has finished.

Run each notebook with **Save Version → Save & Run All (Commit)**. That runs it headless and keeps `/kaggle/working` as a versioned output. To use one notebook's output in another, attach it via **Add Input → Your Work → Notebook output**.

## 2. Manual checks (short)
- **01 and 02/03:** read the 3 sanity-check generations. The text should be an answer, it should stop on its own (EOS), and it should not repeat itself in a loop.
- **01 SFT log:** `val_loss` should drop from about 2.x at step 0 and then flatten out.
- **02/03 DPO log:** step 0 should show `val_loss ≈ 0.693` and `val_reward_acc = 0.0`. After that, accuracy should rise above 0.5 and the margin should be positive.
- **05:** the 2-pair dry run must print parsed verdicts (`"verdict": "A"/"B"/"tie"`). If the model name is rejected, pick one from the printed list and set `JUDGE`.

## 3. After the runs
Download the output zips (`sft_run_outputs.zip`, `dpo_hh_run_outputs.zip`, `dpo_uf_run_outputs.zip`, `lora_run_outputs.zip`, `judge_outputs.zip`) into `results/` in this repo. Then generate the combined plots and fill in the report:

```bash
PYTHONPATH=. python scripts/plot_metrics.py --sft results/sft/metrics.jsonl \
  --dpo results/dpo_hh/metrics.jsonl results/dpo_uf/metrics.jsonl --dpo_labels HH-RLHF UltraFeedback --out_dir results/plots
```

## If something breaks
- **CUDA OOM in DPO:** add `--gradient_checkpointing`. Changing `--max_length` would deviate from the guide, so report it if you do.
- **OOM in SFT:** add `--gradient_checkpointing`, or use `--batch_size 1 --grad_accum 32` (same effective batch of 32).
- **Kaggle disconnects during judging:** attach the 05 output and rerun. `--reuse_judgments` skips pairs that are already judged.
