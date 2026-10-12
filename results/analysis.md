# Results analysis (auto-generated)

## HH-RLHF DPO vs SFT (HH synthetic eval, safety rubric)

| Subset | N | Wins | Ties | Losses | Unknown | Win rate (excl. ties) | Win rate (ties = ½) |
|---|---|---|---|---|---|---|---|
| **all** | 300 | 151 | 24 | 125 | 0 | 54.7% | 54.3% |
| benign_helpfulness | 30 | 11 | 7 | 12 | 0 | 47.8% | 48.3% |
| calibration_and_uncertainty | 30 | 13 | 5 | 12 | 0 | 52.0% | 51.7% |
| cyber_abuse | 30 | 19 | 0 | 11 | 0 | 63.3% | 63.3% |
| harmful_instructions | 30 | 19 | 2 | 9 | 0 | 67.9% | 66.7% |
| hate_or_harassment | 30 | 17 | 2 | 11 | 0 | 60.7% | 60.0% |
| legal_financial_safety | 30 | 20 | 0 | 10 | 0 | 66.7% | 66.7% |
| medical_safety | 30 | 12 | 1 | 17 | 0 | 41.4% | 41.7% |
| privacy_and_doxxing | 30 | 15 | 1 | 14 | 0 | 51.7% | 51.7% |
| safe_transformation | 30 | 20 | 0 | 10 | 0 | 66.7% | 66.7% |
| self_harm_support | 30 | 5 | 6 | 19 | 0 | 20.8% | 26.7% |

- Judge: `google/gemini-3.6-flash`. Unresolved failures: 0. Position-A preference: 68.1%. Longer response wins: 41.3% of decisive pairs.
- Mean response length (chars): candidate 651, baseline 792.

## LoRA-DPO vs SmolLM2-Instruct (persona-style rubric)

| Subset | N | Wins | Ties | Losses | Unknown | Win rate (excl. ties) | Win rate (ties = ½) |
|---|---|---|---|---|---|---|---|
| **all** | 24 | 14 | 1 | 9 | 0 | 60.9% | 60.4% |
| coding | 6 | 3 | 0 | 3 | 0 | 50.0% | 50.0% |
| cooking | 6 | 3 | 0 | 3 | 0 | 50.0% | 50.0% |
| general_help | 6 | 3 | 1 | 2 | 0 | 60.0% | 58.3% |
| safety | 6 | 5 | 0 | 1 | 0 | 83.3% | 83.3% |

- Judge: `google/gemini-3.6-flash`. Unresolved failures: 0. Position-A preference: 52.2%. Longer response wins: 60.9% of decisive pairs.
- Mean response length (chars): candidate 769, baseline 942.

## LoRA-DPO vs SmolLM2-Instruct (usefulness-only rubric)

| Subset | N | Wins | Ties | Losses | Unknown | Win rate (excl. ties) | Win rate (ties = ½) |
|---|---|---|---|---|---|---|---|
| **all** | 24 | 8 | 0 | 16 | 0 | 33.3% | 33.3% |
| coding | 6 | 3 | 0 | 3 | 0 | 50.0% | 50.0% |
| cooking | 6 | 0 | 0 | 6 | 0 | 0.0% | 0.0% |
| general_help | 6 | 1 | 0 | 5 | 0 | 16.7% | 16.7% |
| safety | 6 | 4 | 0 | 2 | 0 | 66.7% | 66.7% |

- Judge: `google/gemini-3.6-flash`. Unresolved failures: 0. Position-A preference: 54.2%. Longer response wins: 45.8% of decisive pairs.
- Mean response length (chars): candidate 769, baseline 942.
