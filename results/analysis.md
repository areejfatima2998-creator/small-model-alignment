# Results analysis (auto-generated)

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
