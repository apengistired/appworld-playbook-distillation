# Recorded evaluation results

[test_normal_20261121/summary.json](test_normal_20261121/summary.json) contains the five-condition, three-run matrix. Each condition has 168 tasks per run, for 504 task-runs per condition and 2,520 overall. [task_results.json](test_normal_20261121/task_results.json) includes task IDs, outcomes, step counts and available guardrail/RSI counters. [task_results.csv](test_normal_20261121/task_results.csv) is a flat metrics export. Scores are reconstructed from the saved task records; no tasks were rerun for this publication.

| Condition | r1 correct | r2 correct | r3 correct | Mean TGC |
| --- | ---: | ---: | ---: | ---: |
| Base | 35 | 34 | 26 | 18.85% |
| Control105 | 48 | 55 | 43 | 28.97% |
| Base + static Top-20 | 88 | 88 | 92 | 53.17% |
| LoRA20, no playbook | 96 | 89 | 91 | 54.76% |
| LoRA20 + full guardrails + RSI V5 | 91 | 96 | 100 | 56.94% |

The environment was AppWorld 0.2.0, vLLM 0.33.0 and an A800 80GB. Generation used non-thinking mode, `temperature=0`, `seed=100`, 3,000 maximum output tokens per decision and 50 maximum execution steps. The repetitions share the same seed; they are not independent seed selections. Baseline conditions ran concurrently with four workers each; the RSI evaluation used eight workers. Continuous batching and runtime variation limit causal comparisons between individual trajectories.

The RSI condition uses `injection_mode=none`, `guardrails=full`, `auto_doc=none`, `rsi_guards=v5`, `retry_mode=none`. Other conditions have no guardrails or retries; only the static-playbook condition injects the Top-20. For inactive guardrails, the summary's `auto_doc=none` means no automatic documentation was delivered. There were no critic calls in this comparison.

Mean TGC is the arithmetic mean of the three run percentages. SD is the sample standard deviation in percentage points. `pass_at_3` counts task IDs solved at least once in the three recorded repetitions; `all_3` counts IDs solved in all three.

The RSI condition rescued 12 / 22 / 20 task-runs and regressed 17 / 15 / 11 against the corresponding LoRA20 runs, for net changes of -5 / +7 / +9. Overall the combination added 11 successes across 504 task-runs, or 2.18 percentage points. Rule hits indicate co-occurrence, not proof that a rule caused a rescue or regression. This comparison does not isolate RSI from the existing full guardrails.

[dev_20261121/v5_lora_core_r1.json](dev_20261121/v5_lora_core_r1.json) records the separate 57-task development run: 39/57 (68.42%). The development gate informed the frozen configuration subsequently used on `test_normal`.

These public exports contain metrics and generic rule-hit identifiers. Complete raw trajectories, model request logs, task data, evaluator details and server snapshots remain in the local archive. Obtain benchmark data and protected components from AppWorld upstream rather than this repository.
