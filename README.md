# AppWorld Playbook Distillation

Distill an ACE-style playbook into a Qwen3.5-9B LoRA adapter. The teacher uses the playbook to solve AppWorld training tasks. The student learns from the successful trajectories and is evaluated with the standard ReAct prompt, with no playbook injected.

The pipeline includes a parallel rollout runner, AppWorld unit-test evaluation, playbook curation and ranking, trajectory-to-SFT conversion, and assistant-only LoRA training.

```text
Train tasks + evaluation feedback → ACE adaptation → Top-20 playbook
Train tasks + top-20 playbook → Teacher rollouts → Keep successes
Successful trajectories → Remove playbook → LoRA SFT → Evaluate without playbook
```

## Results

Historical results use AppWorld 0.2.0, Qwen3.5-9B in non-thinking mode, and the simplified ReAct harness. Task Goal Completion (TGC) is the percentage of tasks that pass the AppWorld unit tests. Each dev run covers 57 tasks. Each test_normal run covers 168 tasks.

| Condition | dev TGC (3 runs) | test_normal TGC |
| --- | ---: | ---: |
| Qwen3.5-9B base | 21.1% | 18.8% (2 runs) |
| LoRA on 105 base-model successes | 32.2% | 31.0% (1 run) |
| Base + top-20 playbook | 53.8% | 52.7% (2 runs) |
| **LoRA on 105 playbook-guided successes, no playbook at inference** | **55.0%** | **53.9% (2 runs)** |

The teacher produced 105 training examples across 64 distinct tasks from two passes over the 90-task train split. On test_normal, the playbook and distilled adapter passed 177/336 and 181/336 task-runs, respectively. The equal-sized base-model control covered 35 training tasks, so it matches the number of examples while task coverage differs. These scores describe this harness and split. A separate full-playbook run with the official ACE harness recorded 58.3% TGC on test_normal.

In the historical interference comparison, adding the same Top-20 playbook back to the distilled adapter as a static prompt reduced dev TGC from 55.0% (94/171 task-runs) to 46.8% (80/171 task-runs). Scoped injection tests whether selectively surfacing rules can reduce this interference.

### Three-Run `test_normal` Matrix & V5 RSI-Harness Results

Three-run evaluation on `test_normal` (168 tasks × 3 runs = 504 task-runs per condition, AppWorld 0.2.0, `Qwen3.5-9B`, `max_tokens=3000`) confirms the distillation ladder and adds deterministic **V5 RSI-Harness** execution guards distilled from `dev` failure clusters (`--guardrails full --auto-doc none --rsi-guards v5 --retry-mode none`):

| Condition | r1 | r2 | r3 | Mean TGC ± SD | Solved (/504) | `pass@3` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` (`qwen3.5-9b`, no playbook) | 35 (20.83%) | 34 (20.24%) | 26 (15.48%) | 18.85% ± 2.94% | 95 | — |
| `control105` (LoRA on base successes) | 48 (28.57%) | 55 (32.74%) | 43 (25.60%) | 28.97% ± 3.59% | 146 | — |
| `base_static20` (`qwen3.5-9b` + static Top-20) | 88 (52.38%) | 88 (52.38%) | 92 (54.76%) | 53.17% ± 1.37% | 268 | — |
| `lora20` (`ace20` LoRA, no playbook) | 96 (57.14%) | 89 (52.98%) | 91 (54.17%) | 54.76% ± 2.15% | 276 | 112/168 (66.67%) |
| **`v5_lora_core` (`ace20` + RSI-Harness v5)** | **91 (54.17%)** | **96 (57.14%)** | **100 (59.52%)** | **56.94% ± 2.68%** | **287** | **119/168 (70.83%)** |

On the `dev` gate (57 tasks) and its 25-task baseline failure subset (`dev_fail25`, selected from baseline LoRA failures):

- **`dev_fail25` (25 tasks, r1)**: V4 (`guardrails=full`, `auto-doc=full`, selective retry) scored `8/25 (32.00%)` before critic fix and `6/25 (24.00%)` after critic fix (`retried=1`); V5 RSI with `auto-doc=full` reached `12/25 (48.00%)`; **V5 RSI core (`--auto-doc none --rsi-guards v5 --retry-mode none`) reached `14/25 (56.00%)`**.
- **`dev` (57 tasks, r1)**: V4 scored `30/57 (52.63%)` before critic fix and `33/57 (57.89%)` with `auto-doc=full` / fixed critic; **`v5_dev_lora_core` reached `39/57 (68.42%)`** (+6 tasks over the best V4 `dev` baseline, with zero RSI-associated regressions).
- Across the three `test_normal` runs, `v5_lora_core` lifted total solves from 276/504 (`lora20`) to 287/504 (+11 solves, `all-3`: 69/168 → 71/168, `pass@3`: 112/168 → 119/168) with zero extra critic model calls or runtime `auto_doc` overhead.

## Setup

Use Python 3.11+ for the AppWorld runner. For inference and LoRA training, use a Linux CUDA environment with vLLM, PyTorch, Transformers and PEFT builds that support Qwen3.5. The original training run used an A800. Keep inference and training in separate environments if their dependencies conflict.

See [requirements.txt](requirements.txt) for the core Python dependencies. For training, install a CUDA-compatible PyTorch build, then run `python -m pip install -r requirements.txt` in the training environment. Install vLLM separately using its [GPU installation instructions](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/). The AppWorld environment below handles the rollout and evaluator dependencies.

From a fresh checkout, with Git LFS installed and `python3` pointing to Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate

mkdir -p third_party
git clone --no-checkout https://github.com/StonyBrookNLP/appworld.git third_party/appworld
git -C third_party/appworld lfs install --local
GIT_LFS_SKIP_SMUDGE=1 git -C third_party/appworld checkout --detach 42b5bcf3cd334fee33f0c37c02070a9f5807add5
git -C third_party/appworld lfs pull
git -C third_party/appworld apply \
  ../../patches/appworld_extra_body_passthrough.patch \
  ../../patches/appworld_reasoning_content_none.patch

python -m pip install -e third_party/appworld \
  -e 'third_party/appworld/experiments[simplified]' \
  'litellm==1.78.2' 'fastapi==0.141.1' 'starlette==1.6.0' 'pydantic==2.13.5' \
  'sqlalchemy==2.0.52' 'sqlmodel==0.0.42'
(cd third_party/appworld && appworld install --repo)
```

LiteLLM 1.78.2 supports the OpenAI SDK 1.x required by the pinned AppWorld agents package. The web and database package versions above were checked with a one-task run using a local stub model endpoint.

Download the dataset from the repository root if `data/` does not exist. The upstream command removes an existing `data/` before downloading.

```bash
test ! -e data && APPWORLD_ROOT="$PWD" appworld download data --version 0.2.0
```

The patches forward vLLM chat-template options and handle a nullable reasoning field. The scripts use an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`. Set `APPWORLD_MODEL_URL` and, if needed, `LLM_API_KEY` to override the default. Use a local model endpoint for runs involving AppWorld reference solutions and evaluator feedback.

### Smoke test

After starting vLLM as in [step 1 below](#run-the-pipeline), run one dev task from the AppWorld environment:

```bash
python scripts/run_lesson.py --cond smoke_base --tasks dev --limit 1 --runs r1 --np 1
```

Check that the runner prints `ALL DONE` with one completed task and that the task record in `experiments/gate1/smoke_base_r1/` has `"error": null` and a positive `num_steps`. A task can run correctly and still fail its unit tests. Use a new condition label to retest after changing the environment or model.

## Run the pipeline

1. Start the base model server in the GPU environment.

   ```bash
   vllm serve Qwen/Qwen3.5-9B --served-model-name qwen3.5-9b \
     --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 65536 \
     --limit-mm-per-prompt '{"image": 0, "video": 0}' --seed 100
   ```

2. In the AppWorld environment, build the playbook on the train split. This ACE-style implementation uses parallel generator/reflector batches and applies curator updates after each batch. The reflector can use AppWorld reference solutions and test reports during training.

   ```bash
   python scripts/ace_offline.py --run ace20 --epochs 2 --batch 30 --rounds 5 --np 8
   python scripts/playbook_topn.py experiments/ace/ace20/playbook_final.json \
     20 artifacts/playbook_top20.txt
   ```

3. Generate teacher trajectories and build the SFT dataset. The builder keeps successful train tasks, removes the injected playbook, and records demonstration turns so they can be masked during training.

   ```bash
   python scripts/run_lesson.py --cond teacher --tasks train --runs r1,r2 --np 8 \
     --header playbook --lesson artifacts/playbook_top20.txt
   python scripts/build_sft.py --cond teacher --runs r1,r2 --header playbook \
     --lesson artifacts/playbook_top20.txt --out artifacts/sft/ace20.jsonl
   ```

4. Stop the inference server if training on the same GPU. In the training environment, run LoRA SFT with rank 16, learning rate 1e-4 and two epochs. Use `--model` to point to local weights.

   ```bash
   python scripts/train_lora.py --data artifacts/sft/ace20.jsonl \
     --out artifacts/lora_ace20 --rank 16 --lr 1e-4 --epochs 2
   ```

5. For vLLM serving Qwen3.5 as `Qwen3_5ForConditionalGeneration`, create a serving copy of the adapter. The script moves causal-LM tensor keys under the language backbone prefix expected by vLLM and preserves the original training checkpoint.

   ```bash
   python -m pip install safetensors
   python scripts/remap_lora_for_vllm.py \
     --src artifacts/lora_ace20 --dst artifacts/lora_ace20_vllm
   ```

   Restart the vLLM command from step 1 with `--enable-lora --max-lora-rank 16 --lora-modules ace20=artifacts/lora_ace20_vllm` appended. Keep the other serving settings from step 1. Run the benchmark evaluations in the AppWorld environment.

   ```bash
   for split in dev test_normal; do
     runs=r1,r2
     if [ "$split" = dev ]; then runs=r1,r2,r3; fi
     python scripts/run_lesson.py --cond "eval_base_$split" --tasks "$split" --runs "$runs" --np 8
     python scripts/run_lesson.py --cond "eval_playbook_$split" --tasks "$split" --runs "$runs" --np 8 \
       --header playbook --lesson artifacts/playbook_top20.txt
     python scripts/run_lesson.py --cond "eval_lora_$split" --model ace20 \
       --tasks "$split" --runs "$runs" --np 8
   done
   ```

The runner prints success counts and saves per-task records under `experiments/gate1/`. It reuses completed records for the same condition and run labels. Use fresh labels after changing model or prompt settings.

For a matched control, run the base model on train with no `--lesson`, then call `build_sft.py` with that condition and `--sample-size` equal to the teacher dataset size. The historical control also drew from a separate pool of prior base rollouts.

## Harness Extensions

Use `scripts/run_lesson.py` for all rollout conditions. The historical configuration is `--injection-mode static --guardrails none`.

| Flag | Modes |
| --- | --- |
| `--injection-mode` | `static` appends the supplied lesson to the initial task prompt. `none` ignores the lesson. `jit_scoped` requires a rendered ACE playbook or JSON bullets file via `--lesson` and selects up to `--jit-topk` rules per triggered request (default 3). |
| `--guardrails` | `none` disables all checks. `basic` adds pagination reminders and tracks writes followed by reads. `full` also holds the first risky submission for verification. |

Scoped JIT runs on the first decision, after an execution error, or when a new app appears in the preceding step. App names in examples and references to the shared supervisor API do not restrict a rule's scope. The hint is attached to a copy of the latest user message and has a one-decision cooldown. Submit verification observations are sent without a JIT hint.

The guardrails recognize AppWorld API calls in Python syntax. Pagination hints apply to reads outside loops that omit explicit paging and large page limits. Evidence includes continuation markers, truncation, a default-sized list, or a printed count. Each API is reminded at most once, with two reminders per task. The submit gate checks unchecked writes, possibly non-null action answers, failure submissions, and action tasks with zero modifications. A clear query with no writes may submit a non-null answer directly. The held cell is skipped as a unit. Each task has a one-intercept limit. These checks are benchmark heuristics for visible calls, with limited coverage of runtime aliases and indirect calls.

Guarded task records include a `guardrails` object with the counters (including `auto_doc_injections`) and `num_agent_steps`, which includes intercepted decisions. `num_steps` still counts executed AppWorld cells. The runner prints average steps and, when checks are enabled, the aggregated counters. Opt-in runs record their settings and require fresh labels when the model, playbook, or harness configuration changes.

V4 adds live API documentation to execution errors in guarded runs. The existing task's `apis.api_docs` supplies parameter constraints or a concise API list after an unknown-method error. This does not execute an extra agent cell. Each `(app, api)` is documented at most once per task.

Use `--retry-mode selective` for runtime-only auditing before scoring. Its default `--retry-trigger risk_only` audits objective risk signals: explicit failure or runner error, readback interception, errors after attempted writes, action tasks with zero writes, multiple execution errors, or at least 40 executed steps. Use `--retry-trigger all_audited` to audit every task. The default critic is the base `qwen3.5-9b`, with thinking enabled and 8192 output tokens; override with `--critic-model` or `--no-critic-thinking`.

V5 audit and judge prompts retain every executed code cell and every successful or failed observation in full. This preserves newly appended items at the end of queues and other collections. The prompts distinguish existing state from requested changes and check both missing changes and unrelated additions. vLLM's `/tokenize` endpoint checks the exact input size before generation; if the complete evidence cannot fit, the request is logged and Pass 1 is preserved instead of truncating evidence.

A concrete repair verdict runs one fresh sandbox with the critique and error facts. Pass 2 uses `--retry-temperature 0.4` by default and the actor seed plus one. Explicit failure or an unhandled final exception followed by normal completion selects the retry directly. Other switches require a high-confidence pairwise preference with exact, validated step quotes from both attempts, followed by agreement after swapping their order. Malformed, unsupported, or inconsistent judgments preserve Pass 1. These checks reduce unsupported switches but cannot guarantee semantic correctness. Raw audit/judge prompts, responses, token counts, context skips, and output truncations are saved under each run's `critic/` directory. Selective runs load tasks with `load_ground_truth=False`, and selection finishes before scoring.

V5 RSI adds four opt-in rule families via `--rsi-guards v5`: default-page-limit mistakes and missing song ratings, direct-card funding and per-item batch recovery, exact user matching and complete message context, and rename date/idempotency checks. These inspect generic API calls and Python syntax without task IDs or answer values. Hints are deduplicated by API/subrule. The default `--rsi-guards none` preserves previous execution behavior. `--auto-doc none` suppresses automatic documentation without disabling RSI or the existing guardrails; its default remains `full`.

The six existing `guardrails` counters retain their keys. RSI runs add separate per-task `rsi_triggers` counters and `rsi_hits` step/key records. RSI can also be explicitly enabled with `--guardrails none` to inspect its hints independently.

`scripts/run_rsi_gate.py` runs a dev-only two-stage comparison with LoRA20, no inference playbook, full core guardrails, and no retries. Stage 1 compares RSI with/without auto-doc on the supplied 25 dev failures. Stage 2 evaluates the winner on all 57 dev tasks, preferring no auto-doc on a tie. Its default `--dev-policy auto` also evaluates both variants on dev when Stage 1 differs by at least three tasks; `--dev-policy winner` restricts it to the winner. It writes `v5_summary.json` / `v5_summary.md` with task-level rescues, regressions, and per-rule co-occurrence counts. The gate cannot dispatch the held-out split and stops after dev. Rule co-occurrence is not causal attribution.

The isolated overnight relay is `scripts/run_v4_overnight.sh` (deploy as `run_v4_overnight.sh` beside `scripts/` on the server). It waits for the existing matrix's successful DONE/COMPLETE handoff and all baseline processes to exit, then runs 25 dev failures, full dev, and test_normal with frozen V4 settings. All results remain under the independent evaluation root; `v4_summary.md` and `v4_summary.json` refresh during runs and after each stage. It never restarts vLLM or shuts down the server.

Task records save `retry` and `selected_exp`. Guardrail counters and step counts describe the selected attempt, and elapsed time includes auditing and retrying. The final summary reports `retried` and `selected_retry`. Use fresh condition labels for selective runs and changed harness revisions.

With the base model and adapter served as in step 5, run a 2×2 comparison on the same split and run labels. Use `baseline=static` for a common static baseline or `baseline=none` for a common no-playbook baseline.

```bash
baseline=static
playbook=artifacts/playbook_top20.txt
split=dev
for model in qwen3.5-9b ace20; do
  python scripts/run_lesson.py --cond "harness_${model}_${baseline}_${split}" \
    --model "$model" --tasks "$split" --runs r1,r2,r3 --np 8 \
    --lesson "$playbook" --header playbook --injection-mode "$baseline" --guardrails none
  python scripts/run_lesson.py --cond "harness_${model}_jit_full_${split}" \
    --model "$model" --tasks "$split" --runs r1,r2,r3 --np 8 \
    --lesson "$playbook" --injection-mode jit_scoped --jit-topk 3 --guardrails full
done
```

This comparison evaluates JIT routing and guardrails together. Vary one flag at a time to measure their individual contributions. Use static injection for the documented teacher-to-SFT workflow.

To run the recommended V5 RSI-Harness core configuration (`LoRA20` + deterministic AST/runtime RSI guards, without noisy per-error `auto-doc` or runtime critic overhead):

```bash
python scripts/run_lesson.py --cond v5_test_normal_lora_core \
  --model ace20 --tasks test_normal --runs r1,r2,r3 --np 8 \
  --injection-mode none --guardrails full --auto-doc none \
  --rsi-guards v5 --retry-mode none --max-tokens 3000
```

## Reuse a saved run

If you have the trained adapter, put its `adapter_config.json` and `adapter_model.safetensors` under `artifacts/lora_ace20/`. Follow step 5 to create the serving copy, load it with the matching Qwen3.5-9B base model, and run the LoRA evaluation directly. A saved adapter does not require the playbook or training trajectories at inference.

If you have `artifacts/sft/ace20.jsonl`, start from step 4 to train a new adapter from those trajectories. A saved `artifacts/playbook_top20.txt` can be used directly for the playbook-guided evaluation in step 5.

## Run the CPU unit tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic inputs to check playbook ranking, trajectory filtering, JIT routing, guardrails, and LoRA tensor key remapping. Install `safetensors` to include the CPU file round-trip test. See [examples](examples/README.md) for guidance on sanitized playbooks and failure examples.

## License

The repository code is available under [Apache License 2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for AppWorld and its data terms.

## References

- [AppWorld](https://github.com/StonyBrookNLP/appworld) provides the simulated environment, ReAct agent and evaluator. See [third-party notices](THIRD_PARTY_NOTICES.md).
- [Agentic Context Engineering (ACE)](https://github.com/ace-agent/ace) provides the Generator–Reflector–Curator design.
