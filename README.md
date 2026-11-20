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

## Setup

Use Python 3.11+ for the AppWorld runner. For inference and LoRA training, use a Linux CUDA environment with vLLM, PyTorch, Transformers and PEFT builds that support Qwen3.5. The original training run used an A800. Keep inference and training in separate environments if their dependencies conflict.

From the repository root, with Git LFS installed:

```bash
python3 -m venv .venv
source .venv/bin/activate

mkdir -p third_party
git clone https://github.com/StonyBrookNLP/appworld.git third_party/appworld
git -C third_party/appworld lfs install --local
git -C third_party/appworld checkout 42b5bcf3cd334fee33f0c37c02070a9f5807add5
git -C third_party/appworld lfs pull
git -C third_party/appworld apply \
  ../../patches/appworld_extra_body_passthrough.patch \
  ../../patches/appworld_reasoning_content_none.patch

python -m pip install -e third_party/appworld -e 'third_party/appworld/experiments[simplified]'
(cd third_party/appworld && appworld install --repo)
appworld download data --version 0.2.0
```

The patches forward vLLM chat-template options and handle a nullable reasoning field. The scripts use an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`. Set `APPWORLD_MODEL_URL` and, if needed, `LLM_API_KEY` to override the default. Use a local model endpoint for runs involving AppWorld reference solutions and evaluator feedback.

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

5. Restart the vLLM command from step 1 with `--enable-lora --max-lora-rank 16 --lora-modules ace20=artifacts/lora_ace20` appended. Run the benchmark evaluations in the AppWorld environment.

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

## Check the data pipeline without a GPU

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic data to exercise playbook ranking, success and split filtering, and removal of the injected playbook. AppWorld data is downloaded from upstream. This repository contains the implementation and aggregate results.

## References

- [AppWorld](https://github.com/StonyBrookNLP/appworld) provides the simulated environment, ReAct agent and evaluator. See [third-party notices](THIRD_PARTY_NOTICES.md).
- [Agentic Context Engineering (ACE)](https://github.com/ace-agent/ace) provides the Generator–Reflector–Curator design.
