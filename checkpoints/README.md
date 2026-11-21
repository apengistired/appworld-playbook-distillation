# LoRA checkpoints

Download the weights from the [2026-11-21 artifact release](https://github.com/apengistired/appworld-playbook-distillation/releases/tag/artifacts-2026-11-21). The base model is `Qwen/Qwen3.5-9B` and is obtained separately. The repository stores the small configurations, original training logs and [release manifest](manifest.json); the weight archives are release attachments.

| Checkpoint | Training data | Tensor layout | vLLM name |
| --- | --- | --- | --- |
| `lora_ace20` | 105 playbook-guided successful trajectories, 64 train tasks | Original causal-LM | Remap before serving |
| `lora_ace20_vllm` | Same learned tensors as `lora_ace20` | Conditional-generation language backbone | `ace20` |
| `lora_ctrl105` | 105 base-model successful trajectories, 35 train tasks | Original causal-LM | Remap before serving |
| `lora_ctrl105_vllm` | Same learned tensors as `lora_ctrl105` | Conditional-generation language backbone | `ctrl105` |

Both training runs used rank 16, alpha 32, dropout 0, learning rate 1e-4, two epochs, gradient accumulation 4, maximum sequence length 24,576 and seed 0. ACE20 recorded 1,065,265 input tokens / 154,431 loss tokens; the control recorded 899,068 / 114,488. Both retained 105 sequences and skipped zero. The JSON logs retain the original run-specific paths.

The original tensor keys start with `base_model.model.model.`. Serving copies change that prefix to `base_model.model.language_model.model.`; tensor values are unchanged. This addresses the Qwen3.5 conditional-generation LoRA compatibility issue. The three-run results used vLLM 0.33.0 on an A800 80GB.

For portable loading, the distributed `adapter_config.json` uses `base_model_name_or_path: Qwen/Qwen3.5-9B`. This is the only change to the training configuration. Weight bytes and training logs are preserved. Original local checkpoints and server backups remain separate.

Each archive extracts to `artifacts/<checkpoint>/` and includes `adapter_model.safetensors`, `adapter_config.json`, `train_log.json` and `README.md`. Verify the archive with its adjacent `.sha256` release attachment or the release's `SHA256SUMS`. The manifest also records individual file hashes.

Use `scripts/remap_lora_for_vllm.py` to regenerate a serving copy from an original adapter. Passing an already mapped adapter is idempotent. See the main README for download and evaluation commands, and [third-party notices](../THIRD_PARTY_NOTICES.md) for upstream references.
