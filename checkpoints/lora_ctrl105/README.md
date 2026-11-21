# lora_ctrl105

LoRA adapter for Qwen/Qwen3.5-9B, rank 16, alpha 32.

Control trained on 105 base-model successful train trajectories across 35 tasks.

Original causal-LM training key layout: base_model.model.model.*.

This package contains adapter weights only. Obtain the base model separately. The adapter_config base model path is normalized for portability; weight bytes and train_log are unchanged.
