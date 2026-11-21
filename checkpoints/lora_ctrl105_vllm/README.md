# lora_ctrl105_vllm

LoRA adapter for Qwen/Qwen3.5-9B, rank 16, alpha 32.

Control trained on 105 base-model successful train trajectories across 35 tasks.

Serving key layout: base_model.model.language_model.model.* (vLLM Qwen3_5ForConditionalGeneration).

This package contains adapter weights only. Obtain the base model separately. The adapter_config base model path is normalized for portability; weight bytes and train_log are unchanged.
