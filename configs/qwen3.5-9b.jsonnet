// Adapted from the AppWorld simplified ReAct configuration (Apache-2.0).
local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "simplified_react_code_agent",
            "model_config": {
                "client_name": "openai",
                "api_type": "chat_completions",
                "base_url": "http://127.0.0.1:8000/v1",
                "api_key_env_name": "LLM_API_KEY",  // Use EMPTY for unauthenticated local vLLM.
                "name": "qwen3.5-9b",  // vLLM --served-model-name
                "temperature": 0.0,
                "seed": 100,
                "drop_reasoning_content": false,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": false}},  // vLLM chat-template option.
                "cost_per_token": {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "input_cache_write": 0.0, "output": 0.0},
                "retry_after_n_seconds": 15,
                "use_cache": false,
                "max_retries": 20,
            },
            "appworld_config": {
                "random_seed": 100,
                "raise_on_extra_parameters": true,
            },
            "logger_config": {
                "color": true,
                "verbose": true,
            },
            "usage_tracker_config": {
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "max_output_tokens_per_task": 100000,
            },
            "prompt_file_path": experiment_prompts_path + "/react_code_agent/instructions.txt",
            "ignore_multiple_calls": true,
            "max_prompt_length": null,
            "max_output_length": null,
            "max_steps": 50,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": "train",
    },
    "metadata": {
        "model": {"file_name": "qwen3.5-9b", "humanized_name": "Qwen3.5 9B", "precise_name": "Qwen/Qwen3.5-9B", "creator": "alibaba", "provider": "local_vllm"},
        "agent": {"file_name": "simplified_react_code_agent", "humanized_name": "ReAct Code Agent"},
    },
}
