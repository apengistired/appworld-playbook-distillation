# Third-party notices

This repository's code is distributed under the [Apache License 2.0](LICENSE).

The [AppWorld project](https://github.com/StonyBrookNLP/appworld) provides the benchmark, evaluator and ReAct agent. The two AppWorld compatibility patches and the adapted ReAct configuration are based on AppWorld revision `42b5bcf3cd334fee33f0c37c02070a9f5807add5`, under the [Apache License 2.0](patches/LICENSE). The patches modify chat option forwarding and handling of a nullable reasoning field. The configuration selects the local Qwen endpoint and generation settings.

Obtain AppWorld data and encrypted benchmark components through upstream's distribution. See [AppWorld's license terms](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/README.md#lock_with_ink_pen-license).

Published playbooks are generated outputs from the train-split adaptation run. Published evaluation files contain task-level metrics rather than raw benchmark conversations or evaluator contents. The protected AppWorld distribution is obtained separately.

The LoRA release contains learned adapter tensors and training metadata for [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). It does not contain the base model weights. Consult the upstream model card for its license and usage information.

This project's offline adaptation uses the Generator, Reflector and Curator design described in [Agentic Context Engineering](https://arxiv.org/abs/2510.04618). The original ACE project is available at [ace-agent/ace](https://github.com/ace-agent/ace).
