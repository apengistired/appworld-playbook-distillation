# Third-party notices

The [AppWorld project](https://github.com/StonyBrookNLP/appworld) provides the benchmark, evaluator and ReAct agent. The two AppWorld compatibility patches and the adapted ReAct configuration are based on AppWorld revision `42b5bcf3cd334fee33f0c37c02070a9f5807add5`, under the [Apache License 2.0](patches/LICENSE). The patches modify chat option forwarding and handling of a nullable reasoning field. The configuration selects the local Qwen endpoint and generation settings.

Obtain AppWorld data and encrypted benchmark components through upstream's distribution. See [AppWorld's license terms](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/README.md#lock_with_ink_pen-license).

This project's offline adaptation uses the Generator, Reflector and Curator design described in [Agentic Context Engineering](https://arxiv.org/abs/2510.04618). The original ACE project is available at [ace-agent/ace](https://github.com/ace-agent/ace).
