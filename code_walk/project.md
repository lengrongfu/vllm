# vllm项目结构记录

```
$ tree -L 1 vllm
├── __init__.py # 指定要导出的函数和class
├── __pycache__
├── _custom_ops.py
├── _ipex_ops.py
├── adapter_commons # 一些涉及到适配的代码，比如model、request
├── assets #
├── attention # attention 的后端实现库
├── beam_search.py
├── benchmarks
├── compilation
├── config.py
├── connections.py
├── core
├── device_allocator
├── distributed
├── engine
├── entrypoints
├── envs.py
├── executor
├── fa_utils.py
├── forward_context.py
├── inputs
├── jsontree.py
├── logger.py
├── logging_utils
├── logits_process.py
├── lora
├── model_executor
├── multimodal
├── outputs.py
├── platforms
├── plugins
├── pooling_params.py
├── profiler
├── prompt_adapter
├── py.typed
├── sampling_params.py
├── scalar_type.py
├── scripts.py
├── sequence.py
├── spec_decode
├── test_utils.py
├── third_party
├── tracing.py
├── transformers_utils
├── triton_utils
├── usage
├── utils.py
├── v1
├── version.py
├── vllm_flash_attn
└── worker
```