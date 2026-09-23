# NVIDIA Model Optimizer

The [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) is a library designed to optimize models for inference with NVIDIA GPUs. It includes tools for Post-Training Quantization (PTQ) and Quantization Aware Training (QAT) of Large Language Models (LLMs), Vision Language Models (VLMs), and diffusion models.

We recommend installing the library with:

```bash
pip install nvidia-modelopt
```

## Supported ModelOpt checkpoint formats

vLLM detects ModelOpt checkpoints via `hf_quant_config.json` and supports the
following `quantization.quant_algo` values:

- `FP8`: per-tensor weight scale (+ optional static activation scale).
- `FP8_PER_CHANNEL_PER_TOKEN`: per-channel weight scale and dynamic per-token activation quantization.
- `FP8_PB_WO` (ModelOpt may emit `fp8_pb_wo`): block-scaled FP8 weight-only (typically 128×128 blocks).
- `NVFP4`: ModelOpt NVFP4 checkpoints (use `quantization="modelopt_fp4"`).
- `W4A16_NVFP4`: weight-only ModelOpt NVFP4 checkpoints with 16-bit
  activations (use `quantization="modelopt_fp4"`).
- `MXFP8`: ModelOpt MXFP8 checkpoints (use `quantization="modelopt_mxfp8"`).

!!! note
    For NVFP4 checkpoints, vLLM selects a GEMM kernel automatically at load
    time from the backends available on the current platform (CUTLASS,
    FlashInfer, Marlin, and others). On GPUs without a supported native FP4
    GEMM kernel, vLLM falls back to weight-only (W4A16) execution via Marlin
    and logs a warning; this may reduce throughput for compute-heavy
    workloads. Use `--linear-backend` to override the automatic selection
    (this replaces the deprecated `VLLM_NVFP4_GEMM_BACKEND` environment
    variable). Values relevant to NVFP4 include `cutlass`,
    `flashinfer_cutlass`, `flashinfer_cutedsl`, `flashinfer_trtllm`,
    `flashinfer_cudnn`, and `marlin`; see `KernelConfig` on the
    [Engine Arguments](../../configuration/engine_args.md) page and shown by
    `vllm serve --help=KernelConfig`. For `W4A16_NVFP4`, `auto` currently
    selects Marlin. Models with BF16 activations can explicitly select the
    FlashInfer CuTe-DSL backend with `--linear-backend flashinfer_cutedsl`.

!!! note
    For models quantized to MXFP8 with BF16 activations on SM100-family GPUs,
    use `--linear-backend flashinfer_trtllm` to select FlashInfer's TensorRT-LLM
    GEMM backend.

## GLM-5.3 NVFP4 with Mega-Gate

`VLLM_GLM_USE_MEGA_GATE=1` opts GLM DSA models into DeepGEMM's fused BF16
router GEMM and expert selection on SM10x GPUs (for example, B200/B300)
or SM90 (H100/H800), with a DeepGEMM build supporting the active architecture.
The NVFP4 expert weights and their quantization scales remain unchanged.
The feature is disabled by default pending model-level accuracy and performance
validation. Both paths require DeepGEMM exposing `bf16_mega_gate`.

For example, on a node with eight sufficiently large SM10x GPUs:

```bash
VLLM_GLM_USE_MEGA_GATE=1 vllm serve RadixArk/GLM-5.3-NVFP4 \
  --quantization modelopt_fp4 \
  --tensor-parallel-size 8 \
  --moe-backend flashinfer_cutlass \
  --max-model-len 8192
```

Mega-Gate requires a modular expert backend that accepts precomputed routing
weights and IDs. Selecting a monolithic backend keeps the original gate/router;
`auto` may select such a backend. This option does not select
`deep_gemm_mega_moe` or convert NVFP4 weights to MXFP4.

On SM90, this branch also connects Mega-Gate to its model-owned NVFP4 MegaMoE
backend. On eight sufficiently large Hopper GPUs, enable it with:

```bash
VLLM_GLM_USE_MEGA_GATE=1 vllm serve RadixArk/GLM-5.3-NVFP4 \
  --quantization modelopt_fp4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --max-model-len 8192
```

This command requires this branch's SM90 NVFP4 MegaMoE DeepGEMM build.
Build and install the local SM90 Mega-Gate branch into the vLLM environment:

```bash
DG_FORCE_BUILD=1 uv pip install --python .venv/bin/python \
  --no-build-isolation --no-deps /home/jovyan/code/DeepGEMM
```

Rebuild its C++ extension after changing its API; vLLM prefers the external
`deep_gemm` package over its bundled copy. Set `CUDA_HOME` to the toolkit used
for its JIT, and ensure a compatible host C++ compiler is on `PATH`.
Host-header changes require a clean extension build: `DG_FORCE_BUILD=1`
bypasses prebuilt wheel downloads but may still reuse stale build objects.
After installing, verify that `get_bf16_mega_gate_config(128, 6144, 256, 8)`
reports `block_tokens=32`, and the 256-token configuration has `use_gemm=False`.
The optimized SM90 backend chooses the implementation by local batch size:
small batches use a warp-specialized, multi-stage TMA/WGMMA kernel with
expert and split-K parallelism, followed by fused reduction and routing when
split-K exceeds one. The GLM-specific tuning uses 32-token tiles for 65–256
tokens, including the 256-token batch that previously selected ordinary GEMM.
Larger batches use DeepGEMM's tuned SM90 BF16 GEMM followed by routing.
Scratch buffers are allocated through PyTorch and
captured by CUDA Graphs; there are no shared cached cross-CTA counters.
vLLM delegates these choices to DeepGEMM's `get_bf16_mega_gate_config` heuristic.
The former Triton implementation and its 64-token upper limit are no longer used.
GLM-5.3 uses hidden size 6144, 256 experts and top-8. On SM90, eligible local
batches of 17–128 tokens, exactly 256 tokens, and at least 3584 tokens use
DeepGEMM. Other batches retain the original GEMM plus fused grouped-top-k:
measurements show regressions in the middle range, especially with cold L2.
This shape policy does not change the SM10x path or other gate dimensions.
Rebuild the updated local DeepGEMM branch to obtain the 32-token tile tuning;
vLLM does not change process-wide tuning environment variables during forward.
This opt-in does not guarantee a speedup at every shape or on every SM90 GPU.
Older builds without SM90 Mega-Gate support fall back to the original gate/router.
SM90 passes the forward context's padding mask to DeepGEMM, so padded rows
emit zero weights and invalid IDs without entering top-k selection.

The fused path uses single-group sigmoid routing with normalized top-k weights.
Local batches of at most 16 tokens use the original gate/router. Unsupported
hardware, tensor layouts, padding-skip
routing, and batch-invariant execution also fall back. LoRA, PCP, and MoE
configurations requiring external logits dispatch do not enable the fused path.
On SM10x, each CUDA stream must execute Mega-Gate before graph capture; capture on an
unwarmed stream falls back without allocating DeepGEMM barriers inside the graph.

To run the focused routing and runner tests, including the GPU kernel cases:

```bash
.venv/bin/python -m pytest tests/kernels/moe/test_routing.py -k mega_gate -v
```

To benchmark the SM90 gate against the existing GEMM plus grouped-top-k path,
install `cupti-python` and run
`.venv/bin/python benchmarks/kernels/benchmark_mega_gate.py`.
Use `--cache warm` for warm-L2 measurements and `--output results.json` to save
the per-round timings, selected kernel configurations and source fingerprints.
The benchmark follows vLLM's shape policy by default; `--raw-native` measures
DeepGEMM directly even for batches that vLLM would send to the original router.
The baseline includes vLLM's fused grouped-top-k, rather than separate PyTorch
sigmoid, bias, top-k, gather and normalization operators.

Compare the same checkpoint and expert backend with the option on and off for
model evaluation and serving benchmarks before using it in production.

## Quantizing HuggingFace Models with PTQ

You can quantize HuggingFace models using the example scripts provided in the Model Optimizer repository. The primary script for LLM PTQ is typically found within the `examples/llm_ptq` directory.

Below is an example showing how to quantize a model using modelopt's PTQ API:

??? code

    ```python
    import modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM

    # Load the model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained("<path_or_model_id>")

    # Select the quantization config, for example, FP8
    config = mtq.FP8_DEFAULT_CFG

    # Define a forward loop function for calibration
    def forward_loop(model):
        for data in calib_set:
            model(data)

    # PTQ with in-place replacement of quantized modules
    model = mtq.quantize(model, config, forward_loop)
    ```

After the model is quantized, you can export it to a quantized checkpoint using the export API:

```python
import torch
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir,  # The directory where the exported files will be stored.
    )
```

The quantized checkpoint can then be deployed with vLLM. As an example, the following code shows how to deploy `nvidia/Llama-3.1-8B-Instruct-FP8`, which is the FP8 quantized checkpoint derived from `meta-llama/Llama-3.1-8B-Instruct`, using vLLM:

??? code

    ```python
    from vllm import LLM, SamplingParams

    def main():
        model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"

        # Ensure you specify quantization="modelopt" when loading the modelopt checkpoint
        llm = LLM(model=model_id, quantization="modelopt", trust_remote_code=True)

        sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    if __name__ == "__main__":
        main()
    ```

## Running the OpenAI-compatible server

To serve a local ModelOpt checkpoint via the OpenAI-compatible API:

```bash
vllm serve <path_to_exported_checkpoint> \
  --quantization modelopt \
  --host 0.0.0.0 --port 8000
```

## Testing (local checkpoints)

vLLM's ModelOpt unit tests are gated by local checkpoint paths and are skipped
by default in CI. To run the tests locally:

```bash
export VLLM_TEST_MODELOPT_FP8_PC_PT_MODEL_PATH=<path_to_fp8_pc_pt_checkpoint>
export VLLM_TEST_MODELOPT_FP8_PB_WO_MODEL_PATH=<path_to_fp8_pb_wo_checkpoint>
pytest -q tests/quantization/test_modelopt.py
```
