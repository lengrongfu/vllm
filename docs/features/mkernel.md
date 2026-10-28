# Experimental mKernel GEMM + AllReduce adapter

`VLLM_MKERNEL_CONFIG` enables a **disabled-by-default, correctness-only prototype**
that replaces one selected BF16 `RowParallelLinear` matrix multiplication and
tensor-parallel all-reduce with mKernel's fused `gemm_ar_multinode` operation.
It is not a general NCCL replacement, and it does not integrate mKernel's MoE,
attention, AllGather, or ReduceScatter kernels.

!!! warning
    The native extension and multi-node inference path have not been validated by
    this prototype's CPU tests. Do not deploy this on a production serving job.
    The adapter allocates communication buffers, packs weights, creates an RDMA
    session, synchronizes, copies its output, and tears down the session on every
    admitted call. It makes **no acceleration claim** and may be substantially
    slower than normal vLLM.

## Supported experimental configuration

- Exactly two hosts, each with eight visible Hopper GPUs, compute capability
  9.0 and 132 SMs, with NVLink multicast and the required network transport.
- Tensor parallel size 16, the full world group, with node-major ranks 0–7 and
  8–15. Each worker's CUDA device index must equal its local rank. A launcher
  that remaps every worker to device 0 is unsupported. Use one engine lifecycle
  per worker process; restarting an enabled engine requires fresh processes
  because the adapter owns process-global native and environment state.
- A matching mKernel `gemm_ar` build with `INTRA_NUM_DEVICES=8`, the CX7 or EFA
  backend, and the companion `get_build_config` / `get_scratch_ints` exports.
  Unpatched upstream binaries do not provide this prototype ABI.
  The native session must retain the default four QPs.
- Eager execution (`enforce_eager=True` / `--enforce-eager`), with no pipeline,
  data, expert, or context parallelism; no LoRA, speculative decoding, batch
  invariance, compilation, CUDA graph capture, dual batch overlap (DBO),
  microbatching, or concurrent calls.
- One exact layer prefix and one exact `[M, N, local_K]` shape. Inputs and
  unquantized weights must be contiguous CUDA BF16 tensors. The input must be
  two-dimensional, output reduction must be enabled, and gradients must be off.
  Bias is supported only when bias addition is deferred by the layer.

The prototype admits `M` multiples of 1024 up to 32768, `N` multiples of 256 up
to 32768, and `local_K` multiples of 64 up to 16384. One BF16 output buffer may
not exceed 512 MiB. Multiple such buffers plus network staging are allocated;
the limit is not an estimate of total additional GPU memory.

## Enabling it

Build the companion patched mKernel source separately on the target cluster,
using the same Python/PyTorch environment as vLLM. The native patch targets
[mKernel commit `ef6225fdde493f4119f519c99bfa4ad9234e323b`](https://github.com/uccl-project/mKernel/tree/ef6225fdde493f4119f519c99bfa4ad9234e323b).
It is included as
[`mkernel-ef6225fd-integration-api.patch`](mkernel/mkernel-ef6225fd-integration-api.patch),
alongside the upstream [MIT license](mkernel/mKernel-MIT.txt).
The patch adds the build-contract and exact scratch-size query exports required
by this adapter; an unpatched upstream binary is not compatible.

From this vLLM checkout, clone a separate native source tree and apply the patch:

```bash
VLLM_ROOT="$(pwd)"
git clone https://github.com/uccl-project/mKernel.git /absolute/path/to/patched-mkernel
git -C /absolute/path/to/patched-mkernel checkout \
  ef6225fdde493f4119f519c99bfa4ad9234e323b
git -C /absolute/path/to/patched-mkernel apply --check \
  "$VLLM_ROOT/docs/features/mkernel/mkernel-ef6225fd-integration-api.patch"
git -C /absolute/path/to/patched-mkernel apply \
  "$VLLM_ROOT/docs/features/mkernel/mkernel-ef6225fd-integration-api.patch"
```

Then build from that patched source checkout:

```bash
cd /absolute/path/to/patched-mkernel
make BACKEND=cx7 INTRA_NUM_DEVICES=8 \
  PYTHON=/absolute/path/to/.venv/bin/python build/libgemm_ar.so
```

Use `BACKEND=efa` instead on the supported AWS EFA installation. mKernel's
Hopper build defaults to CUDA 12.9; its network development libraries must also
be installed. Review and trust the native extension before loading it.

Set the same JSON value on **every worker on both hosts** before startup:

```bash
export VLLM_MKERNEL_CONFIG='{
  "root": "/absolute/path/to/patched-mkernel",
  "prefix": "model.layers.0.mlp.down_proj",
  "shape": [1024, 4096, 896],
  "node_ips": ["192.0.2.1", "192.0.2.2"],
  "broker_key": "unique-mkernel-job-20261027",
  "port": 19730
}'
```

Replace the documentation IP addresses with routable data-plane IPv4 literals
(hostnames and IPv6 addresses are not supported by this prototype).
`node_ips` follows the node-major rank order; ranks on the two hosts pair using
the same local rank. `port` is a base TCP port, with eight consecutive ports
reserved. `root` must be the same absolute path on both hosts and contain
`build/libgemm_ar.so`. Choose `prefix` and `shape` from the actual model: the
example shape matches a 4096-wide output with local intermediate width 896,
not every model or batch.

Use a unique `broker_key` for the job, containing 1–40 letters, digits, `_`, or
`-`. The adapter sets `DIST_BROKER_KEY` for mKernel's host-local IPC broker; an
already-set conflicting value is rejected. Run on exclusive hosts. Do not set
`GEMM_AR_*`, `MKERNEL_*`, `Q2_*`, or `Q5_*` tuning variables: the prototype rejects them and
manages its required retained-allocation-handle setting itself.

!!! important
    All workers must receive identical configuration. Enabled workers compare
    their settings collectively, but a worker with the variable missing does
    not enter that collective. A partially enabled launch is unsupported and
    can hang; verify environment propagation before starting the job.

With the variable unset, vLLM follows its original path. Other layer prefixes,
other token counts, incompatible tensor contracts, or a collectively unavailable
binary use the original linear method and all-reduce before any mKernel session
is created. Once native initialization or execution starts, failures are fatal:
switching one rank to NCCL at that point could deadlock its peers. A native
exception terminates that worker immediately rather than invoking native
collective destructors on just one rank. The launcher or supervisor must
terminate all remaining ranks, and recovery requires restarting the full job.
Use an external timeout as native waits can hang without raising an exception.

The selected layer still performs a CPU-group metadata collective when its
token count or tensor contract falls back. Thus enabled decode fallback is
**not zero-overhead**. Only unsetting the variable preserves the original path
without the adapter's additional collectives.

## Validation still required on the target cluster

The CPU policy checks can run without importing PyTorch or the extension:

```bash
.venv/bin/python tests/distributed/test_mkernel.py -v
```

They validate configuration, tensor admission, binary metadata, rank agreement,
and the adapter's owned control flow against recording external CUDA/native API
services. A subprocess check verifies fail-stop behavior on an injected native
exception. They do not prove a real kernel launch, numerical correctness, CUDA
stream ordering, distributed lifecycle safety, or model accuracy.

Before treating this as usable, run these checks with a timeout across all
16 ranks and preserve the logs and exact repository revisions:

1. Compile the patched native extension and run mKernel's existing correctness
   test for the selected shape and transport. Its benchmark tolerance is not
   sufficient evidence of language-model accuracy.
2. Compare the adapter's real layer output against the original BF16 linear
   plus tensor-parallel sum for multiple deterministic inputs. Check every
   rank, repeated invocations, alternating admitted and non-admitted token
   counts, and session teardown. Derive tolerances from the numeric contract;
   do not loosen them to make the test pass.
3. Use the public `LLM` entry point with local, reviewed model weights,
   `distributed_executor_backend="external_launcher"`, `tensor_parallel_size=16`,
   `pipeline_parallel_size=1`, `dtype="bfloat16"`, `enforce_eager=True`, and a
   fixed seed. Run the same deterministic prompts with the variable unset and
   set in separate processes. Set token budgets to actually exercise the
   configured `M`; ordinary short decode batches deliberately fall back.
   Require the post-success log `mKernel GEMM+AR executed: prefix=... shape=...
   backend=...` as evidence that the native path actually ran; successful engine
   startup alone is not evidence of an admitted call.
   Compare generated tokens and log probabilities, and run the relevant
   model evaluation suite. The existing
   `examples/features/torchrun/torchrun_example_offline.py` shows the launcher
   pattern, but its model/parallelism settings must be adapted to this contract.
4. Confirm missing or incompatible binaries cause all ranks to fall back,
   supported layers preserve deferred-bias behavior, and unsupported topology
   is rejected before native work. Test failure shutdown under an external
   timeout; do not attempt recovery by mixing NCCL and mKernel ranks.

Only after correctness and model evaluations pass should a separate persistent
session/buffer design be benchmarked for prefill throughput, time to first
token, tail latency, GPU memory overhead, and decode regressions.
