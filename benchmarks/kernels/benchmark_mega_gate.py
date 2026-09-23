# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare GLM routing kernels on Hopper using CUDA graphs and CUPTI."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm._custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm
from vllm.model_executor.layers.fused_moe.router.mega_gate_router import (
    is_mega_gate_shape_preferred,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _import_deep_gemm,
    bf16_mega_gate,
    is_mega_gate_supported,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=[1, 16, 17, 32, 64, 128, 256, 512, 1024, 4096, 8192],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", choices=["cold", "warm"], default="cold")
    parser.add_argument(
        "--raw-native", action="store_true", help="Bypass vLLM's shape fallback"
    )
    args = parser.parse_args()
    if not is_mega_gate_supported():
        raise RuntimeError("Install a DeepGEMM build supporting Mega-Gate on this GPU")
    deep_gemm = _import_deep_gemm()
    torch.manual_seed(42)
    rows = []
    with torch.inference_mode():
        for m in args.tokens:
            x = torch.randn(m, 6144, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
            b = torch.randn(256, device="cuda")

            def baseline(x, w, b):
                logits = (
                    ll_bf16_gemm(x, w)
                    if x.shape[0] <= 16
                    else torch.mm(x, w.T, out_dtype=torch.float32)
                )
                return ops.grouped_topk(logits, 1, 1, 8, True, 2.5, b, 1)

            def candidate(x, w, b):
                return bf16_mega_gate(
                    x,
                    w,
                    8,
                    scoring_func="sigmoid",
                    routed_scaling_factor=2.5,
                    ep_rank=0,
                    bias=b,
                )

            use_native = args.raw_native or is_mega_gate_shape_preferred(
                m, 6144, 256, 8, is_sm90=current_platform.is_device_capability(90)
            )
            selected = candidate if use_native else baseline
            a, i = selected(x, w, b)
            ref, j = baseline(x, w, b)
            ia, ja = i.argsort(1), j.argsort(1)
            torch.testing.assert_close(i.gather(1, ia).long(), j.gather(1, ja).long())
            torch.testing.assert_close(
                a.gather(1, ia), ref.gather(1, ja), rtol=1e-3, atol=1e-4
            )
            times = {"baseline_us": [], "candidate_us": []}
            for round_id in range(3):
                order = [("baseline_us", baseline), ("candidate_us", selected)]
                if round_id % 2:
                    order.reverse()
                for name, fn in order:
                    times[name].append(
                        statistics.median(
                            bench_gpu_time_with_cupti(
                                fn,
                                input_args=(x, w, b),
                                dry_run_iters=10,
                                repeat_iters=50,
                                use_cuda_graph=True,
                                cold_l2_cache=args.cache == "cold",
                            )
                        )
                        * 1000
                    )
            row = {"tokens": m, **{k: statistics.median(v) for k, v in times.items()}}
            row["speedup"] = row["baseline_us"] / row["candidate_us"]
            row["candidate_tflops"] = 2 * m * 6144 * 256 / (row["candidate_us"] * 1e6)
            row["rounds_us"] = times
            row["deep_gemm_config"] = dict(
                deep_gemm.get_bf16_mega_gate_config(m, 6144, 256, 8)
            )
            row["vllm_dispatch"] = "mega_gate" if use_native else "baseline"
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "BF16",
        "hidden_size": 6144,
        "experts": 256,
        "top_k": 8,
        "timing": f"CUPTI {args.cache} L2 CUDA graph, median of 3 alternating rounds",
        "raw_native": args.raw_native,
        "deep_gemm_path": deep_gemm.__file__,
        "deep_gemm_version": deep_gemm.__version__,
        "source_sha256": {
            str(path.relative_to(Path(deep_gemm.__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(deep_gemm._C.__file__),
                Path(deep_gemm.__file__).parent
                / "include"
                / "deep_gemm/impls/sm90_bf16_mega_gate.cuh",
            )
            if path.exists()
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("DG_SM90_MEGA_GATE_")
            or key
            in (
                "DG_JIT_USE_NVRTC",
                "DG_USE_TEMP_CUBLASLT_WORKSPACE",
                "VLLM_BATCH_INVARIANT",
                "VLLM_USE_DEEP_GEMM",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
            )
        },
        "rows": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
