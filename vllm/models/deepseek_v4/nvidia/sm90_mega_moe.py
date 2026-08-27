# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe_sm90 import (
    DeepGemmMegaMoESM90Experts,
)

DeepseekV4MegaMoESM90Experts = DeepGemmMegaMoESM90Experts

__all__ = ["DeepseekV4MegaMoESM90Experts"]
