# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.platforms import current_platform

if current_platform.is_cuda():
    from .nvidia.model import GlmMoeDsaForCausalLM
    from .nvidia.mtp import Glm52MTP
else:
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
    from vllm.models.deepseek_v32 import DeepseekV32MTP as Glm52MTP

__all__ = ["Glm52MTP", "GlmMoeDsaForCausalLM"]
