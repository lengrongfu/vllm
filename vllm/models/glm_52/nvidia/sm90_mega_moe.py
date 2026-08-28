# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_mega_moe_sm90 import (
    DeepGemmMegaMoESM90Experts,
    validate_sm90_mega_moe_topology,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    fused_grouped_topk,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLP,
    _get_moe_router_dtype,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk


def _validate_glm52_mega_moe_config(vllm_config: VllmConfig, config) -> None:
    parallel_config = vllm_config.parallel_config
    quant_config = vllm_config.quant_config
    architectures = getattr(config, "architectures", None) or []
    is_glm52_main = config.model_type == "glm_moe_dsa"
    is_glm52_mtp = (
        config.model_type == "deepseek_mtp" and "Glm52MTPModel" in architectures
    )
    if not (is_glm52_main or is_glm52_mtp):
        raise NotImplementedError(
            "GLM-5.2 SM90 MegaMoE only supports glm_moe_dsa checkpoints "
            "and their Glm52MTPModel draft configuration."
        )
    if not parallel_config.enable_expert_parallel:
        raise NotImplementedError(
            "GLM-5.2 SM90 MegaMoE requires --enable-expert-parallel."
        )
    if (
        parallel_config.tensor_parallel_size > 1
        and parallel_config.pipeline_parallel_size != 1
    ):
        raise NotImplementedError(
            "GLM-5.2 SM90 MegaMoE requires pipeline_parallel_size == 1 "
            "when tensor_parallel_size > 1."
        )
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise NotImplementedError(
            "GLM-5.2 SM90 MegaMoE requires bfloat16 model activations."
        )
    if not isinstance(quant_config, Fp8Config):
        raise NotImplementedError(
            "GLM-5.2 SM90 MegaMoE requires a serialized FP8 checkpoint."
        )
    if not quant_config.is_checkpoint_fp8_serialized:
        raise NotImplementedError(
            "GLM-5.2 SM90 MegaMoE does not support online FP8 quantization."
        )
    if quant_config.weight_block_size != [128, 128]:
        raise NotImplementedError(
            "GLM-5.2 SM90 MegaMoE requires FP8 weight_block_size=[128, 128]."
        )
    if quant_config.activation_scheme != "dynamic":
        raise NotImplementedError(
            "GLM-5.2 SM90 MegaMoE requires dynamic activation quantization."
        )
    if config.hidden_act != "silu":
        raise NotImplementedError("GLM-5.2 SM90 MegaMoE requires SwiGLU.")
    if getattr(config, "topk_method", None) != "noaux_tc":
        raise NotImplementedError("GLM-5.2 SM90 MegaMoE requires noaux_tc routing.")
    if getattr(config, "scoring_func", None) != "sigmoid":
        raise NotImplementedError("GLM-5.2 SM90 MegaMoE requires sigmoid routing.")


class Glm52DeepGemmMegaMoE(nn.Module):
    """GLM-5.2 MoE block using the SM90 all-FP8 MegaMoE kernel."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        tp_size = get_tensor_model_parallel_world_size()
        _validate_glm52_mega_moe_config(vllm_config, config)

        self.is_sequence_parallel = tp_size > 1
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.top_k = config.num_experts_per_tok
        self.num_expert_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.renormalize = config.norm_topk_prob
        self.scoring_func = config.scoring_func

        router_dtype = _get_moe_router_dtype(config)
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=router_dtype,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )

        self.shared_experts = None
        if self.n_shared_experts:
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * self.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )

        validate_sm90_mega_moe_topology()
        ep_group = get_ep_group()
        self.ep_size = ep_group.world_size
        self.ep_rank = ep_group.rank_in_group
        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        if self.n_physical_experts % self.ep_size != 0:
            raise ValueError(
                f"n_physical_experts={self.n_physical_experts} must be divisible "
                f"by ep_size={self.ep_size}. Adjust num_redundant_experts."
            )
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = self.ep_rank * self.n_local_physical_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_physical_experts
        self.physical_expert_start = self.experts_start_idx
        self.physical_expert_end = self.experts_end_idx

        self.experts = DeepGemmMegaMoESM90Experts(
            vllm_config,
            num_experts=self.n_physical_experts,
            num_local_experts=self.n_local_physical_experts,
            experts_start_idx=self.experts_start_idx,
            num_logical_experts=self.n_logical_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_shared_experts=self.n_shared_experts,
            fuse_shared_experts=True,
            prefix=f"{prefix}.experts",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            e_score_correction_bias=self.gate.e_score_correction_bias.data,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        output = self.experts(hidden_states, topk_weights, topk_ids)
        if (
            self.shared_experts is not None
            and not self.experts.has_fused_shared_experts
        ):
            output += self.shared_experts(hidden_states)

        if self.is_sequence_parallel and not already_sequence_parallel:
            output = tensor_model_parallel_all_gather(output, 0)[:num_tokens]
        return output.view(num_tokens, hidden_size)

    def finalize_mega_moe_weights(self) -> None:
        self.experts.finalize_weights(self.shared_experts)
