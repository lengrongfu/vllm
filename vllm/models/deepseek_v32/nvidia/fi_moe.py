# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ``moe_ep`` integration for GLM MoE DSA models."""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLP,
    _get_moe_router_dtype,
)
from vllm.models.deepseek_v4.nvidia.fi_moe import DeepseekV4MegaMoEExpertsFI
from vllm.utils.flashinfer_moe_ep import validate_fi_moe_ep_config


class GlmMoeDsaFlashInferMoE(nn.Module):
    _is_flashinfer_moe_ep = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        validate_fi_moe_ep_config(vllm_config)

        if not parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "GLM FlashInfer moe_ep requires expert parallel. Enable it "
                "with --enable-expert-parallel."
            )
        if getattr(config, "n_group", 1) != 1 or getattr(config, "topk_group", 1) != 1:
            raise NotImplementedError(
                "GLM FlashInfer moe_ep currently requires n_group=topk_group=1."
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.renormalize = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts

        ep_group = get_ep_group()
        ep_size = ep_group.world_size
        ep_rank = ep_group.rank_in_group
        assert self.n_physical_experts % ep_size == 0, (
            f"n_physical_experts={self.n_physical_experts} must be divisible by "
            f"ep_size={ep_size}."
        )
        self.n_local_physical_experts = self.n_physical_experts // ep_size
        experts_start_idx = ep_rank * self.n_local_physical_experts

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=_get_moe_router_dtype(config),
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) != "noaux_tc":
            raise NotImplementedError(
                "GLM FlashInfer moe_ep currently supports noaux_tc routing only."
            )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )

        if self.n_shared_experts:
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * self.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                reduce_results=False,
                is_sequence_parallel=use_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = DeepseekV4MegaMoEExpertsFI(
            vllm_config,
            num_experts=self.n_physical_experts,
            num_local_experts=self.n_local_physical_experts,
            experts_start_idx=experts_start_idx,
            num_logical_experts=self.n_logical_experts,
            top_k=self.n_activated_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
            activation_clamp=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        del already_sequence_parallel
        org_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, org_shape[-1])
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias.data,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=torch.int64,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        output = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
            activation_clamp=None,
        )
        if self.shared_experts is not None:
            output += self.shared_experts(hidden_states)
        return output.view(org_shape)

    def finalize_weights(self) -> None:
        self.experts.finalize_weights()
