# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
    grouped_topk,
)
from vllm.model_executor.layers.fused_moe.router.mega_gate_router import MegaGateRouter
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLP,
    _get_moe_router_dtype,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from .ops.prepare_nvfp4_megamoe import prepare_nvfp4_megamoe_inputs

logger = init_logger(__name__)


class DeepseekV32MegaMoEExperts(nn.Module):
    """Model-owned adapter for DeepGEMM's SM90 NVFP4 MegaMoE kernel."""

    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], Any] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.w13_weight: Any
        self.w13_weight_scale: Any
        self.w13_weight_scale_2: Any
        self.w13_input_scale: Any
        self.w2_weight: Any
        self.w2_weight_scale: Any
        self.w2_weight_scale_2: Any
        self.w2_input_scale: Any
        weight_attrs = {"weight_loader": self.weight_loader}
        self._register_weight(
            "w13_weight",
            (num_local_experts, 2 * intermediate_size, hidden_size // 2),
            torch.uint8,
            weight_attrs,
        )
        self._register_weight(
            "w13_weight_scale",
            (num_local_experts, 2 * intermediate_size, hidden_size // 16),
            torch.float8_e4m3fn,
            weight_attrs,
        )
        self._register_weight(
            "w13_weight_scale_2",
            (num_local_experts, 2),
            torch.float32,
            weight_attrs,
        )
        self._register_weight(
            "w13_input_scale",
            (num_local_experts, 2),
            torch.float32,
            weight_attrs,
        )
        self._register_weight(
            "w2_weight",
            (num_local_experts, hidden_size, intermediate_size // 2),
            torch.uint8,
            weight_attrs,
        )
        self._register_weight(
            "w2_weight_scale",
            (num_local_experts, hidden_size, intermediate_size // 16),
            torch.float8_e4m3fn,
            weight_attrs,
        )
        self._register_weight(
            "w2_weight_scale_2",
            (num_local_experts,),
            torch.float32,
            weight_attrs,
        )
        self._register_weight(
            "w2_input_scale",
            (num_local_experts,),
            torch.float32,
            weight_attrs,
        )

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._l1_global_scales: torch.Tensor | None = None
        self._l2_global_scales: torch.Tensor | None = None

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _register_weight(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        attrs: dict[str, Any],
    ) -> None:
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        set_weight_attrs(param, attrs)
        self.register_parameter(name, param)

    def _local_expert_id(self, expert_id: int) -> int | None:
        if self.experts_start_idx <= expert_id < self.experts_end_idx:
            return expert_id - self.experts_start_idx
        return None

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_id = self._local_expert_id(expert_id)
        if local_expert_id is None:
            return False if return_success else None

        expert_data = param.data[local_expert_id]
        if shard_id in ("w1", "w3"):
            if "w13_" not in weight_name:
                return False if return_success else None
            shard_idx = 0 if shard_id == "w1" else 1
            if expert_data.ndim == 1:
                expert_data = expert_data[shard_idx]
            else:
                expert_data = expert_data.narrow(
                    0, shard_idx * self.intermediate_size, self.intermediate_size
                )
        elif shard_id == "w2":
            if "w2_" not in weight_name:
                return False if return_success else None
        else:
            raise ValueError(f"Unsupported expert shard id: {shard_id}")

        loaded_weight = loaded_weight.to(expert_data.device)
        if expert_data.numel() == 1:
            expert_data.copy_(loaded_weight.reshape(()))
        elif expert_data.shape != loaded_weight.shape:
            raise ValueError(
                f"MegaMoE weight shape mismatch for {weight_name}: "
                f"{tuple(expert_data.shape)} vs {tuple(loaded_weight.shape)}"
            )
        else:
            expert_data.copy_(loaded_weight)
        return True if return_success else None

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return
        if not envs.VLLM_USE_DEEP_GEMM:
            raise RuntimeError(
                "deep_gemm_mega_moe was selected while VLLM_USE_DEEP_GEMM=0."
            )
        if not current_platform.is_device_capability(90):
            raise RuntimeError("DeepseekV32 NVFP4 MegaMoE requires an SM90 GPU.")

        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        required_apis = (
            "transform_nvfp4_weights_for_mega_moe_sm90",
            "get_symm_buffer_for_mega_moe",
            "nvfp4_mega_moe",
        )
        if any(not hasattr(deep_gemm, api) for api in required_apis):
            raise RuntimeError(
                "Installed DeepGEMM does not provide the SM90 NVFP4 MegaMoE APIs."
            )

        self._transformed_l1_weights, self._transformed_l2_weights = (
            deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
                (
                    self.w13_weight.data.contiguous(),
                    self.w13_weight_scale.data.view(torch.uint8).contiguous(),
                ),
                (
                    self.w2_weight.data.contiguous(),
                    self.w2_weight_scale.data.view(torch.uint8).contiguous(),
                ),
            )
        )
        w13_scale = self.w13_weight_scale_2.data.float().reshape(-1, 2)
        if not torch.allclose(w13_scale[:, 0], w13_scale[:, 1]):
            logger.warning_once(
                "w1_weight_scale_2 and w3_weight_scale_2 differ; SM90 "
                "MegaMoE uses the w1 value for both projections."
            )
        self._l1_global_scales = w13_scale[:, 0].contiguous()
        self._l2_global_scales = (
            self.w2_weight_scale_2.data.float().reshape(-1).contiguous()
        )

        self.w13_weight = None
        self.w13_weight_scale = None
        self.w13_weight_scale_2 = None
        self.w13_input_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        self.w2_weight_scale_2 = None
        self.w2_input_scale = None
        logger.info_once(
            "Using model-owned DeepGEMM SM90 NVFP4 MegaMoE experts.",
            scope="global",
        )

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        group = get_ep_group().device_group
        device = torch.accelerator.current_device_index()
        key = (
            id(group),
            device,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = _import_deep_gemm().get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                mma_type="fp4xfp4",
                activation="swiglu",
            )
            buffer_nbytes = (
                symm_buffer.buffer.numel() * symm_buffer.buffer.element_size()
            )
            logger.info(
                "Allocated DeepGEMM MegaMoE symmetric buffer: %.2f GiB "
                "(%d bytes) on cuda:%d.",
                buffer_nbytes / (1 << 30),
                buffer_nbytes,
                device,
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    @property
    def layer_id(self) -> int:
        return extract_layer_index(self.prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"NVFP4 MegaMoE got {hidden_states.shape[0]} tokens, but its "
                f"capacity is {self.max_num_tokens}."
            )
        self.finalize_weights()
        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        prepare_nvfp4_megamoe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            is_padding,
        )
        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        y = torch.empty_like(hidden_states)

        from vllm.utils.deep_gemm import _import_deep_gemm

        _import_deep_gemm().nvfp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            l1_global_scales=self._l1_global_scales,
            l2_global_scales=self._l2_global_scales,
            fast_math=True,
        )
        return y


DeepseekV32MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


class DeepseekV32MegaMoE(nn.Module):
    """DeepSeek-V3.2/GLM MoE block backed by model-owned MegaMoE experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        prefix: str,
        use_sequence_parallel: bool,
    ) -> None:
        super().__init__()
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        if not parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepseekV32 MegaMoE requires --enable-expert-parallel."
            )
        if parallel_config.enable_eplb:
            raise NotImplementedError("DeepseekV32 MegaMoE does not support EPLB yet.")
        if parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("DeepseekV32 MegaMoE does not support PP yet.")
        if quant_config is None or quant_config.get_name() != "modelopt_fp4":
            raise ValueError("DeepseekV32 MegaMoE requires a ModelOpt NVFP4 model.")
        if getattr(quant_config, "quant_method", None) != "NVFP4":
            raise ValueError("DeepseekV32 MegaMoE requires W4A4 NVFP4 experts.")
        if config.hidden_act != "silu":
            raise ValueError("DeepseekV32 MegaMoE supports SiLU only.")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("DeepseekV32 MegaMoE requires BF16 activations.")
        if getattr(vllm_config, "weight_transfer_config", None) is not None:
            raise NotImplementedError(
                "DeepseekV32 MegaMoE does not support runtime weight transfer."
            )
        if (
            config.hidden_size != 6144
            or config.moe_intermediate_size != 2048
            or config.num_experts_per_tok != 8
            or config.n_routed_experts > 512
        ):
            raise ValueError(
                "SM90 NVFP4 MegaMoE supports hidden_size=6144, "
                "moe_intermediate_size=2048, top_k=8, and at most 512 experts."
            )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.is_sequence_parallel = use_sequence_parallel
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts

        ep_group = get_ep_group()
        ep_size = ep_group.world_size
        ep_rank = ep_group.rank_in_group
        if self.n_physical_experts % ep_size:
            raise ValueError(
                f"n_routed_experts={self.n_physical_experts} must be divisible "
                f"by EP size {ep_size}."
            )
        self.n_local_physical_experts = self.n_physical_experts // ep_size
        experts_start_idx = ep_rank * self.n_local_physical_experts

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=_get_moe_router_dtype(config),
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        self.num_expert_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.mega_gate: MegaGateRouter | None = None
        if (
            bool(int(os.getenv("VLLM_GLM_USE_MEGA_GATE", "0")))
            and config.model_type == "glm_moe_dsa"
        ):
            router = GroupedTopKRouter(
                top_k=self.top_k,
                global_num_experts=self.n_routed_experts,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                renormalize=self.renormalize,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.gate.e_score_correction_bias,
            )
            if MegaGateRouter.is_supported(router, self.gate):
                mega_gate = MegaGateRouter(router, self.gate)
                if mega_gate.available:
                    self.mega_gate = mega_gate
                    logger.info_once(
                        "Enabled DeepGEMM Mega-Gate for GLM NVFP4 MegaMoE."
                    )
                else:
                    logger.warning_once(
                        "GLM Mega-Gate requested but unavailable in this DeepGEMM "
                        "build for the current GPU; using the original router."
                    )
        self.experts = DeepseekV32MegaMoEExperts(
            vllm_config,
            num_experts=self.n_physical_experts,
            num_local_experts=self.n_local_physical_experts,
            experts_start_idx=experts_start_idx,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
        )

        self.shared_experts: DeepseekV2MLP | None
        if self.n_shared_experts:
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * self.n_shared_experts
                ),
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                is_sequence_parallel=use_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if self.is_sequence_parallel and not already_sequence_parallel:
            raise RuntimeError("MegaMoE expects model-level sequence-parallel input.")
        org_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, org_shape[-1])
        if self.mega_gate is not None:
            topk_weights, topk_ids = self.mega_gate.select_experts(
                hidden_states, hidden_states, torch.int64
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.gate.e_score_correction_bias,
            )
        output = self.experts(hidden_states, topk_weights, topk_ids)
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            output += shared_output
        return output.view(org_shape)

    def finalize_mega_moe_weights(self) -> None:
        self.experts.finalize_weights()

    def update_expert_map(self) -> None:
        raise NotImplementedError("DeepseekV32 MegaMoE does not support EPLB yet.")
