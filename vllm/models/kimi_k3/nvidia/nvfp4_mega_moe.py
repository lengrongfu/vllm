# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 NVFP4xNVFP4 adapter for DeepGEMM's `fp4_fp4_mega_moe` kernel
(deepseek-ai/DeepGEMM#409).

Unlike `KimiK3MegaMoEExperts` (which dequantizes to FP8 activations against
MXFP4-style, block-32/UE8M0 weights via `fp8_fp4_mega_moe`), this module
consumes the checkpoint's native NVFP4 format as published by
nvidia/Kimi-K3-NVFP4: E2M1 weights and activations with per-16-element E4M3
block scales, plus a per-tensor FP32 global scale
(`weight_global_scale` / `input_global_scale`, compressed-tensors'
`modelopt_mixed` scheme).
"""

from collections.abc import Callable

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.kimi_k3.nvidia.ops.prepare_megamoe_nvfp4 import (
    prepare_megamoe_nvfp4_inputs,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


def make_kimi_k3_nvfp4_mega_moe_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    """Maps compressed-tensors' per-expert NVFP4 keys onto the fused
    (w13/w2) parameters this module registers.

    Checkpoint tensor names per expert/shard:
      `experts.{id}.{w1,w2,w3}.weight_packed`
      `experts.{id}.{w1,w2,w3}.weight_scale`
      `experts.{id}.{w1,w2,w3}.weight_global_scale`
    """
    mapping = []
    for expert_id in range(num_experts):
        for shard_id in ("w1", "w2", "w3"):
            param_prefix = "w13" if shard_id in ("w1", "w3") else "w2"
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale"):
                param_suffix = {
                    "weight_packed": "weight",
                    "weight_scale": "weight_scale",
                    "weight_global_scale": "weight_global_scale",
                }[suffix]
                mapping.append(
                    (
                        f"experts.{param_prefix}_{param_suffix}",
                        f"experts.{expert_id}.{shard_id}.{suffix}",
                        expert_id,
                        shard_id,
                    )
                )
    return mapping


class KimiK3NVFP4MegaMoEExperts(nn.Module):
    """Kimi K3 adapter for DeepGEMM's NVFP4xNVFP4 MegaMoE kernel."""

    _symm_buffer_cache: dict[tuple[object, ...], object] = {}
    _synchronized_ep_groups: set[tuple[int, int]] = set()

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
        num_logical_experts: int | None = None,
        activation: str = "situ",
        activation_beta: float | None = None,
        activation_linear_beta: float | None = None,
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
        self.num_logical_experts = num_logical_experts or num_experts
        self.activation = activation
        self.activation_beta = activation_beta
        self.activation_linear_beta = activation_linear_beta
        self.eplb_state = EplbLayerState()

        if hidden_size % 16 or intermediate_size % 16:
            raise ValueError(
                "Kimi K3 NVFP4 MegaMoE requires hidden_size and "
                "intermediate_size to be multiples of 16 (NVFP4 block size)."
            )

        weight_attrs = {"weight_loader": self.weight_loader}

        def _make_weight_group(rows: int, cols: int) -> tuple[nn.Parameter, ...]:
            packed = nn.Parameter(
                torch.zeros(num_local_experts, rows, cols // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            scale = nn.Parameter(
                torch.zeros(
                    num_local_experts, rows, cols // 16, dtype=torch.float8_e4m3fn
                ),
                requires_grad=False,
            )
            global_scale = nn.Parameter(
                torch.zeros(num_local_experts, rows, dtype=torch.float32),
                requires_grad=False,
            )
            for p in (packed, scale, global_scale):
                set_weight_attrs(p, weight_attrs)
            return packed, scale, global_scale

        self.w13_weight, self.w13_weight_scale, self.w13_weight_global_scale = (
            _make_weight_group(2 * intermediate_size, hidden_size)
        )
        self.w2_weight, self.w2_weight_scale, self.w2_weight_global_scale = (
            _make_weight_group(hidden_size, intermediate_size)
        )
        # Static activation quant scale (`input_activations.dynamic == false`
        # in the checkpoint's quantization_config); one scalar per expert,
        # broadcast across the w13/w2 GEMMs it feeds.
        self.input_global_scale = nn.Parameter(
            torch.zeros(num_local_experts, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(self.input_global_scale, weight_attrs)

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._l1_alphas: torch.Tensor | None = None
        self._l2_alphas: torch.Tensor | None = None

        static_context = vllm_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        return [
            physical_id - self.experts_start_idx
            for physical_id in range(self.experts_start_idx, self.experts_end_idx)
            if physical_id % self.num_logical_experts == expert_id
        ]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_ids = self._map_global_expert_id(expert_id)
        if not local_expert_ids:
            return False if return_success else None

        is_w13 = shard_id in ("w1", "w3")
        loaded_any = False
        for local_expert_id in local_expert_ids:
            expert_data = param.data[local_expert_id]
            if is_w13:
                if "w13_" not in weight_name:
                    continue
                shard_size = expert_data.shape[0] // 2
                shard_offset = 0 if shard_id == "w1" else shard_size
                expert_data = expert_data.narrow(0, shard_offset, shard_size)
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")

            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"Kimi K3 NVFP4 MegaMoE weight shape mismatch for "
                    f"{weight_name}: expected {tuple(expert_data.shape)}, got "
                    f"{tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
            loaded_any = True
        return loaded_any if return_success else None

    def _check_runtime_supported(self) -> None:
        if self.w13_weight.device.type != "cuda":
            raise RuntimeError("NVFP4 MegaMoE weights must be loaded on CUDA.")
        if torch.cuda.get_device_capability(self.w13_weight.device)[0] != 10:
            raise NotImplementedError(
                "Kimi K3 NVFP4 MegaMoE requires an SM100 (Blackwell) GPU."
            )

    def synchronize_first_launch(self) -> None:
        ep_group = get_ep_group()
        device = torch.accelerator.current_device_index()
        key = (id(ep_group.cpu_group), device)
        if key in self._synchronized_ep_groups:
            return
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=ep_group.cpu_group)
        self._synchronized_ep_groups.add(key)

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return
        self._check_runtime_supported()
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        if deep_gemm is None or not hasattr(deep_gemm, "fp4_fp4_mega_moe"):
            raise RuntimeError(
                "DeepGEMM's fp4_fp4_mega_moe API is required (DeepGEMM#409); "
                "the installed DeepGEMM build does not provide it."
            )

        # Global (per-expert) alpha = weight_global_scale * input_global_scale,
        # applied by the kernel to dequantize the accumulated GEMM output.
        self._l1_alphas = (
            self.w13_weight_global_scale.amax(dim=1) * self.input_global_scale
        ).contiguous()
        self._l2_alphas = (
            self.w2_weight_global_scale.amax(dim=1) * self.input_global_scale
        ).contiguous()

        self._transformed_l1_weights, self._transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                (
                    self.w13_weight.data.contiguous(),
                    self.w13_weight_scale.data.contiguous(),
                ),
                (
                    self.w2_weight.data.contiguous(),
                    self.w2_weight_scale.data.contiguous(),
                ),
                activation=self.activation,
            )
        )
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w13_weight_global_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None
        self.w2_weight_global_scale = None

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        group = get_ep_group().device_group
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.activation,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                mma_type="fp4xfp4",
                activation=self.activation,
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    def update_expert_map(self) -> None:
        pass

    @property
    def layer_id(self) -> int:
        return extract_layer_index(self.prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None = None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        self.synchronize_first_launch()
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"Kimi K3 NVFP4 MegaMoE got {num_tokens} tokens, but its "
                f"symmetric buffer supports {self.max_num_tokens}."
            )
        if num_tokens == 0:
            return hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)

        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        self.finalize_weights()
        symm_buffer = self.get_symm_buffer()
        # Static per-tensor input scale; broadcasting across local experts is
        # fine here since checkpoints in this format calibrate one shared
        # activation scale per MoE layer (`torch.unique(...).numel() == 1`
        # is asserted the same way vLLM's dense NVFP4 linear scheme does).
        input_global_scale = self.input_global_scale[:1]
        prepare_megamoe_nvfp4_inputs(
            hidden_states,
            input_global_scale,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            is_padding=is_padding,
        )

        deep_gemm = __import__("vllm.utils.deep_gemm", fromlist=["_import_deep_gemm"])
        deep_gemm = deep_gemm._import_deep_gemm()
        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        y = hidden_states.new_empty(hidden_states.shape, dtype=torch.bfloat16)
        deep_gemm.fp4_fp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            activation=self.activation,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            l1_alphas=self._l1_alphas,
            l2_alphas=self._l2_alphas,
            a2_scales=input_global_scale,
        )
        return y
