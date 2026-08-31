# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

_validated_ep_groups: set[int] = set()


def validate_sm90_mega_moe_topology(ep_group=None) -> None:
    if ep_group is None:
        ep_group = get_ep_group()
    group_key = id(ep_group.cpu_group)
    if group_key in _validated_ep_groups:
        return

    visible_device_id = current_platform.logical_device_id_to_visible_device_id(
        ep_group.device_index
    )
    physical_device_id = current_platform.visible_device_id_to_physical_device_id(
        visible_device_id
    )
    device_id = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
    gathered_device_ids = [
        torch.zeros(1, dtype=torch.int, device="cpu")
        for _ in range(ep_group.world_size)
    ]
    torch.distributed.all_gather(
        gathered_device_ids, device_id, group=ep_group.cpu_group
    )
    physical_device_ids = [device.item() for device in gathered_device_ids]
    if not current_platform.is_fully_connected(physical_device_ids):
        raise RuntimeError(
            "SM90 DeepGEMM MegaMoE requires a fully connected NVLink expert "
            f"parallel group, got physical GPUs {physical_device_ids}."
        )
    _validated_ep_groups.add(group_key)


class DeepGemmMegaMoESM90Experts(nn.Module):
    """Hopper FP8 routed experts backed by DeepGEMM MegaMoE."""

    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], object] = {}

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
        prefix: str = "",
        num_logical_experts: int | None = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.block_size = 128
        self.num_logical_experts = num_logical_experts or num_experts
        self.eplb_state = EplbLayerState()

        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)
        self.w13_weight_scale_inv = nn.Parameter(
            torch.zeros(
                num_local_experts,
                cdiv(2 * intermediate_size, self.block_size),
                cdiv(hidden_size, self.block_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale_inv, weight_attrs)
        self.w13_weight_scale_inv.quant_method = "block"

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)
        self.w2_weight_scale_inv = nn.Parameter(
            torch.zeros(
                num_local_experts,
                cdiv(hidden_size, self.block_size),
                cdiv(intermediate_size, self.block_size),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale_inv, weight_attrs)
        self.w2_weight_scale_inv.quant_method = "block"

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cum_local_expert_recv_stats: torch.Tensor | None = None

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

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

        is_scale = "weight_scale" in weight_name
        loaded_any = False
        for local_expert_id in local_expert_ids:
            expert_data = param.data[local_expert_id]
            if shard_id in ("w1", "w3"):
                if "w13_" not in weight_name:
                    continue
                dim0 = (
                    2 * self.intermediate_size
                    if not is_scale
                    else cdiv(2 * self.intermediate_size, self.block_size)
                )
                shard_len = dim0 // 2
                shard_offset = 0 if shard_id == "w1" else shard_len
                expert_data = expert_data.narrow(0, shard_offset, shard_len)
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")

            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"SM90 MegaMoE weight shape mismatch for {weight_name}: "
                    f"{tuple(expert_data.shape)} vs {tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
            loaded_any = True
        return loaded_any if return_success else None

    def _check_runtime_supported(self) -> None:
        if torch.cuda.get_device_capability(self.w13_weight.device) != (9, 0):
            raise NotImplementedError("DeepGEMM FP8 MegaMoE requires SM90 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "SM90 MegaMoE requires hidden and intermediate sizes to be "
                "multiples of 128."
            )
        if self.intermediate_size > 4096:
            raise NotImplementedError(
                "SM90 MegaMoE requires intermediate_size <= 4096, got "
                f"{self.intermediate_size}."
            )
        from vllm.utils.deep_gemm import deep_gemm_supports_sm90_mega_moe

        if not deep_gemm_supports_sm90_mega_moe():
            raise RuntimeError(
                "The installed DeepGEMM build does not expose the SM90 FP8 MegaMoE API."
            )

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return
        self._check_runtime_supported()
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        assert deep_gemm is not None
        self._transformed_l1_weights, self._transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe_sm90(
                (self.w13_weight.data, self.w13_weight_scale_inv.data),
                (self.w2_weight.data, self.w2_weight_scale_inv.data),
            )
        )
        self._cum_local_expert_recv_stats = torch.zeros(
            self.num_local_experts,
            dtype=torch.int32,
            device=self.w13_weight.device,
        )
        self.w13_weight = None
        self.w2_weight = None

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        assert deep_gemm is not None
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
            symm_buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
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

    def get_expert_weights(self) -> list[torch.Tensor]:
        self.finalize_weights()
        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None

        def to_eplb_view(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.is_contiguous():
                return tensor.view(self.num_local_experts, -1)
            contiguous = torch.transpose(tensor, 1, 2)
            if not contiguous.is_contiguous():
                raise AssertionError(f"Unexpected MegaMoE layout: {tensor.stride()}")
            return contiguous.view(self.num_local_experts, -1)

        return [
            to_eplb_view(self._transformed_l1_weights[0]),
            to_eplb_view(self._transformed_l1_weights[1]),
            to_eplb_view(self._transformed_l2_weights[0]),
            to_eplb_view(self._transformed_l2_weights[1]),
        ]

    def update_expert_map(self) -> None:
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"SM90 MegaMoE got {hidden_states.shape[0]} tokens, but the "
                f"buffer supports {self.max_num_tokens}."
            )
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        assert deep_gemm is not None
        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]
        if is_padding is not None:
            topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            topk_weights = torch.where(is_padding.unsqueeze(1), 0.0, topk_weights)

        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
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

        x_fp8, x_sf = deep_gemm.utils.per_token_cast_to_fp8(
            hidden_states, use_ue8m0=False, gran_k=128
        )
        symm_buffer.x[:num_tokens].copy_(x_fp8)
        symm_buffer.x_sf[:num_tokens].copy_(x_sf)
        symm_buffer.topk_idx[:num_tokens].copy_(topk_ids.to(torch.int64))
        symm_buffer.topk_weights[:num_tokens].copy_(topk_weights.to(torch.float32))

        self.finalize_weights()
        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        deep_gemm.fp8_mega_moe_sm90(
            output,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            cumulative_local_expert_recv_stats=self._cum_local_expert_recv_stats,
            activation_clamp=activation_clamp,
        )
        return output


DeepGemmMegaMoESM90Experts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
