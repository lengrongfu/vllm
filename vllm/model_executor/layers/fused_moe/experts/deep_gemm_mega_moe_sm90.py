# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import inspect

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
            "SM90 DeepGEMM MegaMoE requires every GPU in the expert-parallel "
            "group to be connected by NVLink with native peer atomics. The EP "
            f"group uses physical GPUs {physical_device_ids}, which are not in "
            "one fully connected NVLink domain. Use a different MoE backend or "
            "change the GPU topology."
        )
    _validated_ep_groups.add(group_key)


class DeepGemmMegaMoESM90Experts(nn.Module):
    """Hopper FP8 routed experts backed by DeepGEMM MegaMoE."""

    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int, int], object] = {}

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
        num_shared_experts: int | None = None,
        fuse_shared_experts: bool = False,
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
        self.num_shared_experts = (
            (num_shared_experts or 0) if fuse_shared_experts else 0
        )
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
        self._transformed_shared_l1_weights: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        self._transformed_shared_l2_weights: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None
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
                    f"SM90 MegaMoE expert weight shape mismatch for {weight_name}: "
                    f"parameter shard {tuple(expert_data.shape)} vs checkpoint "
                    f"{tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
            loaded_any = True
        return loaded_any if return_success else None

    def _check_runtime_supported(self) -> None:
        device = self.w13_weight.device
        if torch.cuda.get_device_capability(device) != (9, 0):
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
                "The installed DeepGEMM build does not expose the SM90 "
                "all-FP8 MegaMoE API."
            )

    def _finalize_shared_expert_weights(
        self, deep_gemm, shared_experts: nn.Module
    ) -> None:
        shared_intermediate_size = self.intermediate_size * self.num_shared_experts
        gate_up_weight = shared_experts.gate_up_proj.weight
        gate_up_scale = shared_experts.gate_up_proj.weight_scale_inv
        down_weight = shared_experts.down_proj.weight
        down_scale = shared_experts.down_proj.weight_scale_inv

        expected_tensors = (
            (
                "gate_up_proj.weight",
                gate_up_weight,
                (2 * shared_intermediate_size, self.hidden_size),
                torch.float8_e4m3fn,
            ),
            (
                "gate_up_proj.weight_scale_inv",
                gate_up_scale,
                (
                    cdiv(2 * shared_intermediate_size, self.block_size),
                    cdiv(self.hidden_size, self.block_size),
                ),
                torch.float32,
            ),
            (
                "down_proj.weight",
                down_weight,
                (self.hidden_size, shared_intermediate_size),
                torch.float8_e4m3fn,
            ),
            (
                "down_proj.weight_scale_inv",
                down_scale,
                (
                    cdiv(self.hidden_size, self.block_size),
                    cdiv(shared_intermediate_size, self.block_size),
                ),
                torch.float32,
            ),
        )
        for name, tensor, expected_shape, expected_dtype in expected_tensors:
            if tensor.shape != expected_shape or tensor.dtype != expected_dtype:
                raise ValueError(
                    f"SM90 MegaMoE shared expert {name} must have shape "
                    f"{expected_shape} and dtype {expected_dtype}, got "
                    f"shape={tuple(tensor.shape)} and dtype={tensor.dtype}."
                )

        shared_l1, shared_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            (gate_up_weight.data, gate_up_scale.data),
            (down_weight.data, down_scale.data),
        )
        gate_up_weight.data = shared_l1[0]
        self._transformed_shared_l1_weights = (gate_up_weight, shared_l1[1])
        self._transformed_shared_l2_weights = shared_l2

    def finalize_weights(self, shared_experts: nn.Module | None = None) -> None:
        num_shared_experts = getattr(self, "num_shared_experts", 0)
        shared_weights_finalized = (
            getattr(self, "_transformed_shared_l1_weights", None) is not None
        )
        if self._transformed_l1_weights is not None and (
            num_shared_experts == 0
            or shared_experts is None
            or shared_weights_finalized
        ):
            return
        self._check_runtime_supported()
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        assert deep_gemm is not None
        if self._transformed_l1_weights is None:
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

        if (
            num_shared_experts == 0
            or self._transformed_shared_l1_weights is not None
            or shared_experts is None
        ):
            return
        buffer_params = inspect.signature(
            deep_gemm.get_symm_buffer_for_sm90_mega_moe
        ).parameters
        kernel_params = inspect.signature(deep_gemm.fp8_mega_moe_sm90).parameters
        if "num_shared_experts" not in buffer_params or not {
            "shared_l1_weights",
            "shared_l2_weights",
        }.issubset(kernel_params):
            raise RuntimeError(
                "The installed DeepGEMM SM90 MegaMoE API does not support fused "
                "shared experts."
            )
        self._finalize_shared_expert_weights(deep_gemm, shared_experts)

    @property
    def has_fused_shared_experts(self) -> bool:
        return getattr(self, "_transformed_shared_l1_weights", None) is not None

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
            self.num_shared_experts,
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
                num_shared_experts=self.num_shared_experts,
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

        def to_eplb_view(name: str, tensor: torch.Tensor) -> torch.Tensor:
            assert tensor.shape[0] == self.num_local_experts
            if tensor.is_contiguous():
                return tensor.view(self.num_local_experts, -1)
            if (
                tensor.dim() == 3
                and tensor.stride(1) == 1
                and tensor.stride(2) == tensor.shape[1]
            ):
                contiguous = torch.transpose(tensor, 1, 2)
                assert contiguous.is_contiguous()
                return contiguous.view(self.num_local_experts, -1)
            raise AssertionError(
                f"SM90 MegaMoE EPLB {name} has unexpected layout: "
                f"shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, "
                f"dtype={tensor.dtype}"
            )

        return [
            to_eplb_view("l1_weight", self._transformed_l1_weights[0]),
            to_eplb_view("l1_scale", self._transformed_l1_weights[1]),
            to_eplb_view("l2_weight", self._transformed_l2_weights[0]),
            to_eplb_view("l2_scale", self._transformed_l2_weights[1]),
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
        fast_math: bool = True,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"SM90 MegaMoE got {hidden_states.shape[0]} tokens, but its "
                f"symmetric buffer was sized for {self.max_num_tokens}."
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
        if self.num_shared_experts and not self.has_fused_shared_experts:
            raise RuntimeError(
                "SM90 MegaMoE shared expert weights must be finalized before "
                "the first forward pass."
            )
        output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        deep_gemm.fp8_mega_moe_sm90(
            output,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            shared_l1_weights=self._transformed_shared_l1_weights,
            shared_l2_weights=self._transformed_shared_l2_weights,
            cumulative_local_expert_recv_stats=self._cum_local_expert_recv_stats,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        return output


DeepGemmMegaMoESM90Experts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
