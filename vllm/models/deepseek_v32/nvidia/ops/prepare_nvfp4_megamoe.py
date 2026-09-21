# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input staging for the SM90 NVFP4 DeepGEMM MegaMoE kernel."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_nvfp4_megamoe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    top_k: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    group_id = tl.program_id(1)
    offsets = group_id * GROUP_K + tl.arange(0, GROUP_K)
    hidden = tl.load(hidden_states + token_id * hidden_stride_m + offsets).to(
        tl.float32
    )
    amax = tl.maximum(tl.max(tl.abs(hidden), axis=0), 1.0e-4)
    scale = amax / 448.0
    tl.store(x_fp8 + token_id * x_stride_m + offsets, hidden / scale)
    tl.store(x_sf + token_id * x_sf_stride_m + group_id, scale)

    if group_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k
        token_is_padding = False
        if is_padding is not None:
            token_is_padding = tl.load(is_padding + token_id * is_padding_stride_m)

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        )
        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        ids = tl.where(token_is_padding, -1, ids)
        weights = tl.where(token_is_padding, 0.0, weights)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def prepare_nvfp4_megamoe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> None:
    """Quantize BF16 activations to FP8 and stage routing metadata."""
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must be a 2D BF16 tensor.")
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError("MegaMoE hidden_size must be a multiple of 128.")
    if hidden_states.stride(1) != 1:
        raise ValueError("hidden_states must be contiguous in its hidden dimension.")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same 2D shape.")
    if topk_ids.shape[0] != num_tokens:
        raise ValueError("topk tensors must have one row per input token.")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_ids must use int32 or int64 dtype.")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must use FP32 dtype.")
    if x_fp8.shape != hidden_states.shape or x_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("x_fp8 must match hidden_states and use E4M3 dtype.")
    if x_sf.shape != (num_tokens, hidden_size // 128):
        raise ValueError("x_sf must contain one FP32 scale per 128 values.")
    if x_sf.dtype != torch.float32:
        raise TypeError("x_sf must have FP32 dtype.")
    if topk_idx_out.shape != topk_ids.shape or topk_idx_out.dtype != torch.int64:
        raise ValueError("topk_idx_out must match topk_ids and use int64 dtype.")
    if (
        topk_weights_out.shape != topk_weights.shape
        or topk_weights_out.dtype != torch.float32
    ):
        raise ValueError("topk_weights_out must match topk_weights and use FP32.")
    if is_padding is not None and (
        is_padding.shape != (num_tokens,) or is_padding.dtype != torch.bool
    ):
        raise ValueError("is_padding must be a bool vector with one item per token.")

    top_k = topk_ids.shape[1]
    grid = (num_tokens, hidden_size // 128)
    _prepare_nvfp4_megamoe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        x_fp8.stride(0),
        x_sf.stride(0),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        is_padding.stride(0) if is_padding is not None else 0,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        top_k,
        GROUP_K=128,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        num_warps=4,
    )
