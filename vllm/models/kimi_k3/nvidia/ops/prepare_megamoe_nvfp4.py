# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton input-staging kernel for Kimi K3 NVFP4 MegaMoE.

Quantizes hidden states to packed NVFP4 (E2M1, 2 values/byte) with
per-16-element E4M3 block scales (4 scale bytes packed per int32, K-major),
matching the layout `deep_gemm.fp4_fp4_mega_moe`'s symmetric buffer expects.
Also repacks routing top-k tensors into the int64/float32 layout the
MegaMoE kernels consume.

Unlike the FP8 dispatch path (dynamic per-token amax scaling), NVFP4
activation quantization here is *static*: the block scale is normalized by
a single per-tensor `input_global_scale` loaded from the checkpoint
(`input_activations.dynamic == false` in Kimi-K3-NVFP4's quantization
config), not computed from the live activation range.
"""

import torch

from vllm.triton_utils import tl, triton

# E2M1 has 8 representable magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_MAX = 6.0
_E2M1_MAX_INV = 1.0 / 6.0
# Smallest representable E4M3 magnitude (subnormal), used as an SF floor so
# near-zero blocks don't quantize their scale to exact zero.
_E4M3_MIN_SF = 2.0**-9


@triton.jit
def _quantize_e2m1(x):
    """Round each element of `x` to the nearest E2M1 magnitude, keeping sign."""
    sign = tl.where(x < 0, -1.0, 1.0)
    ax = tl.abs(x)
    # Midpoints between consecutive E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}.
    levels = tl.where(
        ax < 0.25,
        0.0,
        tl.where(
            ax < 0.75,
            0.5,
            tl.where(
                ax < 1.25,
                1.0,
                tl.where(
                    ax < 1.75,
                    1.5,
                    tl.where(
                        ax < 2.5,
                        2.0,
                        tl.where(ax < 3.5, 3.0, tl.where(ax < 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return sign * levels


@triton.jit
def _pack_e2m1_to_uint8(hi, lo):
    """Pack two E2M1 magnitudes (already sign-folded into {-6..6}) into one byte.

    Nibble layout: low nibble = `lo`, high nibble = `hi` (mantissa/exp bits of
    the E2M1 format, derived from a small value->code lookup since there are
    only 16 signed codes).
    """

    def code(v):
        s = tl.where(v < 0, 8, 0).to(tl.uint8)
        av = tl.abs(v)
        m = tl.where(
            av < 0.25,
            0,
            tl.where(
                av < 0.75,
                1,
                tl.where(
                    av < 1.25,
                    2,
                    tl.where(
                        av < 1.75,
                        3,
                        tl.where(
                            av < 2.5,
                            4,
                            tl.where(av < 3.5, 5, tl.where(av < 5.0, 6, 7)),
                        ),
                    ),
                ),
            ),
        ).to(tl.uint8)
        return s | m

    return (code(hi) << 4) | code(lo)


@triton.jit
def _prepare_megamoe_nvfp4_inputs_kernel(
    hidden_states,
    input_global_scale,
    x_fp4,
    x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    inv_global = 1.0 / tl.load(input_global_scale)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.maximum(tl.max(hidden_groups, axis=1), 1.0e-8)

    # E4M3 block scale: amax normalized into E2M1's representable range, then
    # rescaled by the per-tensor global scale so it fits E4M3's dynamic range.
    sf = tl.maximum(amax * _E2M1_MAX_INV * inv_global, _E4M3_MIN_SF)
    sf_e4m3 = sf.to(tl.float8e4nv)
    sf_dequant = sf_e4m3.to(tl.float32)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups / (sf_dequant * _E2M1_MAX)[:, None]
    scaled = tl.reshape(_quantize_e2m1(tl.reshape(scaled, [BLOCK_K])), [BLOCK_K])

    # Pack two adjacent E2M1 values per byte (lo = even index, hi = odd index).
    half = BLOCK_K // 2
    lo = tl.reshape(scaled, [half, 2])[:, 0]
    hi = tl.reshape(scaled, [half, 2])[:, 1]
    packed = _pack_e2m1_to_uint8(hi, lo)
    byte_offsets = k_block_id * (BLOCK_K // 2) + tl.arange(0, half)
    tl.store(
        x_fp4 + token_id * x_stride_m + byte_offsets * x_stride_k,
        packed,
        mask=byte_offsets < (hidden_size // 2),
    )

    # Pack 4 E4M3 scale bytes per stored int32 (one per GROUP_K=16 block).
    # NOTE: this only produces one non-colliding int32 per program iff
    # BLOCK_K // GROUP_K == 4 (checked in the launcher below); with more than
    # 4 groups per block, distinct groups would alias onto the same byte.
    sf_bits = sf_e4m3.to(tl.uint8, bitcast=True).to(tl.uint32)
    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(sf_bits << scale_offsets * 8, axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k
        token_is_padding = False
        if is_padding is not None:
            token_is_padding = tl.load(is_padding + token_id * is_padding_stride_m)

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        ids = tl.where(token_is_padding, -1, ids)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        weights = tl.where(token_is_padding, 0.0, weights)
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def prepare_megamoe_nvfp4_inputs(
    hidden_states: torch.Tensor,
    input_global_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    # BLOCK_K must equal 4 * GROUP_K (=64) so each program's scale groups
    # pack into exactly one int32 slot without aliasing; see the kernel note
    # above `packed_scale`.
    if hidden_size % 64 != 0:
        raise ValueError(
            "Kimi K3 NVFP4 MegaMoE input staging requires hidden_size to be "
            "a multiple of 64."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "Kimi K3 NVFP4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    block_k = 64
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_megamoe_nvfp4_inputs_kernel[grid](
        hidden_states,
        input_global_scale,
        x_fp4,
        x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp4.stride(0),
        x_fp4.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        padding_stride_m,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        top_k,
        BLOCK_K=block_k,
        GROUP_K=16,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )
