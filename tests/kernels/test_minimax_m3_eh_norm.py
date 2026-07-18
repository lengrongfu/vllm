# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the fused MiniMax-M3 MTP input kernel (fused_eh_gemma_norm):
zero-at-pos-0 + Gemma RMSNorm(embeds) with enorm + Gemma RMSNorm(prev_hidden)
with hnorm, concatenated side-by-side into the [N, 2H] eh_proj GEMM input.

Reference: unfused fp32 PyTorch Gemma RMSNorm (normalize(x) * (1 + w)). The
kernel keeps the whole pipeline in fp32 and rounds to bf16 once, so the same
rtol/atol=1e-2 tolerance as the sibling deepseek_v32 fused-norm test applies.
"""

import pytest
import torch

from vllm.platforms import current_platform

HIDDEN = 6144
EPS = 1e-6

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")


def gemma_rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Gemma RMSNorm matching kernels._gemma_rms_norm (fp32). Returns fp32."""
    xf = x.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(ms + EPS) * (1.0 + w.float())


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512])
def test_fused_eh_gemma_norm(num_tokens: int):
    from vllm.models.minimax_m3.nvidia.kernels import fused_eh_gemma_norm

    torch.manual_seed(4)
    dev = "cuda"
    # Mix in a position-0 token to exercise the embeds-zeroing branch.
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    pos[0] = 0
    embeds = torch.randn(num_tokens, HIDDEN, device=dev, dtype=torch.bfloat16)
    prev = torch.randn(num_tokens, HIDDEN, device=dev, dtype=torch.bfloat16)
    ew = torch.randn(HIDDEN, device=dev, dtype=torch.bfloat16)
    hw = torch.randn(HIDDEN, device=dev, dtype=torch.bfloat16)

    out = fused_eh_gemma_norm(pos, embeds, prev, ew, hw, EPS)

    masked = torch.where(pos.unsqueeze(-1) == 0, torch.zeros_like(embeds), embeds)
    ref = torch.cat([gemma_rms_norm(masked, ew), gemma_rms_norm(prev, hw)], dim=-1)
    assert out.shape == (num_tokens, 2 * HIDDEN)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
