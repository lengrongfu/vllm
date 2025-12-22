# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F
from sonicmoe.functional import moe_general_routing_inputs


def forward_topk(
    x: torch.Tensor, router_w: torch.Tensor, E, K
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = x.shape[0]

    router_logits = F.linear(x, router_w)

    top_logits, topk_indices = router_logits.topk(K, dim=1)
    router_scores = F.softmax(top_logits, dim=-1, dtype=torch.float32)

    # first sorting, similar to TC
    return (
        router_scores.view(-1),
        torch.arange(T, device="cuda", dtype=torch.int32).repeat_interleave(K),
        topk_indices.view(-1).int(),
    )


def sonic_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if activation != "silu":
        raise ValueError("SonicMoE only supports silu activation for SwigLU experts.")
    if (w1_bias is None) != (w2_bias is None):
        raise ValueError(
            "SonicMoE expects w1_bias and w2_bias to be both None or both set."
        )
    if hidden_states.numel() == 0:
        return hidden_states

    # hidden_states = hidden_states.contiguous()
    # topk_weights = topk_weights.contiguous()
    # topk_ids = topk_ids.contiguous()

    if apply_router_weight_on_input:
        topk = topk_ids.size(1)
        assert topk == 1, "apply_router_weight_on_input is only supported for topk=1."
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        router_scores_selected = torch.ones_like(topk_weights)
    else:
        router_scores_selected = topk_weights

    selected_E = topk_ids.reshape(-1).to(dtype=torch.int32)
    num_tokens = hidden_states.size(0)
    topk = topk_ids.size(1)
    sorted_selected_T = torch.arange(
        num_tokens, device=hidden_states.device, dtype=torch.int32
    ).repeat_interleave(topk)
    output, _ = moe_general_routing_inputs(
        x=hidden_states,
        router_scores_selected=router_scores_selected.reshape(-1),
        sorted_selected_T=sorted_selected_T,
        selected_E=selected_E,
        w1=w1,
        b1=w1_bias,
        w2=w2,
        b2=w2_bias,
        E=w2.size(-1),
        stream_id=torch.cuda.current_stream().cuda_stream,
        is_inference_mode_enabled=True,
    )
    return output
