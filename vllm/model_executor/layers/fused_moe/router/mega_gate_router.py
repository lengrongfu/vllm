# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import bf16_mega_gate, is_mega_gate_supported


def is_mega_gate_shape_preferred(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    *,
    is_sm90: bool,
) -> bool:
    """Avoid GLM batches where the SM90 path loses to GEMM plus fused top-k."""
    if not 16 < num_tokens <= 1 << 20:
        return False
    if is_sm90 and (hidden_size, num_experts, top_k) == (6144, 256, 8):
        return num_tokens <= 128 or num_tokens == 256 or num_tokens >= 3584
    return True


class MegaGateRouter(BaseRouter):
    """Fuse a GLM gate and single-group sigmoid routing on SM90 or SM10x.

    The runner delegates the gate to this router. Small or unsupported batches
    use the original gate and router. BaseRouter retains EPLB and capture handling.
    """

    def __init__(self, fallback: GroupedTopKRouter, gate: GateLinear):
        super().__init__(
            fallback.top_k, fallback.global_num_experts, fallback.eplb_state
        )
        self.fallback = fallback
        self.gate = gate
        # Initialize the DeepGEMM import before graph capture.
        self.is_sm90 = current_platform.is_device_capability(90)
        self.available = is_mega_gate_supported()
        self._warmed_streams: set[torch.cuda.Stream] = set()

    @staticmethod
    def is_supported(router: object, gate: object) -> bool:
        return (
            type(router) is GroupedTopKRouter
            and type(gate) is GateLinear
            and router.num_expert_group == router.topk_group == 1
            and router.scoring_func == "sigmoid"
            and router.renormalize
            and router.num_fused_shared_experts == 0
            and gate.weight.dtype == torch.bfloat16
            and gate.out_dtype == torch.float32
            and gate.bias is None
        )

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return self.fallback.routing_method_type

    def _can_use_mega_gate(self, hidden_states: torch.Tensor) -> bool:
        weight = self.gate.weight
        bias = self.fallback.e_score_correction_bias
        return (
            self.available
            and not envs.VLLM_BATCH_INVARIANT
            and hidden_states.is_cuda
            and hidden_states.dtype == weight.dtype == torch.bfloat16
            and hidden_states.device == weight.device
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and is_mega_gate_shape_preferred(
                hidden_states.shape[0],
                weight.shape[1],
                weight.shape[0],
                self.top_k,
                is_sm90=self.is_sm90,
            )
            and hidden_states.shape[1] == weight.shape[1]
            and weight.shape[1] % 256 == 0
            and 0 < weight.shape[0] <= 512
            and weight.shape[0] % 4 == 0
            and 0 < self.top_k <= min(weight.shape[0], 32)
            and (
                bias is None
                or (
                    bias.dtype == torch.float32
                    and bias.device == weight.device
                    and bias.shape == (weight.shape[0],)
                    and bias.is_contiguous()
                )
            )
            # Keep the existing invalid-route convention for padding-skipping backends.
            and not self.fallback.skip_padding
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._can_use_mega_gate(hidden_states):
            stream = torch.cuda.current_stream(hidden_states.device)
            # SM10x allocates per-stream barriers; the SM90 kernel has none.
            if (
                self.is_sm90
                or not torch.cuda.is_current_stream_capturing()
                or stream in self._warmed_streams
            ):
                mask = None
                if self.is_sm90 and is_forward_context_available():
                    is_padding = get_forward_context().is_padding
                    if is_padding is not None:
                        mask = ~is_padding[: hidden_states.shape[0]]
                result = bf16_mega_gate(
                    hidden_states,
                    self.gate.weight,
                    self.top_k,
                    scoring_func="sigmoid",
                    routed_scaling_factor=self.fallback.routed_scaling_factor,
                    ep_rank=0,  # BaseRouter owns physical mapping and EPLB.
                    mask=mask,
                    bias=self.fallback.e_score_correction_bias,
                )
                self._warmed_streams.add(stream)
                return result

        router_logits, _ = self.gate(hidden_states)
        return self.fallback._compute_routing(
            hidden_states, router_logits, indices_type, input_ids=input_ids
        )
