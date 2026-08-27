# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v32.nvidia.model import (
    DeepseekV32DecoderLayer,
    DeepseekV32ForCausalLM,
    DeepseekV32Model,
)

from .sm90_mega_moe import Glm52DeepGemmMegaMoE


def _use_mega_moe(vllm_config: VllmConfig) -> bool:
    return vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"


class Glm52DecoderLayer(DeepseekV32DecoderLayer):
    @classmethod
    def should_use_sequence_parallel(cls, vllm_config: VllmConfig) -> bool:
        if not _use_mega_moe(vllm_config):
            return super().should_use_sequence_parallel(vllm_config)
        parallel_config = vllm_config.parallel_config
        return (
            parallel_config.pipeline_parallel_size == 1
            and parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
        )

    @classmethod
    def make_moe(
        cls,
        vllm_config: VllmConfig,
        config,
        prefix: str,
    ) -> torch.nn.Module:
        if _use_mega_moe(vllm_config):
            return Glm52DeepGemmMegaMoE(vllm_config, config, prefix)
        return super().make_moe(vllm_config, config, prefix)


class Glm52Model(DeepseekV32Model):
    decoder_layer_cls = Glm52DecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.routed_experts_prefix = (
            "" if _use_mega_moe(vllm_config) else "routed_experts"
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, Glm52DeepGemmMegaMoE):
                layer.mlp.finalize_mega_moe_weights()


class GlmMoeDsaForCausalLM(DeepseekV32ForCausalLM):
    model_cls = Glm52Model

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, (DeepseekV2MoE, Glm52DeepGemmMegaMoE)):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.model.num_redundant_experts,
            routed_experts_prefix=self.model.routed_experts_prefix,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = super().load_weights(weights)
        self.process_weights_after_loading()
        return loaded_params

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mega_moe_weights()
