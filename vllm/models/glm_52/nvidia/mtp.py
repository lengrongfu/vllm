# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.models.deepseek_v32.nvidia.mtp import (
    DeepseekV32MTP,
    DeepseekV32MultiTokenPredictor,
    DeepseekV32MultiTokenPredictorLayer,
)

from .model import Glm52DecoderLayer, _use_mega_moe
from .sm90_mega_moe import Glm52DeepGemmMegaMoE


class Glm52MultiTokenPredictorLayer(DeepseekV32MultiTokenPredictorLayer):
    decoder_layer_cls = Glm52DecoderLayer


class Glm52MultiTokenPredictor(DeepseekV32MultiTokenPredictor):
    layer_cls = Glm52MultiTokenPredictorLayer


class Glm52MTP(DeepseekV32MTP):
    model_cls = Glm52MultiTokenPredictor

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.routed_experts_prefix = (
            "" if _use_mega_moe(vllm_config) else "routed_experts"
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def set_moe_parameters(self) -> None:
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = self.config.n_group
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, (DeepseekV2MoE, Glm52DeepGemmMegaMoE)):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        self.extract_moe_parameters(example_moe)

    def process_weights_after_loading(self) -> None:
        for layer in self.model.layers.values():
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, Glm52DeepGemmMegaMoE):
                mlp.finalize_mega_moe_weights()
