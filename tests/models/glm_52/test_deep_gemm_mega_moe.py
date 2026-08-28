# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="GLM-5.2 SM90 MegaMoE requires CUDA",
)


def test_glm52_mega_moe_accepts_hf_fp8_contract():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.glm_52.nvidia.sm90_mega_moe import (
        _validate_glm52_mega_moe_config,
    )

    config = SimpleNamespace(
        model_type="glm_moe_dsa",
        hidden_act="silu",
        topk_method="noaux_tc",
        scoring_func="sigmoid",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
        ),
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        ),
    )

    _validate_glm52_mega_moe_config(vllm_config, config)

    vllm_config.quant_config.weight_block_size = [64, 128]
    with pytest.raises(NotImplementedError, match="weight_block_size"):
        _validate_glm52_mega_moe_config(vllm_config, config)

    vllm_config.quant_config.weight_block_size = [128, 128]
    config.model_type = "deepseek_mtp"
    config.architectures = ["Glm52MTPModel"]
    _validate_glm52_mega_moe_config(vllm_config, config)

    config.architectures = ["DeepseekV32MTPModel"]
    with pytest.raises(NotImplementedError, match="Glm52MTPModel"):
        _validate_glm52_mega_moe_config(vllm_config, config)


def test_glm52_mega_moe_router_and_shared_expert_contract(monkeypatch):
    from vllm.models.glm_52.nvidia import sm90_mega_moe as glm_moe

    class Gate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(4))

        def forward(self, hidden_states):
            return torch.zeros(hidden_states.shape[0], 4), None

    class Experts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.has_fused_shared_experts = True

        def forward(self, hidden_states, topk_weights, topk_ids):
            self.inputs = (hidden_states, topk_weights, topk_ids)
            return hidden_states + 1

    moe = glm_moe.Glm52DeepGemmMegaMoE.__new__(glm_moe.Glm52DeepGemmMegaMoE)
    torch.nn.Module.__init__(moe)
    moe.is_sequence_parallel = False
    moe.top_k = 2
    moe.renormalize = True
    moe.num_expert_group = 2
    moe.topk_group = 1
    moe.scoring_func = "sigmoid"
    moe.routed_scaling_factor = 2.5
    moe.gate = Gate()
    moe.experts = Experts()
    moe.shared_experts = torch.nn.Identity()

    topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    topk_ids = torch.tensor([[1, 3], [0, 2]])
    router_args = {}

    def grouped_topk(**kwargs):
        router_args.update(kwargs)
        return topk_weights, topk_ids

    monkeypatch.setattr(glm_moe, "fused_grouped_topk", grouped_topk)
    hidden_states = torch.randn(2, 8)

    output = moe(hidden_states)

    assert router_args["scoring_func"] == "sigmoid"
    assert router_args["routed_scaling_factor"] == 2.5
    assert (
        router_args["e_score_correction_bias"].data_ptr()
        == moe.gate.e_score_correction_bias.data_ptr()
    )
    assert moe.experts.inputs == (hidden_states, topk_weights, topk_ids)
    torch.testing.assert_close(output, hidden_states + 1)


def test_glm52_mega_moe_finalizes_fused_shared_expert_weights(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import (
        deep_gemm_mega_moe_sm90 as mega_moe,
    )

    class FakeDeepGemm:
        @staticmethod
        def get_symm_buffer_for_sm90_mega_moe(*args, num_shared_experts=0):
            return None

        @staticmethod
        def fp8_mega_moe_sm90(*args, shared_l1_weights=None, shared_l2_weights=None):
            return None

        @staticmethod
        def transform_weights_for_mega_moe_sm90(l1_weights, l2_weights):
            return l1_weights, l2_weights

    hidden_size = 128
    intermediate_size = 128
    experts = mega_moe.DeepGemmMegaMoESM90Experts.__new__(
        mega_moe.DeepGemmMegaMoESM90Experts
    )
    torch.nn.Module.__init__(experts)
    experts.hidden_size = hidden_size
    experts.intermediate_size = intermediate_size
    experts.block_size = 128
    experts.num_shared_experts = 1
    experts.num_local_experts = 1
    experts._transformed_l1_weights = (
        torch.empty(1, 2 * intermediate_size, hidden_size),
        torch.empty(1, 2, 1),
    )
    experts._transformed_l2_weights = (
        torch.empty(1, hidden_size, intermediate_size),
        torch.empty(1, 1, 1),
    )
    experts._transformed_shared_l1_weights = None
    experts._transformed_shared_l2_weights = None
    experts._check_runtime_supported = lambda: None

    shared_experts = torch.nn.Module()
    shared_experts.gate_up_proj = torch.nn.Module()
    shared_experts.gate_up_proj.weight = torch.nn.Parameter(
        torch.empty(
            2 * intermediate_size,
            hidden_size,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    shared_experts.gate_up_proj.weight_scale_inv = torch.nn.Parameter(
        torch.empty(2, 1, dtype=torch.float32), requires_grad=False
    )
    shared_experts.down_proj = torch.nn.Module()
    shared_experts.down_proj.weight = torch.nn.Parameter(
        torch.empty(
            hidden_size,
            intermediate_size,
            dtype=torch.float8_e4m3fn,
        ),
        requires_grad=False,
    )
    shared_experts.down_proj.weight_scale_inv = torch.nn.Parameter(
        torch.empty(1, 1, dtype=torch.float32), requires_grad=False
    )

    monkeypatch.setattr("vllm.utils.deep_gemm._import_deep_gemm", lambda: FakeDeepGemm)
    experts.finalize_weights(shared_experts)

    assert experts.has_fused_shared_experts
    assert experts._transformed_shared_l1_weights is not None
    assert experts._transformed_shared_l2_weights is not None


def test_glm52_mega_moe_expert_mapping_uses_checkpoint_prefix(monkeypatch):
    from vllm.models.glm_52.nvidia import model as glm_model

    captured = {}

    def make_mapping(owner, **kwargs):
        captured.update(kwargs)
        return [("mapped", "weight", 0, "w1")]

    monkeypatch.setattr(glm_model, "fused_moe_make_expert_params_mapping", make_mapping)
    model = glm_model.GlmMoeDsaForCausalLM.__new__(glm_model.GlmMoeDsaForCausalLM)
    model.config = SimpleNamespace(n_routed_experts=256)
    model.model = SimpleNamespace(
        num_redundant_experts=8,
        routed_experts_prefix="",
    )

    mapping = model.get_expert_mapping()

    assert mapping == [("mapped", "weight", 0, "w1")]
    assert captured["num_experts"] == 256
    assert captured["num_redundant_experts"] == 8
    assert captured["routed_experts_prefix"] == ""


def test_glm52_mtp_uses_glm_decoder_layer():
    from vllm.models.glm_52.nvidia.model import Glm52DecoderLayer
    from vllm.models.glm_52.nvidia.mtp import (
        Glm52MTP,
        Glm52MultiTokenPredictor,
        Glm52MultiTokenPredictorLayer,
    )

    assert Glm52MTP.model_cls is Glm52MultiTokenPredictor
    assert Glm52MultiTokenPredictor.layer_cls is Glm52MultiTokenPredictorLayer
    assert Glm52MultiTokenPredictorLayer.decoder_layer_cls is Glm52DecoderLayer
