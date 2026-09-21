# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia.mega_moe import DeepseekV32MegaMoEExperts
from vllm.models.deepseek_v32.nvidia.model import (
    DeepseekV32DecoderLayer,
    _make_expert_params_mapping,
    _use_nvfp4_mega_moe,
    _use_sequence_parallel,
)
from vllm.models.deepseek_v32.nvidia.ops.prepare_nvfp4_megamoe import (
    prepare_nvfp4_megamoe_inputs,
)
from vllm.platforms import current_platform


def _parallel_config(**overrides):
    values = {
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 8,
        "enable_expert_parallel": True,
        "use_sequence_parallel_moe": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mega_moe_enables_sequence_parallel_for_tp_ep():
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        parallel_config=_parallel_config(),
    )

    assert _use_sequence_parallel(vllm_config)


def test_mega_moe_sequence_parallel_rejects_pipeline_parallel():
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        parallel_config=_parallel_config(pipeline_parallel_size=2),
    )

    assert not _use_sequence_parallel(vllm_config)


def test_mega_moe_respects_modelopt_exclusions():
    class QuantConfig:
        quant_method = "NVFP4"

        @staticmethod
        def get_name():
            return "modelopt_fp4"

        @staticmethod
        def is_layer_excluded(prefix):
            return prefix.startswith("model.layers.78.")

    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        quant_config=QuantConfig(),
    )

    assert _use_nvfp4_mega_moe(vllm_config, "model.layers.3")
    assert not _use_nvfp4_mega_moe(vllm_config, "model.layers.78")


def test_expert_mapping_supports_model_owned_and_routed_experts():
    mapping = _make_expert_params_mapping(torch.nn.Module(), num_experts=1)

    assert (
        "experts.w2_",
        "experts.0.down_proj.",
        0,
        "w2",
    ) in mapping
    assert (
        "experts.routed_experts.w2_",
        "experts.0.down_proj.",
        0,
        "w2",
    ) in mapping


def test_unquantized_mtp_overrides_mega_moe_backend(monkeypatch):
    class QuantConfig:
        quant_method = "NVFP4"

        @staticmethod
        def get_name():
            return "modelopt_fp4"

        @staticmethod
        def is_layer_excluded(prefix):
            return prefix.startswith("model.layers.78.")

    config = SimpleNamespace(
        hidden_size=6144,
        intermediate_size=2048,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        num_hidden_layers=78,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        moe_intermediate_size=2048,
        first_k_dense_replace=3,
        moe_layer_freq=1,
    )
    parallel_config = _parallel_config(
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        quant_config=QuantConfig(),
        parallel_config=parallel_config,
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
    )
    captured = {}

    def make_moe(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(
        "vllm.models.deepseek_v32.nvidia.model.DeepseekV32Attention",
        lambda **kwargs: nn.Identity(),
    )
    monkeypatch.setattr("vllm.models.deepseek_v32.nvidia.model.DeepseekV2MoE", make_moe)
    monkeypatch.setattr(
        "vllm.models.deepseek_v32.nvidia.model.RMSNorm",
        lambda *args, **kwargs: nn.Identity(),
    )

    DeepseekV32DecoderLayer(
        vllm_config,
        "model.layers.78",
        config=config,
    )

    assert captured["moe_backend"] == "triton"


def test_mega_moe_weight_loader_uses_ep_expert_ownership():
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV32MegaMoEExperts(
        vllm_config,
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=128,
        intermediate_size=16,
        prefix="model.layers.0.mlp.experts",
    )

    w1 = torch.full((16, 64), 3, dtype=torch.uint8)
    w3 = torch.full((16, 64), 7, dtype=torch.uint8)
    w2 = torch.full((128, 8), 11, dtype=torch.uint8)
    assert not experts.weight_loader(
        experts.w13_weight,
        w1,
        "experts.w13_weight",
        "w1",
        expert_id=1,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight,
        w1,
        "experts.w13_weight",
        "w1",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w13_weight,
        w3,
        "experts.w13_weight",
        "w3",
        expert_id=2,
        return_success=True,
    )
    assert experts.weight_loader(
        experts.w2_weight,
        w2,
        "experts.w2_weight",
        "w2",
        expert_id=2,
        return_success=True,
    )

    assert torch.equal(experts.w13_weight[0, :16], w1)
    assert torch.equal(experts.w13_weight[0, 16:], w3)
    assert torch.equal(experts.w2_weight[0], w2)


def test_mega_moe_weight_loader_keeps_w13_scalar_shards_separate():
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV32MegaMoEExperts(
        vllm_config,
        num_experts=1,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=16,
        prefix="model.layers.0.mlp.experts",
    )

    for shard_id, value in (("w1", 2.0), ("w3", 3.0)):
        assert experts.weight_loader(
            experts.w13_weight_scale_2,
            torch.tensor(value),
            "experts.w13_weight_scale_2",
            shard_id,
            expert_id=0,
            return_success=True,
        )

    torch.testing.assert_close(
        experts.w13_weight_scale_2,
        torch.tensor([[2.0, 3.0]]),
    )


def test_mega_moe_finalize_uses_modelopt_global_weight_scales(monkeypatch):
    class FakeDeepGemm:
        @staticmethod
        def transform_nvfp4_weights_for_mega_moe_sm90(l1, l2):
            assert l1[0].dtype == l2[0].dtype == torch.uint8
            assert l1[1].dtype == l2[1].dtype == torch.uint8
            return l1, l2

        @staticmethod
        def get_symm_buffer_for_mega_moe(*args, **kwargs):
            return None

        @staticmethod
        def nvfp4_mega_moe(*args, **kwargs):
            return None

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    experts = DeepseekV32MegaMoEExperts(
        vllm_config,
        num_experts=1,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=16,
        prefix="model.layers.0.mlp.experts",
    )
    experts.w13_weight_scale_2.copy_(torch.tensor([[2.0, 2.0]]))
    experts.w2_weight_scale_2.copy_(torch.tensor([3.0]))
    monkeypatch.setattr(current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm", lambda: FakeDeepGemm()
    )

    experts.finalize_weights()

    torch.testing.assert_close(experts._l1_global_scales, torch.tensor([2.0]))
    torch.testing.assert_close(experts._l2_global_scales, torch.tensor([3.0]))
    assert experts.w13_weight is None
    assert experts.w2_weight is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_prepare_nvfp4_megamoe_inputs_matches_fp8_reference():
    num_tokens, hidden_size, top_k = 3, 6144, 8
    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    topk_ids = torch.arange(top_k, device="cuda", dtype=torch.int32).repeat(
        num_tokens, 1
    )
    topk_weights = torch.rand(num_tokens, top_k, device="cuda", dtype=torch.float32)
    x_fp8 = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
    x_sf = torch.empty(
        num_tokens, hidden_size // 128, device="cuda", dtype=torch.float32
    )
    staged_ids = torch.empty_like(topk_ids, dtype=torch.int64)
    staged_weights = torch.empty_like(topk_weights)

    prepare_nvfp4_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        x_fp8,
        x_sf,
        staged_ids,
        staged_weights,
    )

    hidden_groups = hidden_states.float().view(num_tokens, -1, 128)
    expected_sf = hidden_groups.abs().amax(dim=-1).clamp_min(1.0e-4) / 448.0
    torch.testing.assert_close(x_sf, expected_sf)
    expected_fp8 = (hidden_groups / expected_sf.unsqueeze(-1)).to(torch.float8_e4m3fn)
    torch.testing.assert_close(x_fp8.view_as(expected_fp8), expected_fp8)
    torch.testing.assert_close(staged_ids, topk_ids.to(torch.int64))
    torch.testing.assert_close(staged_weights, topk_weights)
