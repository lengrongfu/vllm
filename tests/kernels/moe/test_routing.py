# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm.model_executor.layers.fused_moe.router.mega_gate_router import MegaGateRouter
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.models.llama4 import Llama4MoE
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import bf16_mega_gate, is_mega_gate_supported


def _is_aiter_capable() -> bool:
    """Check if the platform supports AITER (gfx942/gfx950/gfx1250)."""
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import get_cdna_version

        return get_cdna_version() > 2
    except ImportError:
        return False


# Test parameters
MK_S = [(32, 256), (64, 512)]
TOP_KS = [2, 4, 6]
NUM_EXPERTS = [8, 16, 64]


@pytest.fixture
def require_mega_gate():
    if not is_mega_gate_supported():
        pytest.skip("Installed DeepGEMM does not support Mega-Gate on this GPU")


@pytest.mark.parametrize("arch", [80, 90, 100])
@pytest.mark.parametrize(
    "build", ["supported", "missing", "wrong_arch", "split_k_override"]
)
def test_mega_gate_availability_checks_backend_architecture(monkeypatch, arch, build):
    """An SM10x-only installation must not launch its kernel on Hopper."""
    from vllm.utils import deep_gemm

    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        SimpleNamespace(
            is_device_capability=lambda value: arch == value,
            is_device_capability_family=lambda value: arch == value,
        ),
    )
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_bf16_mega_gate_impl", Mock())
    config = (
        Mock(side_effect=RuntimeError("unsupported arch"))
        if build == "wrong_arch"
        else Mock()
    )
    if build == "split_k_override":

        def validate_split_k(m, hidden_size, experts, top_k):
            # A valid split-K=8 override needs at least four 64-wide K tiles
            # per partition; an artificially tiny probe rejects this build.
            assert hidden_size % (64 * 8) == 0
            assert hidden_size // (64 * 8) >= 4
            return {}

        config.side_effect = validate_split_k

    module = (
        SimpleNamespace()
        if build == "missing"
        else SimpleNamespace(get_bf16_mega_gate_config=config)
    )
    monkeypatch.setattr(deep_gemm, "_import_deep_gemm", lambda: module)
    assert deep_gemm.is_mega_gate_supported() == (
        arch in (90, 100)
        and (
            build in ("supported", "split_k_override")
            or (build == "missing" and arch == 100)
        )
    )


@pytest.fixture
def glm_mega_gate(monkeypatch):
    # Avoid distributed initialization: only the gate's weight/forward contract
    # is needed to test routing and runner dispatch.
    gate = GateLinear.__new__(GateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(torch.randn(256, 256, dtype=torch.bfloat16) / 16)
    gate.bias = None
    gate.out_dtype = torch.float32
    monkeypatch.setattr(
        gate,
        "forward",
        Mock(side_effect=lambda x: (x.float() @ gate.weight.float().T, None)),
    )
    fallback = GroupedTopKRouter(
        top_k=8,
        global_num_experts=256,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        e_score_correction_bias=torch.linspace(-0.4, 0.4, 256),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.router.mega_gate_router.is_mega_gate_supported",
        lambda: True,
    )
    return gate, fallback


@pytest.mark.parametrize(
    "mode",
    [
        "fused",
        "fallback",
        "cold_capture",
        "sm90_cold_capture",
        "disabled",
        "monolithic",
        "pcp",
        "naive_dispatch",
        "other_model",
    ],
)
def test_mega_gate_runner_routes_once_and_preserves_capture(
    monkeypatch, glm_mega_gate, mode
):
    """The fused path skips GateLinear and applies bias/scaling and ID mapping once."""
    gate, fallback = glm_mega_gate
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.router.mega_gate_router.current_platform",
        Mock(is_device_capability=Mock(return_value=mode == "sm90_cold_capture")),
    )
    x = torch.randn(17, 256, dtype=torch.bfloat16)
    scores = (x.float() @ gate.weight.float().T).sigmoid()
    ids = (scores + fallback.e_score_correction_bias).topk(8).indices
    expected_weights = scores.gather(1, ids)
    expected_weights *= 2.5 / expected_weights.sum(1, keepdim=True)

    fused = Mock(return_value=(expected_weights, ids))
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.router.mega_gate_router.bf16_mega_gate",
        fused,
    )
    fallback_compute = Mock(return_value=(expected_weights, ids))
    monkeypatch.setattr(fallback, "_compute_routing", fallback_compute)
    monkeypatch.setattr(
        MegaGateRouter, "_can_use_mega_gate", lambda *args: mode != "fallback"
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: 0)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: "cold_capture" in mode
    )
    monkeypatch.setenv("VLLM_GLM_USE_MEGA_GATE", "0" if mode == "disabled" else "1")

    quant_method = SimpleNamespace(
        is_monolithic=mode == "monolithic",
        supports_internal_mk=mode != "naive_dispatch",
        topk_indices_dtype=torch.int32,
    )
    experts = torch.nn.Module()
    experts.quant_method = quant_method
    experts._ensure_moe_quant_config_init = lambda: None
    received = {}

    def forward_modular(**kwargs):
        received.update(kwargs)
        return kwargs["x"] * kwargs["topk_weights"].sum(1, keepdim=True)

    experts.forward_modular = forward_modular
    experts.forward_monolithic = Mock(
        return_value=x * expected_weights.sum(1, keepdim=True)
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v3" if mode == "other_model" else "glm_moe_dsa"
            )
        ),
        lora_config=None,
        compilation_config=SimpleNamespace(
            static_forward_context={}, static_all_moe_layers=[]
        ),
    )
    monkeypatch.setattr(moe_runner, "get_current_vllm_config", lambda: config)
    runner = moe_runner.MoERunner(
        "model.layers.3.mlp.experts",
        SimpleNamespace(
            dp_size=2 if mode == "naive_dispatch" else 1,
            pcp_size=2 if mode == "pcp" else 1,
            is_sequence_parallel=False,
        ),
        fallback,
        experts,
        gate=gate,
    )
    monkeypatch.setattr(runner, "_sequence_parallel_context", nullcontext)
    monkeypatch.setattr(runner, "_maybe_dispatch", lambda x, logits: (x, logits))
    monkeypatch.setattr(runner, "_maybe_combine", lambda shared, routed: routed)
    logical_capture = Mock()
    runner.router.set_capture_fn(logical_capture)
    replay = torch.empty((17, 8), dtype=torch.int16)
    runner.router._routing_replay_out = replay
    mapping = Mock(side_effect=lambda logical_ids: logical_ids + 256)
    monkeypatch.setattr(runner.router, "_apply_eplb_mapping", mapping)

    output = runner._forward_impl(x, x, None)
    torch.testing.assert_close(output, x * expected_weights.sum(1, keepdim=True))
    uses_fused = mode in ("fused", "sm90_cold_capture")
    assert gate.forward.call_count == (not uses_fused)
    assert fused.call_count == uses_fused
    if mode == "monolithic":
        torch.testing.assert_close(
            experts.forward_monolithic.call_args.kwargs["router_logits"],
            x.float() @ gate.weight.float().T,
        )
        assert not logical_capture.called
        return
    torch.testing.assert_close(received["topk_weights"], expected_weights)
    torch.testing.assert_close(received["topk_ids"], (ids + 256).int())
    torch.testing.assert_close(replay, (ids + 256).short())
    torch.testing.assert_close(logical_capture.call_args.args[0], ids)
    assert mapping.call_count == 1
    if uses_fused:
        assert fused.call_args.kwargs["scoring_func"] == "sigmoid"
        assert fused.call_args.kwargs["routed_scaling_factor"] == 2.5
        assert fused.call_args.kwargs["bias"] is fallback.e_score_correction_bias
    else:
        torch.testing.assert_close(
            fallback_compute.call_args.args[1], x.float() @ gate.weight.float().T
        )


@pytest.mark.parametrize(
    "change", ["grouping", "normalization", "softmax", "shared", "fp32_gate"]
)
def test_mega_gate_rejects_incompatible_routing(glm_mega_gate, change):
    gate, router = glm_mega_gate
    assert MegaGateRouter.is_supported(router, gate)
    if change == "grouping":
        router.num_expert_group = 8
    elif change == "normalization":
        router.renormalize = False
    elif change == "softmax":
        router.scoring_func = "softmax"
    elif change == "shared":
        router.num_fused_shared_experts = 1
    else:
        gate.weight.data = gate.weight.float()
    assert not MegaGateRouter.is_supported(router, gate)


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability(90)
    ),
    reason="Mega-Gate requires SM90 or SM10x",
)
@pytest.mark.parametrize(
    "num_tokens",
    [1, 16, 17, 32, 64, 65, 127, 128, 129, 255, 256, 257, 512, 1024, 4095, 4096, 4097],
)
@pytest.mark.parametrize("use_cudagraph", [False, True])
def test_mega_gate_glm_shape_matches_reference(
    require_mega_gate, glm_mega_gate, num_tokens, use_cudagraph
):
    """Validate GLM-5.3's actual gate shape, sigmoid bias and normalized top-8."""
    gate, fallback = glm_mega_gate
    gate.weight = torch.nn.Parameter(
        torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16) * 0.001
    )
    fallback.e_score_correction_bias = torch.linspace(-4, 4, 256, device="cuda")
    router = MegaGateRouter(fallback, gate)
    x = torch.randn(num_tokens, 6144, device="cuda", dtype=torch.bfloat16)
    scores = torch.mm(x, gate.weight.T, out_dtype=torch.float32).sigmoid()
    ids = (scores + fallback.e_score_correction_bias).topk(8).indices
    weights = scores.gather(1, ids)
    weights *= 2.5 / weights.sum(1, keepdim=True)
    if router.is_sm90:
        # Warm the JIT on the default stream, then capture on a fresh stream.
        router.select_experts(x, x, torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if not router.is_sm90:
            for _ in range(3):
                router.select_experts(x, x, torch.int32)
        if use_cudagraph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual_weights, actual_ids = router.select_experts(x, x, torch.int32)
            graph.replay()
        else:
            actual_weights, actual_ids = router.select_experts(x, x, torch.int32)
    torch.cuda.current_stream().wait_stream(stream)
    assert_routing_results_close(actual_weights, actual_ids, weights, ids.int())


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Requires SM90"
)
@pytest.mark.parametrize("num_tokens", [0, 1, 17, 64, 65, 128, 129, 255, 256])
@pytest.mark.parametrize("with_bias", [False, True])
def test_sm90_mega_gate_matches_unbiased_weights(
    require_mega_gate, num_tokens, with_bias
):
    """Correction bias selects experts but must not enter normalized weights."""
    torch.manual_seed(42)
    # Contiguous views with nonzero storage offsets exercise pointer alignment.
    x = torch.randn(num_tokens + 1, 6144, device="cuda", dtype=torch.bfloat16)[1:]
    w = torch.randn(257, 6144, device="cuda", dtype=torch.bfloat16)[1:] * 0.01
    bias = torch.randn(257, device="cuda")[1:] if with_bias else None
    weights, ids = bf16_mega_gate(
        x,
        w,
        8,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        ep_rank=0,
        bias=bias,
    )
    scores = torch.mm(x, w.T, out_dtype=torch.float32).sigmoid()
    ref_ids = (scores if bias is None else scores + bias).topk(8).indices
    ref_weights = scores.gather(1, ref_ids)
    ref_weights *= 2.5 / ref_weights.sum(1, keepdim=True)
    order, ref_order = ids.argsort(1), ref_ids.argsort(1)
    torch.testing.assert_close(ids.gather(1, order), ref_ids.gather(1, ref_order))
    torch.testing.assert_close(
        weights.gather(1, order),
        ref_weights.gather(1, ref_order),
        rtol=1e-3,
        atol=1e-4,
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Requires SM90"
)
@pytest.mark.parametrize(
    "num_tokens,use_native",
    [
        (16, False),
        (32, True),
        (65, True),
        (128, True),
        (129, False),
        (255, False),
        (256, True),
        (257, False),
        (512, False),
        (1024, False),
        (3583, False),
        (3584, True),
        (3585, True),
        (4095, True),
        (4096, True),
    ],
)
def test_mega_gate_sm90_megamoe_consumes_routes_once(
    require_mega_gate, glm_mega_gate, num_tokens, use_native
):
    """The model-owned NVFP4 path consumes routes without a second gate call."""
    from vllm.models.deepseek_v32.nvidia.mega_moe import DeepseekV32MegaMoE

    gate, fallback = glm_mega_gate
    gate.weight = torch.nn.Parameter(
        torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16) * 0.01
    )
    fallback.e_score_correction_bias = fallback.e_score_correction_bias.cuda()
    model = DeepseekV32MegaMoE.__new__(DeepseekV32MegaMoE)
    torch.nn.Module.__init__(model)
    model.is_sequence_parallel = True
    model.gate = gate
    model.mega_gate = MegaGateRouter(fallback, gate)
    model.shared_experts = None
    model.experts = Mock(
        side_effect=lambda x, weights, ids: x * weights.sum(1)[:, None]
    )
    x = torch.randn(num_tokens, 6144, device="cuda", dtype=torch.bfloat16)
    result = model(x, already_sequence_parallel=True)
    torch.testing.assert_close(result, x.float() * 2.5)
    assert model.experts.call_count == 1
    assert gate.forward.call_count == (0 if use_native else 1)


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Requires SM90"
)
@pytest.mark.parametrize("num_tokens", [17, 65, 128, 256, 4096])
def test_sm90_mega_gate_padding_does_not_contaminate_valid_routes(
    require_mega_gate, glm_mega_gate, monkeypatch, num_tokens
):
    from vllm.model_executor.layers.fused_moe.router import mega_gate_router

    x = torch.zeros(num_tokens, 6144, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(256, 6144, device="cuda", dtype=torch.bfloat16)
    x[-1] = float("nan")
    mask = torch.arange(num_tokens, device="cuda") < num_tokens - 1
    gate, fallback = glm_mega_gate
    gate.weight = torch.nn.Parameter(w)
    fallback.e_score_correction_bias = None
    monkeypatch.setattr(mega_gate_router, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mega_gate_router,
        "get_forward_context",
        lambda: SimpleNamespace(is_padding=~mask),
    )
    weights, ids = MegaGateRouter(fallback, gate).select_experts(x, x)
    # Equal valid scores use the lowest IDs; padded rows are ignored by experts.
    torch.testing.assert_close(
        ids[:-1], torch.arange(8, device="cuda").expand(num_tokens - 1, 8)
    )
    torch.testing.assert_close(weights[:-1], torch.full_like(weights[:-1], 2.5 / 8))
    assert (ids[-1] == -1).all()
    assert (weights[-1] == 0).all()


def test_degenerate_grouped_config_uses_standard_topk() -> None:
    router = create_fused_moe_router(
        top_k=4,
        global_num_experts=128,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="softmax",
        renormalize=True,
    )

    assert isinstance(router, FusedTopKRouter)
    hidden_states, router_logits = make_test_data(32, 256, 128)

    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)
    baseline_weights, baseline_ids = baseline_fused_topk(
        router_logits,
        top_k=4,
        renormalize=True,
    )

    assert_routing_results_close(
        topk_weights,
        topk_ids,
        baseline_weights,
        baseline_ids,
    )


def test_multiple_expert_groups_use_grouped_topk() -> None:
    router = create_fused_moe_router(
        top_k=4,
        global_num_experts=128,
        use_grouped_topk=True,
        num_expert_group=8,
        topk_group=4,
        scoring_func="softmax",
        renormalize=True,
    )

    assert isinstance(router, GroupedTopKRouter)


def test_degenerate_grouped_config_with_bias_uses_topk_bias() -> None:
    router = create_fused_moe_router(
        top_k=4,
        global_num_experts=128,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="softmax",
        renormalize=True,
        e_score_correction_bias=torch.empty(128),
    )

    assert isinstance(router, FusedTopKBiasRouter)


def test_degenerate_grouped_config_with_bias_keeps_routed_scale() -> None:
    router = create_fused_moe_router(
        top_k=4,
        global_num_experts=128,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="softmax",
        renormalize=True,
        routed_scaling_factor=1.1,
        e_score_correction_bias=torch.empty(128),
    )

    assert isinstance(router, FusedTopKBiasRouter)
    assert router.routed_scaling_factor == 1.1


def test_degenerate_deepseek_v3_routing_stays_grouped() -> None:
    router = create_fused_moe_router(
        top_k=4,
        global_num_experts=128,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        renormalize=True,
        e_score_correction_bias=torch.empty(128),
    )

    assert isinstance(router, GroupedTopKRouter)
    assert router.routing_method_type == RoutingMethodType.DeepSeekV3


def test_single_expert_group_with_non_unit_scale_uses_grouped_topk() -> None:
    router = create_fused_moe_router(
        top_k=4,
        global_num_experts=128,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        scoring_func="softmax",
        renormalize=True,
        routed_scaling_factor=1.1,
    )

    assert isinstance(router, GroupedTopKRouter)


def setup_eplb_state(
    enable_eplb: bool, global_num_experts: int
) -> EplbLayerState | None:
    if not enable_eplb:
        return None

    # Initialize EPLB state with proper tensors for testing
    # For testing purposes, we use a simple 1:1 mapping (no redundant experts)
    # expert_load_view: tracks load on each expert (shape: num_experts)
    expert_load_view = torch.zeros(global_num_experts, dtype=torch.int32, device="cuda")

    # logical_to_physical_map: maps logical experts to physical experts
    # Shape: (num_logical_experts, max_slots)
    # For testing, use simple 1:1 mapping with single slot per expert
    logical_to_physical_map = torch.arange(
        global_num_experts, dtype=torch.int64, device="cuda"
    ).unsqueeze(-1)

    # logical_replica_count: number of replicas per logical expert
    # Shape: (num_logical_experts,)
    # For testing, each logical expert has exactly 1 replica
    logical_replica_count = torch.ones(
        global_num_experts, dtype=torch.int64, device="cuda"
    )
    should_record_tensor = torch.ones((), dtype=torch.bool, device="cuda")
    num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32, device="cuda")]

    return EplbLayerState(
        expert_load_view=expert_load_view,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
        should_record_tensor=should_record_tensor,
        num_unpadded_tokens_tensors=num_unpadded_tokens_tensors,
    )


def make_test_data(
    m: int, k: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn((m, k), device="cuda") / 10
    logits = torch.randn((m, num_experts), device="cuda")
    return hidden_states, logits


def make_e_score_correction_bias(
    e_score_correction_bias_val: float,
    num_experts: int,
) -> torch.Tensor:
    # return torch.randn(num_experts, device="cuda") * e_score_correction_bias_val
    return torch.full(
        (num_experts,), e_score_correction_bias_val, device="cuda", dtype=torch.float32
    )


def assert_routing_results_close(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    baseline_weights: torch.Tensor,
    baseline_ids: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """
    Compare routing results, sorting by expert ID first to handle non-deterministic
    ordering from sorted=False in topk.
    """
    # Sort both results by expert IDs for consistent comparison
    sorted_indices_actual = torch.argsort(topk_ids, dim=-1)
    sorted_indices_baseline = torch.argsort(baseline_ids.to(topk_ids.dtype), dim=-1)

    # Gather the sorted values
    topk_ids_sorted = torch.gather(topk_ids, 1, sorted_indices_actual)
    topk_weights_sorted = torch.gather(topk_weights, 1, sorted_indices_actual)
    baseline_ids_sorted = torch.gather(
        baseline_ids.to(topk_ids.dtype), 1, sorted_indices_baseline
    )
    baseline_weights_sorted = torch.gather(baseline_weights, 1, sorted_indices_baseline)

    # Compare
    torch.testing.assert_close(topk_ids_sorted, baseline_ids_sorted)
    torch.testing.assert_close(
        topk_weights_sorted, baseline_weights_sorted, rtol=rtol, atol=atol
    )


def assert_aiter_routing_valid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
):
    """Validate AITER routing outputs are structurally correct.

    AITER grouped_topk is a fundamentally different implementation from
    the Python baseline (different group selection, scoring internals),
    so numerical comparison is not meaningful. Instead we verify the
    outputs satisfy the routing contract: correct shapes, valid expert
    IDs, non-negative weights, and proper normalization."""
    n_tokens = topk_weights.shape[0]

    # Shape
    assert topk_weights.shape == (n_tokens, top_k), (
        f"weights shape {topk_weights.shape} != ({n_tokens}, {top_k})"
    )
    assert topk_ids.shape == (n_tokens, top_k), (
        f"ids shape {topk_ids.shape} != ({n_tokens}, {top_k})"
    )

    # Expert IDs in valid range
    assert (topk_ids >= 0).all() and (topk_ids < num_experts).all(), (
        f"expert IDs out of range [0, {num_experts}): "
        f"min={topk_ids.min().item()}, max={topk_ids.max().item()}"
    )

    # No duplicate expert IDs per token
    for i in range(n_tokens):
        ids = topk_ids[i]
        assert ids.unique().numel() == top_k, (
            f"token {i}: duplicate expert IDs {ids.tolist()}"
        )

    # Weights are non-negative
    assert (topk_weights >= 0).all(), "negative routing weights"

    # If renormalized, weights should sum to ~scaling_factor per token
    # (renormalization to 1.0 happens before scaling)
    if renormalize:
        expected_sum = routed_scaling_factor
        sums = topk_weights.sum(dim=-1)
        torch.testing.assert_close(
            sums,
            torch.full_like(sums, expected_sum),
            atol=1e-3,
            rtol=1e-3,
        )


def baseline_fused_topk(
    router_logits: torch.Tensor, top_k: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for standard fused top-k routing.

    Algorithm:
    1. Apply softmax to router logits
    2. Select top-k experts
    3. Optionally renormalize the weights
    """
    scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    # Use sorted=False to match vllm implementation (vllm_is_batch_invariant
    # defaults to False)
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def baseline_fused_topk_bias(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for fused top-k with bias correction.

    Algorithm:
    1. Apply softmax to router logits
    2. Add bias to scores for expert selection
    3. Select top-k experts using biased scores
    4. Get weights from original (unbiased) scores
    5. Apply routed scaling factor
    6. Optionally renormalize the weights
    """
    # Apply softmax to get scores
    scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)

    # Add bias for expert selection
    scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)

    # Select top-k using biased scores (sorted=False to match implementation)
    topk_ids = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]

    # Get weights from original scores (not biased)
    topk_weights = scores.gather(1, topk_ids)

    # Renormalize if needed (BEFORE applying scaling factor)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Apply scaling factor (AFTER renormalization, if applicable)
    if routed_scaling_factor != 1.0:
        topk_weights *= routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def baseline_grouped_topk(
    router_logits: torch.Tensor,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for grouped top-k routing (e.g., DeepSeek).

    Algorithm:
    1. Apply scoring function (softmax or sigmoid)
    2. Optionally add bias
    3. Select top-k groups based on max scores within each group
    4. Mask scores to only include selected groups
    5. Select top-k experts from masked scores
    6. Apply scaling factor
    7. Optionally renormalize
    """
    num_token = router_logits.shape[0]

    # Apply scoring function
    if scoring_func == "softmax":
        scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(router_logits.float())
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # Handle bias correction
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        # For bias case, use sum of top-2 scores in each group
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        # Use max score in each group
        group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values

    # Select top-k groups
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]

    # Create mask for selected groups
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Expand mask to all experts
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )

    # Mask scores (set non-selected to -inf)
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    # Select top-k experts
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)

    # Renormalize if needed
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Apply scaling factor
    if routed_scaling_factor != 1.0:
        topk_weights *= routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def baseline_custom_llama4(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline for Llama4 custom routing.

    Algorithm:
    1. Select top-k expert indices (without softmax)
    2. Apply sigmoid to the selected scores
    """
    router_scores, router_indices = torch.topk(router_logits, top_k, dim=-1)
    router_scores = torch.sigmoid(router_scores.float())
    return router_scores.to(torch.float32), router_indices.to(torch.int32)


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
def test_fused_topk(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
):
    if top_k > global_num_experts:
        pytest.skip(f"top_k ({top_k}) > global_num_experts ({global_num_experts})")

    eplb_state = setup_eplb_state(enable_eplb, global_num_experts)
    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)

    # Get router output
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)

    # Compute baseline
    baseline_weights, baseline_ids = baseline_fused_topk(
        router_logits, top_k, renormalize
    )

    # Compare results
    assert_routing_results_close(topk_weights, topk_ids, baseline_weights, baseline_ids)


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("e_score_correction_bias_val", [0.9])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 1.1])
def test_fused_topk_bias(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
    e_score_correction_bias_val: float,
    routed_scaling_factor: float,
):
    if top_k > global_num_experts:
        pytest.skip(f"top_k ({top_k}) > global_num_experts ({global_num_experts})")

    eplb_state = setup_eplb_state(enable_eplb, global_num_experts)

    e_score_correction_bias = make_e_score_correction_bias(
        e_score_correction_bias_val,
        global_num_experts,
    )

    router = create_fused_moe_router(
        e_score_correction_bias=e_score_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)

    # Get router output
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)

    # Compute baseline
    baseline_weights, baseline_ids = baseline_fused_topk_bias(
        router_logits,
        top_k,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
    )

    # Compare results
    assert_routing_results_close(topk_weights, topk_ids, baseline_weights, baseline_ids)


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize(
    "global_num_experts,num_expert_group,topk_group",
    [
        (64, 8, 4),  # 8 groups of 8 experts, select 4 groups
        (32, 4, 2),  # 4 groups of 8 experts, select 2 groups
    ],
)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("e_score_correction_bias_val", [0.9])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 1.1])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
def test_grouped_topk(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    e_score_correction_bias_val: float,
    routed_scaling_factor: float,
):
    if top_k > global_num_experts:
        pytest.skip(f"top_k ({top_k}) > global_num_experts ({global_num_experts})")

    eplb_state = setup_eplb_state(enable_eplb, global_num_experts)

    e_score_correction_bias = make_e_score_correction_bias(
        e_score_correction_bias_val,
        global_num_experts,
    )

    router = create_fused_moe_router(
        use_grouped_topk=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
        top_k=top_k,
        global_num_experts=global_num_experts,
        renormalize=renormalize,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)

    # Compute baseline (pure Python implementation)
    baseline_weights, baseline_ids = baseline_grouped_topk(
        router_logits,
        top_k,
        num_expert_group,
        topk_group,
        scoring_func,
        renormalize,
        e_score_correction_bias,
        routed_scaling_factor,
    )

    # Test 1: Python/Triton path against baseline (exact match)
    with patch(
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router.rocm_aiter_ops.is_fused_moe_enabled",
        return_value=False,
    ):
        py_weights, py_ids = router.select_experts(hidden_states, router_logits)
    assert_routing_results_close(py_weights, py_ids, baseline_weights, baseline_ids)

    # Test 2: AITER path — verify outputs are structurally valid.
    # AITER grouped_topk is a different implementation so we can't
    # compare numerically against the Python baseline.
    if _is_aiter_capable():
        # Force-enable AITER for gfx942/gfx950 regardless of env var,
        # so CI always exercises this path on capable hardware.
        with patch.object(rocm_aiter_ops, "_AITER_ENABLED", True):
            aiter_weights, aiter_ids = router.select_experts(
                hidden_states, router_logits
            )
        assert_aiter_routing_valid(
            aiter_weights,
            aiter_ids,
            top_k,
            global_num_experts,
            renormalize,
            routed_scaling_factor,
        )


@pytest.mark.parametrize("m,k", MK_S)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("enable_eplb", [False, True])
@pytest.mark.parametrize("custom_routing_function", [Llama4MoE.custom_routing_function])
def test_custom(
    m: int,
    k: int,
    top_k: int,
    global_num_experts: int,
    renormalize: bool,
    enable_eplb: bool,
    custom_routing_function: Callable,
):
    if top_k > global_num_experts:
        pytest.skip(f"top_k ({top_k}) > global_num_experts ({global_num_experts})")

    eplb_state = setup_eplb_state(enable_eplb, global_num_experts)

    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        custom_routing_function=custom_routing_function,
        renormalize=renormalize,
        eplb_state=eplb_state,
    )

    hidden_states, router_logits = make_test_data(m, k, global_num_experts)

    # Get router output
    topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)

    # Compute baseline (Llama4 uses sigmoid)
    baseline_weights, baseline_ids = baseline_custom_llama4(router_logits, top_k)

    # Compare results
    assert_routing_results_close(topk_weights, topk_ids, baseline_weights, baseline_ids)


# TODO: is other test sufficient?
# # See tests/test_routing_simulatator.py
# @pytest.mark.parametrize("m,k", MK_S)
# @pytest.mark.parametrize("top_k", TOP_KS)
# @pytest.mark.parametrize("global_num_experts", NUM_EXPERTS)
# @pytest.mark.parametrize("renormalize", [False, True])
# @pytest.mark.parametrize("enable_eplb", [False, True])
# @pytest.mark.parameterize("strategy", ["uniform_random", "normal_routing"])
# def test_simulated(
#     m: int,
#     k: int,
#     top_k: int,
#     global_num_experts: int,
#     renormalize: bool,
#     enable_eplb: bool,
#     strategy: str,
#     monkeypatch,
# ):
#     eplb_state = setup_eplb_state(enable_eplb)

#     monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", strategy)
#     router = create_fused_moe_router(
#         top_k=top_k,
#         global_num_experts=global_num_experts,
#         eplb_state=eplb_state,
#     )

#     hidden_states, router_logits = make_test_data(m, k, global_num_experts)
#     topk_weights, topk_ids = router.select_experts(hidden_states, router_logits)


# ---------------------------------------------------------------------------
# Tests for eplb_map_to_physical_and_record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("record_enabled", [True, False])
@pytest.mark.parametrize(
    "l2p_map, replica_count, num_physical, topk_ids, expected_out, expected_load",
    [
        pytest.param(
            # logical i → physical i
            [[0], [1], [2], [3]],
            [1, 1, 1, 1],
            4,
            [[0, 1], [2, 3], [0, 2]],
            [[0, 1], [2, 3], [0, 2]],
            [2, 1, 2, 1],
            id="identity",
        ),
        pytest.param(
            # logical 0→3, 1→0, 2→1, 3→2
            [[3], [0], [1], [2]],
            [1, 1, 1, 1],
            4,
            [[0, 1], [2, 3], [0, 2]],
            [[3, 0], [1, 2], [3, 1]],
            [1, 2, 1, 2],
            id="shuffled",
        ),
        pytest.param(
            # logical 0→5, 1→2, 2→7, 3→0 in a larger physical space
            [[5], [2], [7], [0]],
            [1, 1, 1, 1],
            8,
            [[0, 1], [2, 3]],
            [[5, 2], [7, 0]],
            [1, 0, 1, 0, 0, 1, 0, 1],
            id="sparse",
        ),
    ],
)
def test_eplb_map_no_redundancy(
    record_enabled,
    l2p_map,
    replica_count,
    num_physical,
    topk_ids,
    expected_out,
    expected_load,
):
    l2p = torch.tensor(l2p_map, dtype=torch.int64, device="cuda")
    rc = torch.tensor(replica_count, dtype=torch.int64, device="cuda")
    load = torch.zeros(num_physical, dtype=torch.int32, device="cuda")
    rec = torch.tensor(record_enabled, dtype=torch.bool, device="cuda")
    ids = torch.tensor(topk_ids, dtype=torch.int32, device="cuda")

    out = eplb_map_to_physical_and_record(
        topk_ids=ids,
        expert_load_view=load,
        logical_to_physical_map=l2p,
        logical_replica_count=rc,
        record_enabled=rec,
    )

    exp_out = torch.tensor(expected_out, dtype=out.dtype, device="cuda")
    torch.testing.assert_close(out, exp_out)

    if record_enabled:
        exp_load = torch.tensor(expected_load, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(load, exp_load)
    else:
        assert load.sum().item() == 0


@pytest.mark.parametrize("top_k,R", [(2, 2), (4, 2), (8, 4), (8, 8)])
def test_eplb_map_hot_expert_replica_balance(top_k, R):
    """Hot logical expert with R replicas must be balanced across replicas
    even when ``top_k`` is a multiple of ``R``. In that regime every top-k
    offset for the hot expert lands on a multiple of ``top_k`` in the flat
    ``topk_ids`` view, so per-replica assignment must not collapse onto a
    single replica.
    """
    num_tokens = 8192
    num_logical = 16
    num_physical = R + (num_logical - 1)

    l2p = torch.full((num_logical, R), -1, dtype=torch.int64, device="cuda")
    l2p[0] = torch.arange(R, dtype=torch.int64, device="cuda")
    for i in range(1, num_logical):
        l2p[i, 0] = R + i - 1
    rc = torch.tensor([R] + [1] * (num_logical - 1), dtype=torch.int64, device="cuda")

    torch.manual_seed(0)
    topk_ids = torch.randint(
        1,
        num_logical,
        (num_tokens, top_k),
        dtype=torch.int32,
        device="cuda",
    )
    topk_ids[:, 0] = 0

    load = torch.zeros(num_physical, dtype=torch.int32, device="cuda")
    rec = torch.tensor(True, dtype=torch.bool, device="cuda")

    eplb_map_to_physical_and_record(
        topk_ids=topk_ids,
        expert_load_view=load,
        logical_to_physical_map=l2p,
        logical_replica_count=rc,
        record_enabled=rec,
    )

    hot_load = load[:R].float()
    max_mean = (hot_load.max() / hot_load.mean()).item()
    assert max_mean < 1.15, (
        f"Hot expert replicas uneven: {hot_load.tolist()}, max/mean={max_mean:.3f}"
    )


@pytest.mark.parametrize("record_enabled", [True, False])
@pytest.mark.parametrize(
    "l2p_map, replica_count, num_physical, topk_ids, expected_out, expected_load",
    [
        pytest.param(
            # experts 0,1 have 2 replicas; 2,3 have 1
            [[0, 4], [1, 5], [2, -1], [3, -1]],
            [2, 2, 1, 1],
            6,
            [[0, 1], [2, 3], [0, 2]],
            # replica = (token_idx * KNUTH) & 0xFFFFFFFF % R.
            # token 0 hash=0x00000000: %2=0, %1=0.
            # token 1 hash=0x9E3779B9: %2=1, %1=0.
            # token 2 hash=0x3C6EF372: %2=0, %1=0.
            [[0, 1], [2, 3], [0, 2]],
            [2, 1, 2, 1, 0, 0],
            id="partial",
        ),
        pytest.param(
            # all 4 experts have 2 replicas
            [[0, 4], [1, 5], [2, 6], [3, 7]],
            [2, 2, 2, 2],
            8,
            [[0, 1], [2, 3], [0, 2]],
            # token 0 hash=0x00000000: %2=0.
            # token 1 hash=0x9E3779B9: %2=1.
            # token 2 hash=0x3C6EF372: %2=0.
            [[0, 1], [6, 7], [0, 2]],
            [2, 1, 1, 0, 0, 0, 1, 1],
            id="full",
        ),
        pytest.param(
            # expert 0: 4 replicas, experts 1,2: 2 replicas
            [[0, 3, 5, 7], [1, 4, -1, -1], [2, 6, -1, -1]],
            [4, 2, 2],
            8,
            [[0, 1], [2, 0], [1, 2]],
            # token 0 hash=0x00000000: %4=0, %2=0.
            # token 1 hash=0x9E3779B9: %4=1, %2=1.
            # token 2 hash=0x3C6EF372: %4=2, %2=0.
            [[0, 1], [6, 3], [1, 2]],
            [1, 2, 1, 1, 0, 0, 1, 0],
            id="uneven",
        ),
    ],
)
def test_eplb_map_with_redundancy(
    record_enabled,
    l2p_map,
    replica_count,
    num_physical,
    topk_ids,
    expected_out,
    expected_load,
):
    l2p = torch.tensor(l2p_map, dtype=torch.int64, device="cuda")
    rc = torch.tensor(replica_count, dtype=torch.int64, device="cuda")
    load = torch.zeros(num_physical, dtype=torch.int32, device="cuda")
    rec = torch.tensor(record_enabled, dtype=torch.bool, device="cuda")
    ids = torch.tensor(topk_ids, dtype=torch.int32, device="cuda")

    out = eplb_map_to_physical_and_record(
        topk_ids=ids,
        expert_load_view=load,
        logical_to_physical_map=l2p,
        logical_replica_count=rc,
        record_enabled=rec,
    )

    exp_out = torch.tensor(expected_out, dtype=out.dtype, device="cuda")
    torch.testing.assert_close(out, exp_out)

    if record_enabled:
        exp_load = torch.tensor(expected_load, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(load, exp_load)
    else:
        assert load.sum().item() == 0


@pytest.mark.parametrize(
    "l2p_map, replica_count, num_physical, topk_ids, "
    "num_unpadded, expected_out, expected_load",
    [
        pytest.param(
            [[0], [1], [2], [3]],
            [1, 1, 1, 1],
            4,
            [[0, 1], [2, 3], [0, 2], [1, 3]],
            2,
            [[0, 1], [2, 3], [0, 2], [1, 3]],
            # only rows 0,1 counted: expert 0→1, 1→1, 2→1, 3→1
            [1, 1, 1, 1],
            id="half_padded",
        ),
        pytest.param(
            # record everything (None = no padding info)
            [[0], [1], [2], [3]],
            [1, 1, 1, 1],
            4,
            [[0, 1], [2, 3], [0, 2], [1, 3]],
            None,
            [[0, 1], [2, 3], [0, 2], [1, 3]],
            [2, 2, 2, 2],
            id="no_padding_info",
        ),
    ],
)
def test_eplb_map_num_unpadded_tokens(
    l2p_map,
    replica_count,
    num_physical,
    topk_ids,
    num_unpadded,
    expected_out,
    expected_load,
):
    l2p = torch.tensor(l2p_map, dtype=torch.int64, device="cuda")
    rc = torch.tensor(replica_count, dtype=torch.int64, device="cuda")
    load = torch.zeros(num_physical, dtype=torch.int32, device="cuda")
    rec = torch.tensor(True, dtype=torch.bool, device="cuda")
    ids = torch.tensor(topk_ids, dtype=torch.int32, device="cuda")
    num_unpadded_t = (
        torch.tensor(num_unpadded, dtype=torch.int32, device="cuda")
        if num_unpadded is not None
        else None
    )

    out = eplb_map_to_physical_and_record(
        topk_ids=ids,
        expert_load_view=load,
        logical_to_physical_map=l2p,
        logical_replica_count=rc,
        record_enabled=rec,
        num_unpadded_tokens=num_unpadded_t,
    )

    exp_out = torch.tensor(expected_out, dtype=out.dtype, device="cuda")
    torch.testing.assert_close(out, exp_out)

    exp_load = torch.tensor(expected_load, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(load, exp_load)
