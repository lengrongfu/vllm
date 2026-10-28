# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission-policy checks; these do not validate the native CUDA kernel."""

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


def _load_policy():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm/distributed/device_communicators/mkernel.py"
    )
    spec = importlib.util.spec_from_file_location("mkernel_policy_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


policy = _load_policy()


def _exercise_native(fail=False):
    events = []

    class Tensor:
        device = "cuda:1"

        def __init__(self, name, ptr):
            self.name, self.ptr = name, ptr

        def t(self):
            events.append(("transpose", self.name))
            return Tensor(self.name + ".t", self.ptr)

        def contiguous(self):
            events.append(("contiguous", self.name))
            return Tensor(self.name + ".contiguous", self.ptr)

        def zero_(self):
            events.append(("zero", self.name))

        def data_ptr(self):
            return self.ptr

        def clone(self):
            events.append(("clone", self.name))
            return Tensor(self.name + ".clone", self.ptr + 1000)

    class Buffer:
        def __init__(self, name, ptr):
            self.name = name
            self.data_ = Tensor(name, ptr)

        def __del__(self):
            events.append(("delete", self.name))
            if fail:
                print("UNSAFE_NATIVE_FINALIZER", file=sys.stderr, flush=True)

    class ExternalServices:
        bfloat16, int32 = "bf16", "int32"

        def __init__(self):
            self.cuda = self
            self.buffer_count = 0

        def DistBuffer(self, *args):
            name = ("c", "barrier", "output")[self.buffer_count]
            self.buffer_count += 1
            events.append(("buffer", name, args))
            return Buffer(name, 100 + self.buffer_count)

        def empty(self, *args, **kwargs):
            return Tensor("staging", 104)

        def zeros(self, *args, **kwargs):
            return Tensor("scratch", 105)

        def create_session(self, *args, **kwargs):
            events.append(("session", args, kwargs))

        def gemm_ar_multinode(self, *args, **kwargs):
            events.append(
                ("launch", tuple(getattr(a, "name", a) for a in args), kwargs)
            )
            if fail:
                raise RuntimeError("injected native launch failure")

        def __getattr__(self, name):
            values = {
                "get_num_qps": 4,
                "get_scratch_ints": 64,
                "get_fifo_handles": (202, 203, 204, 205, -4),
                "get_recv_buf_ptr": 201,
                "get_arrival_flags_ptr": 206,
                "get_arrival_tails_ptr": 207,
                "get_barrier_device_ptr": 208,
            }
            if name in values:
                return lambda *args: values[name]
            if name == "synchronize":
                return lambda device: events.append(("sync", device))
            if name in ("reset_arrival_flags", "set_epoch", "destroy_session"):
                return lambda *args: events.append((name, *args))
            raise AttributeError(name)

    service = ExternalServices()
    config = TestMKernelAdmission().config()
    args = (config, 9, service, service, lambda: events.append(("barrier",)))
    result = policy.run_native(*args, Tensor("x", 11), Tensor("weight", 12))
    return result, events


class TestMKernelAdmission(unittest.TestCase):
    def config(self, **overrides):
        data = {
            "root": "/opt/mkernel",
            "prefix": "model.layers.0.mlp.down_proj",
            "shape": [1024, 4096, 896],
            "node_ips": ["192.0.2.1", "192.0.2.2"],
            "broker_key": "mkernel-test-job",
        }
        data.update(overrides)
        return policy.MKernelConfig.from_json(json.dumps(data))

    def call(self, **overrides):
        options = {
            "input_shape": (1024, 896),
            "weight_shape": (4096, 896),
            "bf16": True,
            "cuda": True,
            "contiguous": True,
            "unquantized": True,
            "reduce_results": True,
            "has_bias": False,
            "skip_bias_add": False,
            "tp_size": 16,
            "requires_grad": False,
        }
        options.update(overrides)
        return self.config().supports(**options)

    def test_configuration_normalizes_rank_comparable_fields(self):
        config = self.config()
        self.assertEqual(config.shape, (1024, 4096, 896))
        self.assertEqual(config.node_ips, ("192.0.2.1", "192.0.2.2"))
        self.assertEqual(config.port, 19730)

    def test_rejects_partial_tiles_and_excessive_buffer_sizes(self):
        for shape in (
            [1, 4096, 896],
            [1025, 4096, 896],
            [1024, 4097, 896],
            [1024, 4096, 897],
            [32768, 32768, 896],
            [1024, 4096, 32768],
            [True, 4096, 896],
            [1024, 4096],
        ):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.config(shape=shape)

    def test_rejects_ambiguous_or_unsafe_transport_configuration(self):
        for override in (
            {"root": "relative/path"},
            {"prefix": ""},
            {"node_ips": ["192.0.2.1", "192.0.2.1"]},
            {"node_ips": ["192.0.2.1"]},
            {"node_ips": ["node-a.example", "node-b.example"]},
            {"node_ips": ["2001:db8::1", "2001:db8::2"]},
            {"broker_key": "../shared"},
            {"broker_key": "x" * 41},
            {"port": 65529},
            {"port": True},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.config(**override)

    def test_only_exact_supported_tensor_contract_is_admitted(self):
        self.assertTrue(self.call())
        for override in (
            {"input_shape": (1, 896)},
            {"input_shape": (1, 1024, 896)},
            {"weight_shape": (896, 4096)},
            {"bf16": False},
            {"cuda": False},
            {"contiguous": False},
            {"unquantized": False},
            {"reduce_results": False},
            {"tp_size": 8},
            {"requires_grad": True},
            {"has_bias": True},
        ):
            with self.subTest(override=override):
                self.assertFalse(self.call(**override))

    def test_deferred_bias_is_not_added_once_per_rank(self):
        self.assertTrue(self.call(has_bias=True, skip_bias_add=True))
        self.assertFalse(self.call(has_bias=True, skip_bias_add=False))

    def test_rank_disagreement_never_admits_a_native_collective(self):
        accepted = (True, (1024, 896))
        self.assertTrue(policy.agree_on_call([accepted] * 16))
        for decisions in (
            [],
            [(False, (1024, 896))] * 16,
            [accepted] * 15 + [(False, (1024, 896))],
            [accepted] * 15 + [(True, (2048, 896))],
        ):
            with self.subTest(decisions=decisions):
                self.assertFalse(policy.agree_on_call(decisions))

    def test_binary_contract_must_match_before_native_state_is_created(self):
        build = {
            "abi_version": 1,
            "intra_num_devices": 8,
            "row_block": 128,
            "col_block": 256,
            "k_block": 64,
            "num_blocks": 132,
            "cuda_arch": "hopper",
            "dtype": "bfloat16",
            "backend": "cx7",
        }
        policy.validate_build_config(build)
        policy.validate_build_config(dict(build, backend="efa"))
        for key, value in (
            ("abi_version", 0),
            ("intra_num_devices", 4),
            ("row_block", 64),
            ("col_block", 128),
            ("k_block", 32),
            ("num_blocks", 114),
            ("cuda_arch", "blackwell"),
            ("backend", "nccl"),
            ("dtype", "float16"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                policy.validate_build_config(dict(build, **{key: value}))

    def test_native_lifecycle_owns_buffers_and_preserves_pybind_argument_order(self):
        result, events = _exercise_native()
        self.assertEqual(result.name, "output.clone")
        self.assertEqual(
            events[:2], [("transpose", "weight"), ("contiguous", "weight.t")]
        )
        session = next(e for e in events if e[0] == "session")
        self.assertEqual(session[1][:4], (1, "192.0.2.1", 19731, 104))
        expected_sizes = (1048576, 1048576, 128, 2048, 1, 101, 8388608, 8192)
        self.assertEqual(session[1][4:], expected_sizes)
        self.assertEqual(
            session[2], {"peer_ips": ["192.0.2.1"], "peer_tcp_ports": [19731]}
        )
        launch = next(e for e in events if e[0] == "launch")
        tensors = ("x", "weight.t.contiguous", "c", "barrier", "output")
        handles = (104, 201, 202, 203, 204, 205, -4, 206, 1, 1, 4, 8, 105)
        self.assertEqual(launch[1], tensors + handles)
        self.assertEqual(
            launch[2],
            {
                "num_nodes": 2,
                "arrival_tails_ptr": 207,
                "scratch_ints": 64,
                "num_qps": 4,
                "num_remote_queues": 1,
                "num_allocated_remote_queues": 4,
                "cross_node_barrier_ptr": 208,
                "use_acquire_poll": True,
            },
        )
        self.assertEqual(
            events[-8:],
            [
                ("clone", "output"),
                ("sync", "cuda:1"),
                ("barrier",),
                ("destroy_session",),
                ("delete", "output"),
                ("delete", "barrier"),
                ("delete", "c"),
                ("barrier",),
            ],
        )

    def test_native_exception_terminates_worker_without_rank_local_cleanup(self):
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--native-failure-child"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(child.returncode, 1, child.stderr)
        self.assertIn("mKernel native failure: terminating worker", child.stderr)
        self.assertIn("injected native launch failure", child.stderr)
        self.assertNotIn("UNSAFE_NATIVE_FINALIZER", child.stderr)
        self.assertNotIn("RETURNED_AFTER_NATIVE_FAILURE", child.stdout)


if __name__ == "__main__":
    if "--native-failure-child" in sys.argv:
        _exercise_native(fail=True)
        print("RETURNED_AFTER_NATIVE_FAILURE")
    else:
        unittest.main()
