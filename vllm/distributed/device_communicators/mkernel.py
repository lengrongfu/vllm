# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, eager-only mKernel GEMM/all-reduce correctness prototype.

The configuration and admission policy intentionally have no torch imports so
they can be validated without loading either native library.
"""

import hashlib
import importlib.util
import ipaddress
import json
import logging
import os
import re
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MKernelConfig:
    root: str
    prefix: str
    shape: tuple[int, int, int]  # M, N, local K
    node_ips: tuple[str, str]
    broker_key: str
    port: int = 19730

    @classmethod
    def from_json(cls, value: str) -> "MKernelConfig":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("VLLM_MKERNEL_CONFIG must be a JSON object")
        config = cls(**data)
        if not isinstance(config.root, str) or not Path(config.root).is_absolute():
            raise ValueError("mKernel root must be an absolute path")
        if not isinstance(config.prefix, str) or not config.prefix:
            raise ValueError("mKernel requires one exact nonempty layer prefix")
        shape = config.shape
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError("mKernel shape must be [M, N, local_K]")
        if any(type(d) is not int for d in shape):
            raise ValueError("mKernel dimensions must be integers")
        m, n, k = shape
        if not (
            1024 <= m <= 32768
            and m % 1024 == 0
            and 256 <= n <= 32768
            and n % 256 == 0
            and 64 <= k <= 16384
            and k % 64 == 0
            and m * n * 2 <= 512 * 1024 * 1024
        ):
            raise ValueError("mKernel shape exceeds the prototype's tile/memory limits")
        if (
            not isinstance(config.node_ips, (list, tuple))
            or len(config.node_ips) != 2
            or any(not isinstance(ip, str) or not ip for ip in config.node_ips)
            or config.node_ips[0] == config.node_ips[1]
        ):
            raise ValueError("mKernel requires two distinct data-plane node addresses")
        for address in config.node_ips:
            ipaddress.IPv4Address(address)
        if not isinstance(config.broker_key, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,40}", config.broker_key
        ):
            raise ValueError("mKernel broker_key must be a unique 1-40 character ID")
        if type(config.port) is not int or not 1024 <= config.port <= 65528:
            raise ValueError("mKernel port must leave room for eight rank ports")
        return cls(
            config.root,
            config.prefix,
            tuple(shape),
            tuple(config.node_ips),
            config.broker_key,
            config.port,
        )

    def supports(
        self,
        *,
        input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        bf16: bool,
        cuda: bool,
        contiguous: bool,
        unquantized: bool,
        reduce_results: bool,
        has_bias: bool,
        skip_bias_add: bool,
        tp_size: int,
        requires_grad: bool,
    ) -> bool:
        m, n, k = self.shape
        return (
            input_shape == (m, k)
            and weight_shape == (n, k)
            and bf16
            and cuda
            and contiguous
            and unquantized
            and reduce_results
            and (not has_bias or skip_bias_add)
            and tp_size == 16
            and not requires_grad
        )


def validate_build_config(build: dict[str, Any]) -> None:
    expected = {
        "abi_version": 1,
        "intra_num_devices": 8,
        "row_block": 128,
        "col_block": 256,
        "k_block": 64,
        "num_blocks": 132,
        "cuda_arch": "hopper",
        "dtype": "bfloat16",
    }
    if any(build.get(key) != value for key, value in expected.items()):
        raise ValueError("mKernel build does not match the prototype ABI/topology")
    if build.get("backend") not in ("cx7", "efa"):
        raise ValueError("mKernel requires the CX7 or EFA transport")


def agree_on_call(decisions: list[tuple[bool, tuple[int, ...]]]) -> bool:
    """Never launch a collective if even one rank rejects or changes shape."""
    return (
        bool(decisions)
        and all(item == decisions[0] for item in decisions)
        and all(item[0] for item in decisions)
    )


class MKernelRuntime:
    def __init__(self, config: MKernelConfig, group: Any, vllm_config: Any):
        import torch
        import torch.distributed as dist

        from vllm import envs
        from vllm.config.compilation import CompilationMode, CUDAGraphMode

        self.config = config
        self.group = group
        self.module: Any = None
        self.load_attempted = False
        self.successful_calls = 0
        self.lock = threading.Lock()
        parallel = vllm_config.parallel_config
        model = vllm_config.model_config
        tuning = {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("GEMM_AR_", "MKERNEL_", "Q2_", "Q5_"))
        }
        supported = (
            group.world_size == dist.get_world_size() == 16
            and group.ranks == list(range(16))
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and not parallel.enable_expert_parallel
            and not parallel.use_ubatching
            and parallel.decode_context_parallel_size == 1
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
            and model is not None
            and model.enforce_eager
            and vllm_config.compilation_config.mode == CompilationMode.NONE
            and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            and vllm_config.speculative_config is None
            and vllm_config.lora_config is None
            and not envs.VLLM_BATCH_INVARIANT
            and torch.cuda.is_available()
            and not torch.version.hip
            and torch.cuda.device_count() == 8
        )
        if supported:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            supported = (
                device == group.local_rank == group.rank_in_group % 8
                and (props.major, props.minor) == (9, 0)
                and props.multi_processor_count == 132
            )
        metadata = self._gather((config, supported, socket.gethostname(), tuning))
        if any(item[0] != config or not item[1] or item[3] for item in metadata):
            raise ValueError(
                "mKernel requires identical configuration, 2x8 full Hopper TP, "
                "eager mode, no LoRA/speculation/batch invariance or native tuning"
            )
        hosts = [item[2] for item in metadata]
        if len(set(hosts[:8])) != 1 or len(set(hosts[8:])) != 1 or hosts[0] == hosts[8]:
            raise ValueError("mKernel requires node-major ranks on two distinct hosts")
        broker = os.environ.get("DIST_BROKER_KEY", config.broker_key)
        if any(key != config.broker_key for key in self._gather(broker)):
            raise ValueError("DIST_BROKER_KEY conflicts with mKernel broker_key")
        os.environ["DIST_BROKER_KEY"] = config.broker_key
        os.environ["MKERNEL_BIND_RETAINED_HANDLE"] = "1"

    def _gather(self, value: Any) -> list[Any]:
        import torch.distributed as dist

        values: list[Any] = [None] * self.group.world_size
        dist.all_gather_object(values, value, group=self.group.cpu_group)
        return values

    def _load(self) -> bool:
        from vllm.logger import init_logger

        if self.load_attempted:
            return self.module is not None
        self.load_attempted = True
        error = None
        fingerprint = None
        try:
            path = Path(self.config.root) / "build" / "libgemm_ar.so"
            with path.open("rb") as library:
                fingerprint = hashlib.file_digest(library, "sha256").hexdigest()
            spec = importlib.util.spec_from_file_location(
                "mkernel_release_gemm_ar", path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            validate_build_config(module.get_build_config())
            self.module = module
        except (ImportError, OSError, AttributeError, ValueError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        states = self._gather((error, fingerprint))
        if any(item[0] for item in states) or any(
            item[1] != fingerprint for item in states
        ):
            self.module = None
            init_logger(__name__).warning(
                "mKernel unavailable or inconsistent; retaining vLLM linear: %s",
                states,
            )
        return self.module is not None

    def try_apply(self, layer: Any, x: Any) -> Any:
        import torch

        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        if layer.prefix != self.config.prefix:
            return None
        weight = getattr(layer, "weight", None)
        supported = weight is not None and self.config.supports(
            input_shape=tuple(x.shape),
            weight_shape=tuple(weight.shape),
            bf16=x.dtype == weight.dtype == torch.bfloat16,
            cuda=x.is_cuda and weight.is_cuda and x.device == weight.device,
            contiguous=x.is_contiguous() and weight.is_contiguous(),
            unquantized=type(layer.quant_method) is UnquantizedLinearMethod,
            reduce_results=layer.reduce_results,
            has_bias=layer.bias is not None,
            skip_bias_add=layer.skip_bias_add,
            tp_size=layer.tp_size,
            requires_grad=x.requires_grad or weight.requires_grad,
        )
        supported = supported and not torch.cuda.is_current_stream_capturing()
        supported = supported and x.device.index == self.group.local_rank
        supported = supported and x.data_ptr() % 16 == 0
        if not agree_on_call(self._gather((bool(supported), tuple(x.shape)))):
            return None
        if not self._load():
            return None
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Concurrent mKernel invocation is unsupported")
        try:
            result = self._run(x, weight)
            self.successful_calls += 1
            if self.successful_calls == 1 and self.group.rank_in_group == 0:
                from vllm.logger import init_logger

                init_logger(__name__).info(
                    "mKernel GEMM+AR executed: prefix=%s shape=%s backend=%s "
                    "(correctness prototype, not a performance optimization)",
                    self.config.prefix,
                    self.config.shape,
                    self.module.get_build_config()["backend"],
                )
            return result
        finally:
            self.lock.release()

    def _run(self, x: Any, weight: Any) -> Any:
        import torch
        import torch.distributed as dist

        return run_native(
            self.config,
            self.group.rank_in_group,
            self.module,
            torch,
            lambda: dist.barrier(group=self.group.cpu_group),
            x,
            weight,
        )


def run_native(config, rank, mod, torch_api, barrier_sync, x, weight):
    """Execute one owned native lifecycle; dependencies represent external APIs."""
    torch = torch_api
    try:
        m, n, _ = config.shape
        local_rank, node = rank % 8, rank // 8
        peer_ip = config.node_ips[1 - node]
        port = config.port + local_rank
        # This prototype deliberately pays packing, allocation, synchronization,
        # and session setup on every call. It makes no performance claim.
        b = weight.t().contiguous()
        c = mod.DistBuffer((m, n), torch.bfloat16, local_rank, 8, True)
        barrier = mod.DistBuffer((2, 1024, 1024), torch.int32, local_rank, 8, True)
        output = mod.DistBuffer((m, n), torch.bfloat16, local_rank, 8, True)
        c.data_.zero_()
        barrier.data_.zero_()
        output.data_.zero_()
        staging_bytes = (m // 8) * n * 2
        staging = torch.empty(staging_bytes // 2, dtype=torch.bfloat16, device=x.device)
        total_tiles = (m // 128) * (n // 256)
        fifo_capacity = max(2048, 1 << (2 * total_tiles - 1).bit_length())
        barrier_sync()
        mod.create_session(
            node,
            peer_ip,
            port,
            staging.data_ptr(),
            staging_bytes,
            staging_bytes,
            total_tiles,
            fifo_capacity,
            local_rank,
            c.data_.data_ptr(),
            m * n * 2,
            n * 2,
            peer_ips=[peer_ip],
            peer_tcp_ports=[port],
        )
        # From this point a failure is fatal, never a rank-local NCCL fallback.
        num_qps = int(mod.get_num_qps())
        if num_qps != 4:
            raise RuntimeError("mKernel prototype requires the default four QPs")
        queues = 1
        scratch_ints = mod.get_scratch_ints(m, n, queues, num_qps, 2)
        scratch = torch.zeros(scratch_ints, dtype=torch.int32, device=x.device)
        fifo = mod.get_fifo_handles()
        mod.reset_arrival_flags()
        mod.set_epoch(1)
        barrier_sync()
        mod.gemm_ar_multinode(
            x,
            b,
            c,
            barrier,
            output,
            staging.data_ptr(),
            mod.get_recv_buf_ptr(),
            *fifo,
            mod.get_arrival_flags_ptr(),
            1,
            node,
            4,
            8,
            scratch.data_ptr(),
            num_nodes=2,
            arrival_tails_ptr=mod.get_arrival_tails_ptr(),
            scratch_ints=scratch_ints,
            num_qps=num_qps,
            num_remote_queues=queues,
            num_allocated_remote_queues=num_qps,
            cross_node_barrier_ptr=mod.get_barrier_device_ptr(),
            use_acquire_poll=True,
        )
        result = output.data_.clone()
        torch.cuda.synchronize(x.device)
        barrier_sync()
        mod.destroy_session()
        # ParallelBuffer destruction exchanges IPC state; keep all ranks in the
        # same destruction order and do not let shared output escape.
        del output, barrier, c
        barrier_sync()
        return result
    except BaseException:
        logging.getLogger(__name__).critical(
            "mKernel native failure: terminating worker without collective "
            "buffer finalizers; the supervisor must terminate all ranks",
            exc_info=True,
        )
        os._exit(1)


_runtimes: dict[str, MKernelRuntime] = {}


def get_mkernel_runtime(value: str, vllm_config: Any) -> MKernelRuntime:
    import torch.distributed as dist

    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if group.unique_name not in _runtimes:
        values: list[Any] = [None] * group.world_size
        dist.all_gather_object(values, value, group=group.cpu_group)
        if any(item != value for item in values):
            raise ValueError("mKernel configuration must be identical on every rank")
        _runtimes[group.unique_name] = MKernelRuntime(
            MKernelConfig.from_json(value), group, vllm_config
        )
    return _runtimes[group.unique_name]
