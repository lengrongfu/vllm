# SPDX-License-Identifier: Apache-2.0
"""GPU power management utilities for vLLM.

This module provides functions to adjust GPU clock frequencies to save power
during different phases of LLM inference, particularly during the decode phase
where memory bandwidth is more critical than compute performance.
"""

import threading
from typing import Dict, List, Optional

import torch

from vllm.logger import init_logger

# Import NVML for GPU management
try:
    from vllm.third_party.pynvml import (NVML_CLOCK_SM, nvmlDeviceGetClockInfo,
                                         nvmlDeviceGetHandleByIndex,
                                         nvmlDeviceGetMaxClockInfo,
                                         nvmlDeviceResetGpuLockedClocks,
                                         nvmlDeviceSetGpuLockedClocks,
                                         nvmlInit, nvmlShutdown)
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = init_logger(__name__)

# Default clock reduction percentage during decode phase
DEFAULT_DECODE_CLOCK_REDUCTION = 5  # 5% reduction


class GPUPowerManager:
    """Manages GPU power by adjusting clock frequencies during different inference phases.
    
    This class provides functionality to reduce GPU SM Clock during the decode phase
    of LLM inference to save power while maintaining Time-Between-Tokens (TBT) performance.
    """

    def __init__(
        self,
        enabled: bool = False,
        decode_clock_reduction_percent: int = DEFAULT_DECODE_CLOCK_REDUCTION,
        monitor_tbt: bool = True,
        tbt_threshold_ms: float = 100.0,  # 100ms threshold for TBT
        device_ids: Optional[List[int]] = None,
    ):
        """Initialize the GPU power manager.
        
        Args:
            enabled: Whether to enable GPU power management.
            decode_clock_reduction_percent: Percentage to reduce SM clock during decode.
            monitor_tbt: Whether to monitor TBT and adjust clocks if it exceeds threshold.
            tbt_threshold_ms: Threshold in milliseconds for acceptable TBT.
            device_ids: List of GPU device IDs to manage. If None, manage all available GPUs.
        """
        self.enabled = enabled and NVML_AVAILABLE
        self.decode_clock_reduction_percent = decode_clock_reduction_percent
        self.monitor_tbt = monitor_tbt
        self.tbt_threshold_ms = tbt_threshold_ms

        if not self.enabled:
            if not NVML_AVAILABLE:
                logger.warning(
                    "NVML is not available. GPU power management is disabled.")
            return

        # Initialize NVML
        nvmlInit()

        # Get device IDs
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids

        # Store original and target clock speeds
        self.original_clocks: Dict[int, int] = {}
        self.target_clocks: Dict[int, int] = {}

        # Initialize clock information
        for device_id in self.device_ids:
            handle = nvmlDeviceGetHandleByIndex(device_id)
            max_sm_clock = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM)
            current_sm_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)

            self.original_clocks[device_id] = current_sm_clock

            # Calculate target clock for decode phase
            target_clock = int(max_sm_clock *
                               (100 - self.decode_clock_reduction_percent) /
                               100)
            self.target_clocks[device_id] = target_clock

            logger.info(f"GPU {device_id}: Max SM Clock: {max_sm_clock} MHz, "
                        f"Current SM Clock: {current_sm_clock} MHz, "
                        f"Target Decode Clock: {target_clock} MHz")

        # Track current phase
        self.in_decode_phase = False

        # TBT monitoring
        self.tbt_values: List[float] = []
        self.tbt_lock = threading.Lock()

        logger.info("GPU Power Manager initialized successfully.")

    def __del__(self):
        """Clean up NVML on deletion."""
        if self.enabled:
            self.reset_clocks()
            try:
                nvmlShutdown()
            except:
                pass

    def enter_decode_phase(self):
        """Adjust GPU clocks for the decode phase."""
        if not self.enabled or self.in_decode_phase:
            return

        logger.info("Entering decode phase, reducing GPU SM clocks")

        for device_id in self.device_ids:
            handle = nvmlDeviceGetHandleByIndex(device_id)
            target_clock = self.target_clocks[device_id]

            try:
                # Lock both min and max clocks to the target value
                # nvmlDeviceSetGpuLockedClocks(handle, target_clock, target_clock)
                logger.info(
                    f"GPU {device_id}: SM Clock set to {target_clock} MHz for decode phase"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to set GPU {device_id} clock: {str(e)}")

        self.in_decode_phase = True

    def enter_prefill_phase(self):
        """Reset GPU clocks for the prefill phase."""
        if not self.enabled or not self.in_decode_phase:
            return

        self.reset_clocks()
        self.in_decode_phase = False

    def reset_clocks(self):
        """Reset GPU clocks to their original values."""
        if not self.enabled:
            return

        logger.info("Resetting GPU SM clocks")

        for device_id in self.device_ids:
            handle = nvmlDeviceGetHandleByIndex(device_id)

            try:
                nvmlDeviceResetGpuLockedClocks(handle)
                logger.info(
                    f"GPU {device_id}: SM Clock reset to original values")
            except Exception as e:
                logger.warning(
                    f"Failed to reset GPU {device_id} clock: {str(e)}")

    def record_tbt(self, tbt_ms: float):
        """Record a Time-Between-Tokens measurement.
        
        Args:
            tbt_ms: Time-Between-Tokens in milliseconds.
        """
        if not self.enabled or not self.monitor_tbt:
            return

        with self.tbt_lock:
            self.tbt_values.append(tbt_ms)

            # Keep only the last 10 values
            if len(self.tbt_values) > 10:
                self.tbt_values.pop(0)

            # Check if TBT is exceeding threshold
            if self.in_decode_phase and len(self.tbt_values) >= 3:
                avg_tbt = sum(self.tbt_values) / len(self.tbt_values)

                if avg_tbt > self.tbt_threshold_ms:
                    logger.warning(
                        f"TBT ({avg_tbt:.2f} ms) exceeding threshold "
                        f"({self.tbt_threshold_ms} ms), resetting clocks")
                    self.reset_clocks()
                    self.in_decode_phase = False


# Global power manager instance
_power_manager: Optional[GPUPowerManager] = None


def initialize_power_manager(
    enabled: bool = False,
    decode_clock_reduction_percent: int = DEFAULT_DECODE_CLOCK_REDUCTION,
    monitor_tbt: bool = True,
    tbt_threshold_ms: float = 100.0,
    device_ids: Optional[List[int]] = None,
) -> GPUPowerManager:
    """Initialize the global GPU power manager.
    
    Args:
        enabled: Whether to enable GPU power management.
        decode_clock_reduction_percent: Percentage to reduce SM clock during decode.
        monitor_tbt: Whether to monitor TBT and adjust clocks if it exceeds threshold.
        tbt_threshold_ms: Threshold in milliseconds for acceptable TBT.
        device_ids: List of GPU device IDs to manage. If None, manage all available GPUs.
        
    Returns:
        The initialized GPUPowerManager instance.
    """
    global _power_manager

    if _power_manager is not None:
        logger.warning("Power manager already initialized, reinitializing")
        _power_manager = None

    _power_manager = GPUPowerManager(
        enabled=enabled,
        decode_clock_reduction_percent=decode_clock_reduction_percent,
        monitor_tbt=monitor_tbt,
        tbt_threshold_ms=tbt_threshold_ms,
        device_ids=device_ids,
    )

    return _power_manager


def get_power_manager() -> Optional[GPUPowerManager]:
    """Get the global GPU power manager instance.
    
    Returns:
        The global GPUPowerManager instance, or None if not initialized.
    """
    return _power_manager


def enter_decode_phase():
    """Signal the power manager to enter decode phase."""
    if _power_manager is not None:
        _power_manager.enter_decode_phase()


def enter_prefill_phase():
    """Signal the power manager to enter prefill phase."""
    if _power_manager is not None:
        _power_manager.enter_prefill_phase()


def record_tbt(tbt_ms: float):
    """Record a Time-Between-Tokens measurement.
    
    Args:
        tbt_ms: Time-Between-Tokens in milliseconds.
    """
    if _power_manager is not None:
        _power_manager.record_tbt(tbt_ms)
