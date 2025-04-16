#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script for vLLM GPU power management feature.

This script initializes a vLLM engine with power management enabled and
runs a simple inference test to verify that the GPU SM Clock is adjusted
during the decode phase.
"""

import argparse
import os
import time

import torch

from vllm import LLM, SamplingParams

os.environ.setdefault("VLLM_USE_MODELSCOPE", "False")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test vLLM GPU power management")
    parser.add_argument("--model",
                        type=str,
                        default="Qwen/Qwen1.5-7B",
                        help="Model to use for testing")
    parser.add_argument(
        "--reduction",
        type=int,
        default=5,
        help="Percentage to reduce SM clock during decode (0-100)")
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU device IDs to manage")
    parser.add_argument("--disable-power-management",
                        action="store_true",
                        help="Disable power management for comparison")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize LLM with power management
    llm = LLM(
        model=args.model,
        enable_power_management=not args.disable_power_management,
        power_decode_clock_reduction_percent=args.reduction,
        power_monitor_tbt=True,
        power_tbt_threshold_ms=100.0,
        power_device_ids=args.device_ids,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=256,
    )

    # Run a simple inference test
    print("Running inference with a short prompt (mostly decode phase)...")
    prompt = "Once upon a time,"

    # Measure time and power during inference
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end_time = time.time()

    # Print results
    print(f"Inference completed in {end_time - start_time:.2f} seconds")
    print(f"Generated text: {outputs[0].outputs[0].text}")

    # Run a test with a longer prompt (more prefill phase)
    print("\nRunning inference with a longer prompt (more prefill phase)...")
    long_prompt = "Explain the theory of relativity in detail: " + "a " * 500

    # Measure time and power during inference
    start_time = time.time()
    outputs = llm.generate([long_prompt], sampling_params)
    end_time = time.time()

    # Print results
    print(f"Inference completed in {end_time - start_time:.2f} seconds")
    print(
        f"Generated text length: {len(outputs[0].outputs[0].text)} characters")

    print("\nPower management test completed.")

    # If NVML is available, print current GPU clock information
    try:
        from vllm.third_party.pynvml import (NVML_CLOCK_SM,
                                             nvmlDeviceGetClockInfo,
                                             nvmlDeviceGetHandleByIndex,
                                             nvmlDeviceGetMaxClockInfo,
                                             nvmlInit)

        nvmlInit()
        device_count = torch.cuda.device_count()

        print("\nCurrent GPU Clock Information:")
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            current_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)
            max_clock = nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM)
            print(
                f"GPU {i}: Current SM Clock: {current_clock} MHz, Max SM Clock: {max_clock} MHz"
            )
    except:
        print("\nNVML not available, cannot display GPU clock information.")


if __name__ == "__main__":
    main()
