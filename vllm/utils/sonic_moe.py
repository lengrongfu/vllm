# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for optional SonicMoE integration."""

from __future__ import annotations

import functools
import importlib
import importlib.util

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def has_sonic_moe() -> bool:
    """Return ``True`` if the SonicMoE package is available."""
    if importlib.util.find_spec("sonicmoe") is None:
        logger.debug_once("SonicMoE unavailable since package was not found")
        return False
    return True
