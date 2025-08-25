# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Utility helpers for device specific operations.

This module provides wrappers around device specific cache clearing and
synchronization to seamlessly support CUDA, MPS and CPU execution.
"""

from __future__ import annotations

import torch

__all__ = ["empty_device_cache", "synchronize_device"]


def _default_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache(device: str | torch.device | None = None) -> None:
    """Release cached memory on the given device.

    Args:
        device: Optional device specification. When ``None`` the best
            available device is used.
    """
    dev = torch.device(device) if device is not None else _default_device()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    # CPU requires no action


def synchronize_device(device: str | torch.device | None = None) -> None:
    """Synchronize the given device if necessary.

    Args:
        device: Optional device specification. When ``None`` the best
            available device is used.
    """
    dev = torch.device(device) if device is not None else _default_device()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    # CPU requires no action
