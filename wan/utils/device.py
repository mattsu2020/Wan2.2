# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Utility helpers for device specific operations.

This module provides wrappers around device specific cache clearing and
synchronization to seamlessly support CUDA, MPS and CPU execution.
"""

from __future__ import annotations

import torch

__all__ = ["empty_device_cache", "synchronize_device", "get_best_device"]


def get_best_device(index: int | None = None) -> torch.device:
    """Return the best available device as a :class:`torch.device`.

    Preference order is CUDA, then MPS (Apple Silicon), and finally CPU.
    When CUDA is available, an optional ``index`` can be supplied to select
    a specific GPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda", index or 0)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache(device: str | torch.device | None = None) -> None:
    """Release cached memory on the given device.

    Typically invoked after offloading large models or completing
    memory-intensive allocations to return resources to the system.

    Args:
        device: Optional device specification. When ``None`` the best
            available device is used.
    """
    dev = torch.device(device) if device is not None else get_best_device()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    # CPU requires no action


def synchronize_device(device: str | torch.device | None = None) -> None:
    """Synchronize the given device if necessary.

    Useful after calling :func:`empty_device_cache` or performing
    asynchronous transfers to ensure memory is actually freed.

    Args:
        device: Optional device specification. When ``None`` the best
            available device is used.
    """
    dev = torch.device(device) if device is not None else get_best_device()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    # CPU requires no action
