# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Utility helpers for device specific operations.

This module provides wrappers around device specific cache clearing and
synchronization to seamlessly support CUDA, MPS and CPU execution.
"""

from __future__ import annotations

import os
import torch

__all__ = ["get_best_device", "empty_device_cache", "synchronize_device"]


def get_best_device(device_id: int | None = None) -> torch.device:
    """Return the best available device.

    Args:
        device_id: Preferred CUDA device index when CUDA is available. If
            ``None``, the ``LOCAL_RANK`` environment variable is used when
            present, otherwise GPU ``0`` is selected.
    """
    if torch.cuda.is_available():
        if device_id is None:
            env_rank = os.environ.get("LOCAL_RANK")
            device_id = int(env_rank) if env_rank is not None else 0
        return torch.device(f"cuda:{device_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache(device: str | torch.device | None = None) -> None:
    """Release cached memory on the given device.

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
