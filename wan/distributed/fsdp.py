# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)
    return model


def free_model(model, device=None):
    """Release a sharded model and clear device caches.

    Args:
        model: The model instance to be freed.
        device (optional): Target device to clear cache for. If ``None``,
            the currently available accelerator is used. Accepts a
            :class:`torch.device` or a device type string such as ``"cuda"``
            or ``"mps"``.
    """

    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)

    del model
    gc.collect()

    if device is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    else:
        device_type = device.type if isinstance(device, torch.device) else str(device)

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
