# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# precision utilities
_PRECISION_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def _default_dtype():
    """Detect a reasonable default dtype based on the runtime device."""
    return torch.float32 if torch.backends.mps.is_available() else torch.float16

def set_precision(precision: str | None):
    """Set the global precision for Wan models.

    Parameters
    ----------
    precision: str or None
        Desired precision string ("fp32", "fp16" or "bf16"). If ``None``, a
        device specific default will be used. When running on Apple's MPS
        backend, unsupported half-precision modes will fall back to ``fp32``.
    """
    if precision is None:
        dtype = _default_dtype()
    else:
        dtype = _PRECISION_MAP.get(precision.lower(), _default_dtype())
    if torch.backends.mps.is_available() and dtype != torch.float32:
        # Some MPS operations require float32; ensure compatibility.
        dtype = torch.float32
    wan_shared_cfg.dtype = dtype
    wan_shared_cfg.t5_dtype = dtype
    wan_shared_cfg.param_dtype = dtype
    return dtype

# Initialize with the default precision.
set_precision(None)

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_quantization = None
wan_shared_cfg.text_len = 512

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = (
    '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，'
    '低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，'
    '毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
)
wan_shared_cfg.frame_num = 81
