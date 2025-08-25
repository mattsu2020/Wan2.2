# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# dtype
# Users can modify this option in configuration files to switch
# the precision of all model parameters. Half precision (float16)
# inference is supported when FSDP is disabled. Setting this option to
# ``"int8"`` or ``"nf4"`` enables bitsandbytes quantization.
if torch.backends.mps.is_available():
    default_dtype = "float16"
else:
    default_dtype = "bfloat16"
wan_shared_cfg.dtype = default_dtype

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

if wan_shared_cfg.dtype in {"int8", "nf4"}:
    # Quantized weights use float16 for computation
    wan_shared_cfg.param_dtype = torch.float16
else:
    wan_shared_cfg.param_dtype = _DTYPE_MAP.get(wan_shared_cfg.dtype,
                                                torch.bfloat16)

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = wan_shared_cfg.param_dtype
wan_shared_cfg.text_len = 512

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
wan_shared_cfg.frame_num = 81
