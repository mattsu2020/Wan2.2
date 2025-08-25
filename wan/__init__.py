# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import os

import torch

if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
from .textimage2video import WanTI2V
