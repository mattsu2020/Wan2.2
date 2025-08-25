import sys
import types
import importlib.util
from pathlib import Path

import pytest
import torch

# Dynamically load modules from the repository without importing the whole package
repo_root = Path(__file__).resolve().parents[1]
wan_path = repo_root / "wan"

# Create a minimal package structure for 'wan' and 'wan.utils'
wan_module = types.ModuleType("wan")
wan_module.__path__ = [str(wan_path)]
sys.modules.setdefault("wan", wan_module)

utils_module = types.ModuleType("wan.utils")
utils_module.__path__ = [str(wan_path / "utils")]
sys.modules.setdefault("wan.utils", utils_module)

# Load fm_solvers
spec = importlib.util.spec_from_file_location(
    "wan.utils.fm_solvers", wan_path / "utils" / "fm_solvers.py"
)
fm_solvers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm_solvers)

# Load fm_solvers_unipc
spec_unipc = importlib.util.spec_from_file_location(
    "wan.utils.fm_solvers_unipc", wan_path / "utils" / "fm_solvers_unipc.py"
)
fm_solvers_unipc = importlib.util.module_from_spec(spec_unipc)
spec_unipc.loader.exec_module(fm_solvers_unipc)

FlowDPMSolverMultistepScheduler = fm_solvers.FlowDPMSolverMultistepScheduler
get_sampling_sigmas = fm_solvers.get_sampling_sigmas
retrieve_timesteps = fm_solvers.retrieve_timesteps
FlowUniPCMultistepScheduler = fm_solvers_unipc.FlowUniPCMultistepScheduler


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dpm_solver_dtype(dtype):
    sampling_steps = 5
    scheduler_fp32 = FlowDPMSolverMultistepScheduler(num_train_timesteps=10)
    sigmas = get_sampling_sigmas(sampling_steps, shift=1.0)
    retrieve_timesteps(scheduler_fp32, device="cpu", sigmas=sigmas, dtype=torch.float32)
    sigmas_fp32 = scheduler_fp32.sigmas.to(torch.float32)
    timesteps_fp32 = scheduler_fp32.timesteps.to(torch.float32)

    scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=10)
    retrieve_timesteps(scheduler, device="cpu", sigmas=sigmas, dtype=dtype)

    assert scheduler.sigmas.dtype == dtype
    assert scheduler.timesteps.dtype == dtype
    assert torch.allclose(scheduler.sigmas.to(torch.float32), sigmas_fp32, atol=2e-3)
    assert torch.allclose(
        scheduler.timesteps.to(torch.float32), timesteps_fp32, atol=2e-2
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unipc_solver_dtype(dtype):
    sampling_steps = 5
    scheduler_fp32 = FlowUniPCMultistepScheduler(
        num_train_timesteps=10, shift=1.0, use_dynamic_shifting=False
    )
    scheduler_fp32.set_timesteps(sampling_steps, device="cpu", shift=1.0, dtype=torch.float32)
    sigmas_fp32 = scheduler_fp32.sigmas.to(torch.float32)
    timesteps_fp32 = scheduler_fp32.timesteps.to(torch.float32)

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=10, shift=1.0, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(sampling_steps, device="cpu", shift=1.0, dtype=dtype)

    assert scheduler.sigmas.dtype == dtype
    assert scheduler.timesteps.dtype == dtype
    assert torch.allclose(scheduler.sigmas.to(torch.float32), sigmas_fp32, atol=2e-3)
    assert torch.allclose(
        scheduler.timesteps.to(torch.float32), timesteps_fp32, atol=2e-2
    )
