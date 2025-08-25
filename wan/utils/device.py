import torch


def get_default_device(device_id: int = 0) -> torch.device:
    """Return the default torch.device for the current environment.

    The function prefers CUDA when available, then Apple's MPS backend,
    and finally CPU.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache() -> None:
    """Free cached memory of the current accelerator if needed."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        # torch.mps may not expose empty_cache on older versions
        empty = getattr(torch.mps, "empty_cache", None)
        if callable(empty):
            empty()
