import torch


def get_best_device() -> torch.device:
    """Return the best available device in priority order CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device() -> None:
    """Synchronize the current device if it supports synchronization."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def empty_device_cache() -> None:
    """Empty the cache of the current device if supported."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
