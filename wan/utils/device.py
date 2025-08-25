import torch


def device_synchronize(device):
    """Synchronize operations on the given device if supported."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device):
    """Empty cache for the given device if supported."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def set_device(device):
    """Set the current device."""
    if isinstance(device, torch.device):
        if device.type == "cuda":
            torch.cuda.set_device(device)
        elif device.type == "mps":
            torch.mps.set_device(device)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.set_device(device)
