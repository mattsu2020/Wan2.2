import torch


def ensure_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure ``tensor`` is ``float32`` on MPS devices.

    MPS backend does not support ``float64`` tensors. This helper converts
    ``float64`` tensors to ``float32`` when the tensor resides on an MPS
    device. Tensors on other backends or with different dtypes are returned
    unchanged.

    Parameters
    ----------
    tensor: torch.Tensor
        The input tensor to check.

    Returns
    -------
    torch.Tensor
        The converted tensor if necessary, otherwise the original tensor.
    """
    if isinstance(tensor, torch.Tensor) and tensor.device.type == "mps" and tensor.dtype == torch.float64:
        return tensor.float()
    return tensor
