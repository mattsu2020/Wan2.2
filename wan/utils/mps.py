import torch


def ensure_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Promote ``tensor`` to ``float32`` on MPS when required.

    Apple's MPS backend has limited support for some operations in half
    precision and does not support ``float64``. This helper casts tensors to
    ``float32`` only when they reside on an MPS device and are not already in
    that dtype. Tensors on other backends or already using ``float32`` are
    returned unchanged.

    Parameters
    ----------
    tensor: torch.Tensor
        The input tensor to check.

    Returns
    -------
    torch.Tensor
        The converted tensor if necessary, otherwise the original tensor.
    """
    if isinstance(tensor, torch.Tensor) and tensor.device.type == "mps" and tensor.dtype != torch.float32:
        return tensor.float()
    return tensor
