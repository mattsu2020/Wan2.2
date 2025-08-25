import torch


def quantize_model(model: torch.nn.Module, quant_type: str) -> torch.nn.Module:
    """Apply bitsandbytes quantization to all linear layers in ``model``.

    Args:
        model: The model whose ``nn.Linear`` layers will be replaced.
        quant_type: Either ``"int8"`` or ``"nf4"``.

    Returns:
        The quantized model (modifications happen in-place).
    """
    import bitsandbytes as bnb

    if quant_type not in {"int8", "nf4"}:
        raise ValueError(f"Unsupported quantization type: {quant_type}")

    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            if quant_type == "int8":
                qmodule = bnb.nn.Linear8bitLt.from_float(module)
            else:
                qmodule = bnb.nn.Linear4bit.from_float(module, quant_type="nf4")
            setattr(model, name, qmodule)
        else:
            quantize_model(module, quant_type)
    return model
