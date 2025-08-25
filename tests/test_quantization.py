import importlib.util
import pathlib

import pytest
import torch

quant_module_path = pathlib.Path(__file__).resolve().parent.parent / "wan" / "utils" / "quantization.py"
spec = importlib.util.spec_from_file_location("quantization", quant_module_path)
quantization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantization)
quantize_model = quantization.quantize_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("qtype", ["int8", "nf4"])
def test_quantized_linear_close(qtype):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
    ).cuda()
    inp = torch.randn(2, 32, device="cuda", dtype=torch.float16)
    ref = model(inp)
    qmodel = quantize_model(model, qtype).cuda()
    out = qmodel(inp)
    diff = torch.mean(torch.abs(ref - out)) / torch.mean(torch.abs(ref))
    assert diff < 0.1
