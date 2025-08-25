import pytest
import torch

from wan.modules.attention import attention


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_attention_matches_scaled_dot_product(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    torch.manual_seed(0)
    b, l, h, d = 2, 4, 2, 8

    q_cpu = torch.randn(b, l, h, d, dtype=torch.float32, device="cpu")
    k_cpu = torch.randn(b, l, h, d, dtype=torch.float32, device="cpu")
    v_cpu = torch.randn(b, l, h, d, dtype=torch.float32, device="cpu")

    baseline = torch.nn.functional.scaled_dot_product_attention(
        q_cpu.transpose(1, 2),
        k_cpu.transpose(1, 2),
        v_cpu.transpose(1, 2),
        is_causal=False,
        dropout_p=0.0,
    ).transpose(1, 2)

    dtype = torch.float32 if device == "cpu" else torch.float16
    q = q_cpu.to(device, dtype=dtype)
    k = k_cpu.to(device, dtype=dtype)
    v = v_cpu.to(device, dtype=dtype)

    out = attention(q, k, v, dtype=dtype).to("cpu")

    assert out.shape == baseline.shape
    assert torch.allclose(out.float(), baseline.float(), atol=1e-3, rtol=1e-3)
