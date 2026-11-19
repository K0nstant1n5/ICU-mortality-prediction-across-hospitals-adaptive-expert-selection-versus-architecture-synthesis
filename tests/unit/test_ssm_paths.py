"""The Mamba block's CUDA-kernel path and pure-PyTorch path must compute the same function from the same weights."""

import pytest
import torch

from icu_aes_nas.core.blocks.ssm import KERNEL_AVAILABLE, SelectiveSSM


def test_cpu_path_runs_even_if_kernel_installed():
    torch.manual_seed(0)
    ssm = SelectiveSSM(16, d_state=8)
    y = ssm(torch.randn(2, 12, 16))           # CPU tensor -> must not touch the CUDA kernel
    assert y.shape == (2, 12, 16) and torch.isfinite(y).all()


@pytest.mark.skipif(not (KERNEL_AVAILABLE and torch.cuda.is_available()), reason="needs mamba_ssm + a GPU")
def test_kernel_matches_reference_scan():
    torch.manual_seed(0)
    ssm = SelectiveSSM(32, d_state=16).cuda()
    x = torch.randn(3, 24, 32, device="cuda")
    with torch.no_grad():
        fast = ssm(x)
        ssm.use_cuda_kernel = False
        ref = ssm(x)
    torch.testing.assert_close(fast, ref, rtol=1e-3, atol=1e-4)
