"""Every block family must be causal: outputs at hour t may not depend on inputs after t."""
import pytest
import torch

from icu_aes_nas.core.blocks.base import BLOCK_REGISTRY, build_block
from icu_aes_nas.core.network import BlockNetwork, NetworkSpec

FAMILIES = sorted(BLOCK_REGISTRY)


@pytest.mark.parametrize("family", FAMILIES)
def test_block_is_causal(family):
    torch.manual_seed(0)
    d, T = 16, 10
    block = build_block(family, d, dropout=0.0).eval()
    from icu_aes_nas.core.bus import TokenBus
    x = torch.randn(2, T, d)
    def mk(t):
        return TokenBus(tokens=t, mask=torch.ones(2, T, dtype=torch.bool),
                        time=torch.arange(T).float().expand(2, T), type=torch.ones(2, T, dtype=torch.long))
    y1 = block(mk(x)).tokens
    x2 = x.clone()
    x2[:, 6:] += torch.randn_like(x2[:, 6:]) * 5
    y2 = block(mk(x2)).tokens
    assert torch.allclose(y1[:, :6], y2[:, :6], atol=1e-5), family


@pytest.mark.parametrize("stem", ["linear", "grud"])
@pytest.mark.parametrize("task_type", ["static", "dynamic"])
def test_network_forward_shapes(stem, task_type):
    spec = NetworkSpec(d_model=16, stem=stem, blocks=[{"family": "gru"}, {"family": "attn"}], head="mean",
                       task_type=task_type)
    net = BlockNetwork(spec)
    B, T, V = 3, 7, spec.n_vars
    batch = {"x": torch.randn(B, T, V), "m": torch.ones(B, T, V), "delta": torch.zeros(B, T, V),
             "static": torch.randn(B, spec.static_dim), "valid": torch.ones(B, T),
             "time": torch.arange(T).float().expand(B, T)}
    out = net(batch)
    assert out.shape == ((B,) if task_type == "static" else (B, T))
