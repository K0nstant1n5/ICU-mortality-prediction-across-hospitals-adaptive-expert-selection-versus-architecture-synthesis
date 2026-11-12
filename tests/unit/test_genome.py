import numpy as np
from omegaconf import OmegaConf

from icu_aes_nas.nas.genome import is_legal
from icu_aes_nas.nas.space import SearchSpace

NAS = OmegaConf.create({"max_slots": 4, "min_slots": 2, "d_model_choices": [16, 32],
                        "families": ["gru", "tcn", "attn", "mamba", "ffn", "identity"],
                        "stems": ["linear", "grud"], "heads": ["last", "mean", "attnpool"], "inherit": False,
                        "inherit_from": []})


def test_sampled_and_mutated_genomes_are_legal():
    space = SearchSpace(NAS, {}, "static", None, np.random.default_rng(0))
    for _ in range(50):
        g = space.sample()
        assert is_legal(g, {}, "static")[0]
        child, op = space.mutate(g)
        assert is_legal(child, {}, "static")[0], op


def test_genome_key_is_stable_and_spec_roundtrips():
    space = SearchSpace(NAS, {}, "static", None, np.random.default_rng(1))
    g = space.sample()
    assert g.key() == g.key()
    spec = g.to_spec(n_vars=48, static_dim=8, use_mask=True, task_type="static")
    assert spec.d_model in (16, 32)
