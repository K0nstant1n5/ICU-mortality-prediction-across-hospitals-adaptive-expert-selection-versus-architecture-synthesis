import torch
from torch import nn

from ..bus import TimeEncoding, TokenBus, causal_mask
from .base import Block, register


@register("attn")
class AttentionBlock(Block):
    """Pre-LN Transformer encoder layer (attention + FFN), always causal.

    Adapter in: add time encoding (attention is permutation-invariant) and a token-type embedding,
    pass the padding mask as key_padding_mask and a causal mask over positions.
    """

    can_fuse_modalities = True

    def __init__(self, d: int, dropout: float = 0.1, n_heads: int = 4, n_types: int = 4, **kwargs):
        super().__init__(d, dropout)
        self.time_enc = TimeEncoding(d)
        self.type_emb = nn.Embedding(n_types, d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d, d))

    def mix(self, bus: TokenBus) -> torch.Tensor:
        h = bus.tokens + self.time_enc(bus.time) + self.type_emb(bus.type)
        L = h.shape[1]
        # A fully padded row would make softmax NaN; keep position 0 attendable for such rows.
        kpm = ~bus.mask
        kpm[:, 0] = False
        a, _ = self.attn(h, h, h, key_padding_mask=kpm, attn_mask=causal_mask(L, h.device), need_weights=False)
        return a + self.ffn(h + a)
