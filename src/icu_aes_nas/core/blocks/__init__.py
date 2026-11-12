"""Block families. Each block maps TokenBus -> TokenBus and declares its native format (for the typed graph)."""

from .base import BLOCK_REGISTRY, Block, build_block  # noqa: F401
from . import attention, conv, ffn, recurrent, ssm  # noqa: F401  (registers families)
