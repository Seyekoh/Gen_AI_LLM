"""
GPT model configuration.

Sized small intentionally: ~3M parameters. The corpus is small (<1 MB),
so a tiny model will train quickly and avoid catastrophic overfitting.
Adjust here if you have more time/compute budget.
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Tokenizer
    vocab_size: int = 4096       # must match tokenizer vocab_size

    # Architecture
    block_size: int = 128        # max sequence length / context window
    n_layer: int = 4             # number of transformer blocks
    n_head: int = 4              # number of attention heads
    n_embd: int = 192            # embedding dimension (must be divisible by n_head)
    dropout: float = 0.1
    bias: bool = True            # use bias in Linear / LayerNorm

    # Special token IDs (must match the tokenizer's special tokens)
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3


# Default config used by training scripts. Override fields as needed.
DEFAULT_CONFIG = GPTConfig()
