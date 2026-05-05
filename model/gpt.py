"""
A from-scratch decoder-only Transformer (GPT-style) implemented in PyTorch.

Components implemented manually (no nn.Transformer used):
    - Token + positional embeddings
    - Multi-head causal self-attention
    - Position-wise feed-forward network
    - Pre-LayerNorm transformer block
    - Stacked transformer + output projection
    - Autoregressive text generation (greedy / top-k)

This follows the GPT-2 architectural conventions: pre-norm, learned positional
embeddings, GELU activations, weight tying between input embedding and output head.
"""
from __future__ import annotations
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Computes Q, K, V projections in one matmul, splits into heads, applies
    a causal mask so position i can only attend to positions <= i, then
    re-combines heads.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # Combined Q, K, V projection (3x n_embd output)
        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Output projection
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular causal mask (registered as buffer so it moves with .to(device))
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time, channels (== n_embd)

        # Project to Q, K, V — shape (B, T, 3*C) → split → 3 × (B, T, C)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape into heads: (B, T, n_head, head_dim) → transpose to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, nh, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask: positions can only attend to themselves and earlier
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values: (B, nh, T, head_dim)
        y = attn @ v

        # Re-combine heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    """Position-wise feed-forward network: two linear layers with GELU in between.

    Standard GPT-2 uses 4x expansion ratio.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """One transformer block: pre-norm + attention + residual, pre-norm + FFN + residual."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # pre-norm + residual
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer for autoregressive language modeling."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: input embedding and output projection share weights.
        # Standard trick — saves params and tends to improve generalization.
        self.head.weight = self.token_emb.weight

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)
        # Special init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx:     (B, T) token IDs
        targets: (B, T) token IDs to predict (shifted by one). Optional.
        Returns: (logits, loss) where loss is None if targets is None.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence length {T} exceeds block size {self.cfg.block_size}"
        )

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)  # (B, T, C)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens from the model."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_tok], dim=1)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return idx


def save_checkpoint(model: GPT, path: str | Path, **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(model.cfg),
            **extra,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> GPT:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


if __name__ == "__main__":
    # Smoke test: build a tiny model, forward + backward pass, count params.
    cfg = GPTConfig()
    model = GPT(cfg)
    print(f"Model parameters: {model.num_params():,} (~{model.num_params()/1e6:.2f}M)")

    B, T = 2, 16
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(x, y)
    print(f"Forward OK. logits: {tuple(logits.shape)}, loss: {loss.item():.4f}")
    loss.backward()
    print("Backward OK.")
