"""
Pre-train the GPT model on the templated job-market corpus.

Loads tokenizer + corpus, tokenizes everything, then trains the model to
predict the next token. Supports CUDA + mixed precision automatically.

Run:
    python -m training.pretrain
    python -m training.pretrain --max_steps 2000 --batch_size 64
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from model.config import GPTConfig
from model.gpt import GPT, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.txt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
CKPT_DIR = PROJECT_ROOT / "training" / "checkpoints"


class TokenDataset(Dataset):
    """Wraps a 1-D tensor of token IDs into fixed-length training examples.

    Each example yields (x, y) where y is x shifted right by one token —
    i.e. the standard next-token-prediction objective.
    """

    def __init__(self, token_ids: torch.Tensor, block_size: int):
        self.tokens = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of non-overlapping training windows
        return max(0, (len(self.tokens) - 1) // self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def load_and_tokenize(tokenizer: Tokenizer, corpus_path: Path) -> torch.Tensor:
    text = corpus_path.read_text(encoding="utf-8")
    # Encode the entire corpus as one long stream, separating lines with EOS
    # so the model learns sentence boundaries.
    eos_id = tokenizer.token_to_id("<eos>")
    all_ids: list[int] = []
    for line in text.splitlines():
        ids = tokenizer.encode(line).ids
        all_ids.extend(ids)
        all_ids.append(eos_id)
    return torch.tensor(all_ids, dtype=torch.long)


def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"
    print(f"Device: {device} ({device_name})")

    # Load tokenizer
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run `python -m tokenizer.train_tokenizer` first."
        )
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    # Build config (vocab from tokenizer, block size from CLI)
    cfg = GPTConfig(vocab_size=vocab_size, block_size=args.block_size)

    # Tokenize corpus
    print("Tokenizing corpus ...")
    tokens = load_and_tokenize(tokenizer, CORPUS_PATH)
    print(f"Total tokens: {len(tokens):,}")

    # Train/val split (90/10)
    split_idx = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")

    train_ds = TokenDataset(train_tokens, cfg.block_size)
    val_ds = TokenDataset(val_tokens, cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"Train batches/epoch: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = GPT(cfg).to(device)
    print(f"Model parameters: {model.num_params():,} (~{model.num_params()/1e6:.2f}M)")

    # Optimizer with weight decay only on 2D params (matrices), not biases/LayerNorm
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr,
        betas=(0.9, 0.95),
    )

    # Mixed precision (only useful on CUDA)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        losses = []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device, enabled=use_amp, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / max(1, len(losses))

    # Training loop
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    print(f"\nStarting training for {args.max_steps} steps ...")
    while step < args.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=use_amp, dtype=torch.bfloat16):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate()
            print(f"  >> step {step}: val_loss {val_loss:.4f} (perplexity {math.exp(val_loss):.2f})")

        if step > 0 and step % args.save_interval == 0:
            ckpt = CKPT_DIR / f"pretrain_step{step}.pt"
            save_checkpoint(model, ckpt, step=step, val_loss=evaluate())
            print(f"  >> saved checkpoint to {ckpt}")

        step += 1

    # Final eval + checkpoint
    val_loss = evaluate()
    print(f"\nFinal val loss: {val_loss:.4f} (perplexity {math.exp(val_loss):.2f})")
    final_ckpt = CKPT_DIR / "pretrain_final.pt"
    save_checkpoint(model, final_ckpt, step=step, val_loss=val_loss)
    print(f"Saved final checkpoint to {final_ckpt}")

    # Quick generation sample
    print("\n--- Sample generation ---")
    prompt = "In 2025, a Senior"
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=60, temperature=0.8, top_k=40,
                         eos_token_id=cfg.eos_token_id)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
