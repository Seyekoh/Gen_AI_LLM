from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from pathlib import Path
from model.config import GPTConfig
from model.gpt import load_checkpoint, save_checkpoint
import torch.nn.functional as F

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer/tokenizer.json"
DATA_PATH = PROJECT_ROOT / "instruction_data/instructions.txt"
CKPT_DIR = PROJECT_ROOT / "training/checkpoints"
FINETUNE_BEST = CKPT_DIR / "finetune_best.pt"

# ---------------- Dataset ----------------
class SFTDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.data = []
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.block_size = block_size

        for t in texts:
            ids = tokenizer.encode(t).ids[: block_size + 1]

            if len(ids) < 2:
                continue

            while len(ids) < block_size + 1:
                ids.append(self.pad_id)

            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)

            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------- Evaluation ----------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp, pad_id):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            logits, _ = model(x, y)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_id
            )
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))

# ---------------- Main ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    pad_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    texts = DATA_PATH.read_text(encoding="utf-8").splitlines()
    print(f"Total samples: {len(texts)}")

    cfg = GPTConfig(vocab_size=vocab_size)
    dataset = SFTDataset(texts, tokenizer, cfg.block_size)

    if len(dataset) < 10:
        raise ValueError(f"Dataset too small: {len(dataset)}")

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Load pretrained model
    model = load_checkpoint(CKPT_DIR / "pretrain_best.pt", device=device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    step = 0
    max_steps = 3000

    model.train()
    while step < max_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device, enabled=use_amp):
                logits, _ = model(x, y)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=pad_id
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            if step % 20 == 0:
                print(f"step {step} | loss {loss.item():.4f}")

            if step % 100 == 0 and step > 0:
                val_loss = evaluate(model, val_loader, device, use_amp, pad_id)
                print(f"  >> val_loss {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, FINETUNE_BEST, step=step, val_loss=val_loss)
                    print("  >> saved finetune_best.pt")

            step += 1
            if step >= max_steps:
                break

    print("\nFine-tuning complete.")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Saved at: {FINETUNE_BEST}")


if __name__ == "__main__":
    main()
