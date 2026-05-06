from __future__ import annotations
import math
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from tokenizers import Tokenizer

from model.gpt import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "training/checkpoints/finetune_best.pt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer/tokenizer.json"
DATA_PATH = PROJECT_ROOT / "instruction_data/instructions.txt"

BATCH_SIZE = 16
EVAL_SAMPLES = 500  # number of samples to randomly select for evaluation


class EvalDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        pad_id = tokenizer.token_to_id("<pad>")

        for t in texts:
            ids = tokenizer.encode(t).ids
            ids = ids[:block_size + 1]
            if len(ids) < 2:
                continue
            while len(ids) < block_size + 1:
                ids.append(pad_id)
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    model = load_checkpoint(MODEL_PATH, device=device)
    model.eval()

    raw_text = DATA_PATH.read_text(encoding="utf-8")
    texts = [block.strip() for block in raw_text.split("<eos>") if block.strip()]
    texts = [block + "\n<eos>" for block in texts]
    dataset = EvalDataset(texts, tokenizer, block_size=128)

    # Random subset
    if len(dataset) > EVAL_SAMPLES:
        indices = random.sample(range(len(dataset)), EVAL_SAMPLES)
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    total_loss = 0.0
    total_tokens = 0

    print(f"\nEvaluating on {len(dataset)} samples, batch size {BATCH_SIZE}...\n")

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            print(f"Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Tokens in batch: {batch_tokens}")

    perplexity = math.exp(total_loss / total_tokens)
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Average loss per token: {total_loss / total_tokens:.4f}")
    print(f"Perplexity over {len(dataset)} samples: {perplexity:.4f}")


if __name__ == "__main__":
    main()
