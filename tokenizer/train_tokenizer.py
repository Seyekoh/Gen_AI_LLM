"""
Train a small Byte-Pair Encoding (BPE) tokenizer on the corpus.

Reads data/processed/corpus.txt
Writes tokenizer/tokenizer.json

Run:  python -m tokenizer.train_tokenizer
"""
from __future__ import annotations
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "corpus.txt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"

# Small vocab for a small corpus. Lower vocab = better coverage given small data.
VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def main() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_PATH}. "
            "Run `python -m data.prepare_data` first."
        )

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training BPE tokenizer on {CORPUS_PATH} ...")
    tokenizer.train(files=[str(CORPUS_PATH)], trainer=trainer)

    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Saved tokenizer to {TOKENIZER_PATH}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Quick demo
    sample = "In 2025, a Senior ML Engineer in USA earned $150,000."
    enc = tokenizer.encode(sample)
    print()
    print(f"Sample: {sample!r}")
    print(f"Tokens ({len(enc.ids)}): {enc.tokens}")
    print(f"IDs:    {enc.ids}")
    print(f"Decoded: {tokenizer.decode(enc.ids)!r}")


if __name__ == "__main__":
    main()
