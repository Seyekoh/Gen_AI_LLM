# LLM From Scratch — AI Job Market

A from-scratch implementation of the full LLM development pipeline, built for
**CS4985 GenAI: Theories & Practices** (Spring 2026).

We implement every stage of an end-to-end LLM pipeline — data preparation,
tokenization, model architecture, pre-training, fine-tuning, evaluation, and
deployment — using only PyTorch and Hugging Face's `tokenizers` library.

The model is trained on the
[AI Job Market Trends (2022–2026)](https://www.kaggle.com/datasets/hassanali789/ai-job-market-trends-20222026)
dataset, converted to a templated text corpus.

---

## Honest scope

This is a research/educational project, not a competitive language model.
The corpus is small (~6,000 sentences derived from a 2,000-row CSV), and the
model is small (~3M parameters). The goal is to demonstrate that we built and
understood every piece of the pipeline, not to produce GPT-quality text. The
model overfits its templated corpus and generates plausible-looking but
shallow text about AI job market trends.

---

## Pipeline

```
Kaggle CSV
    │
    ▼
[1] data/prepare_data.py     →  data/processed/corpus.txt
    │
    ▼
[2] tokenizer/train_tokenizer.py  →  tokenizer/tokenizer.json
    │
    ▼
[3] model/gpt.py               (architecture)
    │
    ▼
[4] training/pretrain.py     →  training/checkpoints/pretrain_final.pt
    │
    ▼
[5] training/finetune.py     →  training/checkpoints/finetune_final.pt
    │
    ▼
[6] eval/evaluate.py           
    │
    ▼
[7] app/gradio_app.py        →  live demo
```

---

## Repo structure

```
.
├── data/
│   ├── raw/                       # Place ai_job_market_dataset.csv here
│   ├── processed/                 # Generated corpus.txt
│   └── prepare_data.py            # CSV → text corpus
├── tokenizer/
│   ├── train_tokenizer.py         # Trains BPE tokenizer
│   └── tokenizer.json             # (generated)
├── model/
│   ├── config.py                  # GPTConfig dataclass
│   └── gpt.py                     # From-scratch GPT implementation
├── training/
│   ├── pretrain.py                # Pre-training loop
│   ├── finetune.py                # SFT fine-tuning
│   └── checkpoints/               # (generated)
├── instruction_data/
│   └── build_instructions.py      # Generate Q&A pairs
├── analysis/
│   └── trends.py                  # Pandas trend analysis
├── eval/
│   └── evaluate.py                # Perplexity + samples
├── app/
│   └── gradio_app.py              # Web demo
├── docs/
│   ├── HANDOFF_TEAMMATE_A.md
│   └── HANDOFF_TEAMMATE_B.md
├── requirements.txt
└── README.md
```

---

## Setup

Requires Python 3.10–3.12. We recommend a virtual environment.

```bash
git clone <repo-url>
cd <repo>

python -m venv .venv
source .venv/bin/activate           # Linux/Mac
# .venv\Scripts\activate            # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training (recommended), install PyTorch with CUDA support instead:

```bash
# CUDA 12.8 (e.g. RTX 40-series, RTX 50-series Blackwell)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Other CUDA versions: see https://pytorch.org/get-started/locally/
```

Place the dataset at `data/raw/ai_job_market_dataset.csv`.

---

## Running the pipeline

End-to-end (run in this order):

```bash
# 1. Build the text corpus from the CSV
python -m data.prepare_data

# 2. Train the BPE tokenizer
python -m tokenizer.train_tokenizer

# 3. Pre-train the GPT model
python -m training.pretrain --max_steps 2000 --batch_size 32

# 4. Fine-tune on instruction data
python -m instruction_data.build_instructions
python -m training.finetune

# 5. Evaluate                        
python -m eval.evaluate

# 6. Launch the demo               
python -m app.gradio_app
```

---

## Pipeline stages explained

### 1. Data preparation

The CSV has 8 columns: `Year, Job_Title, Country, Company_Type,
Experience_Level, Salary_USD, Remote, Top_Skill`. Each row is rendered into
multiple natural-language sentences via templates (see
`data/prepare_data.py`). We also generate aggregate sentences (per-year,
per-role averages) so the model sees summary patterns. Final corpus:
~6,000 sentences, ~840 KB.

### 2. Tokenization

We train a Byte-Pair Encoding (BPE) tokenizer with vocab size 4,096 using
Hugging Face's `tokenizers` library. Special tokens: `<pad>`, `<unk>`,
`<bos>`, `<eos>`.

### 3. Architecture

A decoder-only transformer (GPT-2 style), implemented from scratch in
PyTorch in `model/gpt.py`:

- Token + learned positional embeddings
- Multi-head causal self-attention (manually implemented; no `nn.Transformer`)
- Position-wise feed-forward (4× expansion, GELU)
- Pre-LayerNorm transformer blocks
- Weight tying between input embeddings and output head
- ~3M parameters (4 layers, 4 heads, embedding dim 192, context length 128)

### 4. Pre-training

Standard next-token prediction with cross-entropy loss. Features:

- AdamW optimizer with decoupled weight decay
- Linear warmup + cosine learning rate decay
- Gradient clipping
- bfloat16 mixed precision (when CUDA is available)
- Periodic validation perplexity reporting
- Checkpointing every N steps

### 5. Fine-tuning *(Teammate A)*

Supervised fine-tuning on (instruction, response) pairs derived from the
trend analysis. See `docs/HANDOFF_TEAMMATE_A.md`.

### 6. Evaluation *(Teammate A)*

Perplexity on a held-out validation set, plus qualitative sample generations
across a fixed prompt suite. See `docs/HANDOFF_TEAMMATE_A.md`.

### 7. Deployment *(Teammate B)*

Gradio web app that lets users query the model with grounding from the
trend analysis. See `docs/HANDOFF_TEAMMATE_B.md`.

---

## Team

- **James Bridges** — Data prep, tokenization, model architecture, pre-training
- **Teammate A** — Instruction dataset, fine-tuning, evaluation
- **Teammate B** — Trend analysis, RAG/grounding, Gradio app, documentation

---

## Acknowledgments

Architecture follows the conventions established by Radford et al.'s GPT-2
(2019) and the implementation patterns in Sebastian Raschka's
[*LLMs from Scratch*](https://github.com/rasbt/LLMs-from-scratch) and
Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). All code
in this repository was written by the team.
