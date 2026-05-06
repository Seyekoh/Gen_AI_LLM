# LLM From Scratch — AI Job Market

A from-scratch implementation of a small end-to-end LLM development pipeline for **CS4985 GenAI: Theories & Practices (Spring 2026)**.

This project implements each major stage of an LLM workflow: data preparation, tokenization, transformer architecture, pre-training, supervised fine-tuning, evaluation, trend analysis, grounding, and deployment through a Gradio app. The model is trained on the **AI Job Market Trends (2022–2026)** dataset, which is converted from tabular CSV records into a templated natural-language corpus.

## Honest Scope

This is a **research and educational project**, not a competitive language model. The dataset is small, with about **2,000 CSV rows** converted into roughly **6,000 templated sentences**, and the model is intentionally small at about **2.1M parameters**. The goal is to demonstrate that the team built and understood the full LLM pipeline from scratch, not to produce GPT-quality text.

The from-scratch model can generate plausible AI job-market-style sentences, especially when prompted with patterns similar to its training corpus. However, because the corpus and model are small, raw model generations may be shallow, repetitive, or unstable. For this reason, the deployed Gradio app uses deterministic trend analysis as the factual grounding layer. The LLM remains part of the pipeline, but the final demo answers are grounded in Pandas/sklearn analysis rather than relying only on sampled model text.

## Project Summary

The project has two connected goals:

1. **Build an LLM pipeline from scratch** using PyTorch and Hugging Face's `tokenizers` library.
2. **Deploy a useful AI job-market demo** that answers salary, skill, forecast, trend, comparison, and remote-work questions using grounded dataset analysis.

The final app is best understood as a grounded AI job-market assistant:

```text
User question
    ↓
Intent detection
    ↓
Pandas/sklearn trend analysis
    ↓
Grounded final answer + supporting table
    ↓
Optional from-scratch LLM draft
```

## Dataset

The project uses the **AI Job Market Trends (2022–2026)** dataset.

Expected local path:

```text
data/raw/ai_job_market_dataset.csv
```

The CSV contains the following columns:

| Column | Description |
|---|---|
| `Year` | Dataset year, from 2022 through 2026 |
| `Job_Title` | AI-related job title |
| `Country` | Country associated with the job record |
| `Company_Type` | Type of company/employer |
| `Experience_Level` | Experience level for the role |
| `Salary_USD` | Salary in USD |
| `Remote` | Whether remote work is available |
| `Top_Skill` | Main listed skill for the record |

## Pipeline

```text
Kaggle CSV
    │
    ▼
[1] data/prepare_data.py
    → data/processed/corpus.txt
    │
    ▼
[2] tokenizer/train_tokenizer.py
    → tokenizer/tokenizer.json
    │
    ▼
[3] model/gpt.py
    → from-scratch decoder-only transformer
    │
    ▼
[4] training/pretrain.py
    → training/checkpoints/pretrain_best.pt
    │
    ▼
[5] instruction_data/build_instructions.py
    → instruction_data/instructions.txt
    │
    ▼
[6] training/finetune.py
    → training/checkpoints/finetune_best.pt
    │
    ▼
[7] eval/evaluate.py
    → perplexity / qualitative evaluation
    │
    ▼
[8] analysis/trends.py
    → deterministic trend analysis + grounding facts
    │
    ▼
[9] app/gradio_app.py
    → grounded web demo
```

## Repository Structure

```text
.
├── app/
│   └── gradio_app.py              # Grounded Gradio demo
├── analysis/
│   └── trends.py                  # Pandas/sklearn trend analysis and grounding utilities
├── data/
│   ├── raw/                       # Place ai_job_market_dataset.csv here
│   ├── processed/                 # Generated corpus.txt
│   └── prepare_data.py            # CSV → templated text corpus
├── docs/
│   ├── HANDOFF_TEAMMATE_A.md
│   └── HANDOFF_TEAMMATE_B.md
├── eval/
│   └── evaluate.py                # Perplexity evaluation
├── instruction_data/
│   ├── build_instructions.py      # Builds SFT instruction/response data
│   └── instructions.txt           # Generated instruction data
├── model/
│   ├── config.py                  # GPTConfig dataclass
│   └── gpt.py                     # From-scratch GPT implementation
├── tokenizer/
│   ├── train_tokenizer.py         # Trains BPE tokenizer
│   └── tokenizer.json             # Generated tokenizer
├── training/
│   ├── pretrain.py                # Pre-training loop
│   ├── finetune.py                # SFT fine-tuning loop
│   └── checkpoints/               # Generated model checkpoints
├── requirements.txt
└── README.md
```

## Setup

Requires **Python 3.10+**. The project has been tested with newer Python versions as well, but a virtual environment is recommended.

```bash
git clone <repo-url>
cd <repo>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/Mac:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch build. Example for CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

For other CUDA versions, use the official PyTorch installation selector.

## Dataset Placement

Before running the pipeline, place the dataset here:

```text
data/raw/ai_job_market_dataset.csv
```

If the file is missing, `data.prepare_data` and `analysis.trends` will raise a `FileNotFoundError`.

## Running the Full Pipeline

Run these commands from the project root.

### 1. Prepare the text corpus

```bash
python -m data.prepare_data
```

This reads:

```text
data/raw/ai_job_market_dataset.csv
```

and writes:

```text
data/processed/corpus.txt
```

Each CSV row is rendered into multiple natural-language templates, and additional aggregate salary summaries are generated by year and role.

### 2. Train the tokenizer

```bash
python -m tokenizer.train_tokenizer
```

This trains a Byte-Pair Encoding tokenizer and writes:

```text
tokenizer/tokenizer.json
```

### 3. Pre-train the GPT model

Recommended command:

```bash
python -m training.pretrain --max_steps 3000 --batch_size 64 --eval_interval 250 --save_interval 500
```

This trains the model with next-token prediction and saves checkpoints under:

```text
training/checkpoints/
```

Important outputs include:

```text
training/checkpoints/pretrain_best.pt
training/checkpoints/pretrain_final.pt
training/checkpoints/pretrain_step<step>.pt
```

The best checkpoint is selected using validation loss/perplexity and should be used for fine-tuning.

### 4. Build instruction data

```bash
python -m instruction_data.build_instructions
```

This writes:

```text
instruction_data/instructions.txt
```

The instruction data uses an instruction/response format similar to:

```text
<bos>### Instruction:
Question text here

### Response:
Answer text here
<eos>
```

The app also uses grounded prompt patterns with `Facts:` and `Question:` fields, so the instruction data should include examples that match that format.

### 5. Fine-tune the model

```bash
python -m training.finetune
```

Fine-tuning starts from:

```text
training/checkpoints/pretrain_best.pt
```

and saves the best fine-tuned checkpoint as:

```text
training/checkpoints/finetune_best.pt
```

If an old fine-tuned checkpoint already exists and you want to make sure the app loads the new one, delete it before rerunning fine-tuning.

Linux/Mac:

```bash
rm training/checkpoints/finetune_best.pt
```

Windows PowerShell:

```powershell
Remove-Item training/checkpoints/finetune_best.pt
```

Then rerun:

```bash
python -m training.finetune
```

### 6. Evaluate the model

```bash
python -m eval.evaluate
```

This evaluates the fine-tuned model on the instruction data and reports loss/perplexity.

### 7. Launch the Gradio app

```bash
python -m app.gradio_app
```

Alternative:

```bash
python app/gradio_app.py
```

The app loads the first available checkpoint in this order:

```text
training/checkpoints/finetune_final.pt
training/checkpoints/finetune_best.pt
training/checkpoints/pretrain_best.pt
```

If no fine-tuned model is available, the app can still run using `pretrain_best.pt`.

## Quick Rerun Guide

You do not always need to rerun the full pipeline.

### If you only changed `app/gradio_app.py`

Only relaunch the app:

```bash
python -m app.gradio_app
```

### If you changed `analysis/trends.py`

Only relaunch the app:

```bash
python -m app.gradio_app
```

### If you changed `instruction_data/build_instructions.py`

Rerun instruction generation, fine-tuning, evaluation, and the app:

```bash
python -m instruction_data.build_instructions
python -m training.finetune
python -m eval.evaluate
python -m app.gradio_app
```

### If you changed `training/finetune.py`

Rerun fine-tuning, evaluation, and the app:

```bash
python -m training.finetune
python -m eval.evaluate
python -m app.gradio_app
```

### If you changed `data/prepare_data.py`

Rerun the full data/model pipeline:

```bash
python -m data.prepare_data
python -m tokenizer.train_tokenizer
python -m training.pretrain --max_steps 3000 --batch_size 64 --eval_interval 250 --save_interval 500
python -m instruction_data.build_instructions
python -m training.finetune
python -m eval.evaluate
python -m app.gradio_app
```

### If you changed `model/gpt.py` or `model/config.py`

Rerun pre-training and everything after it:

```bash
python -m training.pretrain --max_steps 3000 --batch_size 64 --eval_interval 250 --save_interval 500
python -m instruction_data.build_instructions
python -m training.finetune
python -m eval.evaluate
python -m app.gradio_app
```

If the architecture shape changed, old checkpoints may no longer load.

## Pipeline Stages Explained

### 1. Data Preparation

`data/prepare_data.py` converts the raw CSV into a natural-language corpus. Each row is rendered through multiple templates, and the script also creates aggregate sentences for average salary by year and role.

Example generated sentence style:

```text
In 2026, a Senior-level ML Engineer in USA working at a Freelance company earned $48,724. Their top skill was PyTorch. Remote work: No.
```

The resulting corpus is small, about 6,000 sentences and roughly 840 KB, which is appropriate for a small educational model but not enough for general-purpose language modeling.

### 2. Tokenization

`tokenizer/train_tokenizer.py` trains a Byte-Pair Encoding tokenizer using Hugging Face's `tokenizers` library.

Configuration:

| Setting | Value |
|---|---:|
| Tokenizer type | BPE |
| Vocab cap | 4,096 |
| Actual learned vocab | about 1,521 tokens |
| Special tokens | `<pad>`, `<unk>`, `<bos>`, `<eos>` |

### 3. Model Architecture

`model/gpt.py` implements a decoder-only transformer in PyTorch.

Main components:

- Token embeddings
- Learned positional embeddings
- Multi-head causal self-attention
- Causal attention mask
- Position-wise feed-forward layers
- GELU activation
- Pre-LayerNorm transformer blocks
- Residual connections
- Final language-modeling head
- Weight tying between token embeddings and output projection
- Autoregressive generation with temperature and top-k sampling

Default configuration:

| Setting | Value |
|---|---:|
| Layers | 4 |
| Attention heads | 4 |
| Embedding size | 192 |
| Context length | 128 |
| Dropout | 0.1 |
| Parameters | about 2.1M |

### 4. Pre-training

`training/pretrain.py` trains the model with standard next-token prediction.

Training features:

- AdamW optimizer
- Decoupled weight decay
- Linear warmup
- Cosine learning-rate decay
- Gradient clipping
- Mixed precision when CUDA is available
- Validation loss/perplexity reporting
- Periodic checkpointing

Recorded pre-training results:

| Metric | Value |
|---|---:|
| GPU | NVIDIA RTX 5080 |
| Total steps | 3,000 |
| Batch size | 64 |
| Wall-clock time | ~19 seconds |
| Final train loss | 0.33 |
| Best validation perplexity | 1.96 |
| Best validation step | around 1,250-1,500 |
| Canonical checkpoint | `pretrain_best.pt` |

Validation loss bottomed out around step 1,250-1,500 and rose afterward, showing overfitting on the small templated corpus. The best validation checkpoint is therefore a more honest downstream choice than the final checkpoint.

### 5. Fine-tuning

`training/finetune.py` performs supervised fine-tuning on generated instruction/response examples.

The fine-tuning data is produced by:

```bash
python -m instruction_data.build_instructions
```

Important implementation note: instruction examples are multi-line blocks. They should be loaded as complete examples, not individual lines. The expected training unit is a full block from `<bos>` through `<eos>`.

Fine-tuning writes:

```text
training/checkpoints/finetune_best.pt
```

### 6. Evaluation

`eval/evaluate.py` computes loss and perplexity over instruction examples.

Run:

```bash
python -m eval.evaluate
```

The evaluation is useful for checking whether the model learned the instruction format, but low perplexity does not guarantee high-quality open-ended answers because the evaluation data is still small and template-heavy.

### 7. Trend Analysis and Grounding

`analysis/trends.py` provides deterministic analysis functions used by the app.

Key functions:

| Function | Purpose |
|---|---|
| `top_skills_by_year()` | Finds the most frequent skill for each year |
| `avg_salary_by_role(year=None)` | Computes average salary by job title, optionally filtered by year |
| `salary_trend(role)` | Computes year-over-year average salary for a role |
| `skill_growth()` | Computes skill frequency over time and a simple linear trend slope |
| `forecast_next_year(role, target_year=2027)` | Uses linear regression to estimate a future average salary |
| `get_grounding_facts(query)` | Returns short relevant facts for grounding the app/model |

The forecast function is illustrative, not predictive. It uses a simple linear regression over a small dataset, so it should be presented as a demo trend rather than a real salary forecast.

### 8. Gradio Deployment

`app/gradio_app.py` launches a web interface for interacting with the project.

The updated app is **grounding-first**. It detects the user's intent, runs the appropriate deterministic analysis, and returns a grounded final answer. The from-scratch LLM can still be called, but its draft is shown separately because raw generation may be unstable.

Supported question types include:

- Salary questions
- Skill-growth questions
- 2027 forecast questions
- Role comparisons
- Salary trend questions
- Remote-work questions
- Dataset summaries
- Highest-paying role questions

Example questions:

```text
What is the average salary for an ML Engineer?
Forecast AI Engineer salary in 2027.
Compare ML Engineer and Data Scientist.
Which role has the highest average salary?
How has Data Scientist salary changed over time?
Which skills are growing the fastest?
Are AI jobs remote?
Give me a summary of the dataset.
```

## Gradio App Design

The final deployed app uses this flow:

```text
User question
    ↓
Detect intent
    ↓
Retrieve grounding facts
    ↓
Run deterministic analysis
    ↓
Return grounded final answer and supporting table
    ↓
Optionally call the from-scratch LLM for a draft response
```

The app includes:

- A question textbox
- Generate button
- Grounded final answer box
- Supporting analysis table
- Retrieved grounding facts
- Optional from-scratch LLM draft
- App diagnostics, including checkpoint, device, dataset years, roles, and skills

This design keeps the project honest and useful: the LLM demonstrates the from-scratch pipeline, while the trend-analysis layer provides the reliable factual signal.

## Sample Pre-training Outputs

After pre-training with `pretrain_best.pt`, the model produces plausible templated job-market text when prompted with corpus-like prefixes.

| Prompt | Output |
|---|---|
| `In 2026, an Entry-level` | `In 2026, an Entry-level ML Engineer in USA working at a Freelance company earned $48,724. Their top skill was PyTorch. Remote work: No.` |
| `A Data Scientist based in` | `A Data Scientist based in Germany at a Freelance firm reported a salary of $182,964 in 2026. The role required NLP as the primary skill, at Senior level. Remote: Yes.` |
| `The ML Engineer position in` | `The ML Engineer position in India paid $61,868 in 2023 for Senior-level professionals. Python was the most important skill. The role was on-site only.` |
| `In 2024, the average salary` | `In 2024, the average salary for a AI Engineer was $97,747 across 72 reported positions.` |

Generated values are sampled from the model's learned distribution. They are not retrieved from the CSV and should not be treated as factual salary values.

## Interpreting Model Quality

The model can learn local template structure but does not reliably answer broad natural-language questions by itself. This limitation is expected because:

- The training corpus is small.
- The corpus is heavily templated.
- The model has only about 2.1M parameters.
- The context window is short.
- Instruction fine-tuning data is generated from the same narrow dataset.

For the final demo, the strongest system behavior comes from combining:

```text
Small from-scratch LLM pipeline + deterministic grounding + transparent app design
```

## Troubleshooting

### The app says no checkpoint was found

Make sure at least one of these files exists:

```text
training/checkpoints/finetune_final.pt
training/checkpoints/finetune_best.pt
training/checkpoints/pretrain_best.pt
```

If none exist, rerun pre-training:

```bash
python -m training.pretrain --max_steps 3000 --batch_size 64 --eval_interval 250 --save_interval 500
```

### The app says the tokenizer was not found

Run:

```bash
python -m tokenizer.train_tokenizer
```

### The dataset is missing

Place the CSV here:

```text
data/raw/ai_job_market_dataset.csv
```

Then rerun:

```bash
python -m data.prepare_data
```

### Fine-tuning appears to do nothing

Make sure instruction examples are loaded as complete `<bos>...<eos>` blocks, not individual lines. If you changed `build_instructions.py`, rerun:

```bash
python -m instruction_data.build_instructions
python -m training.finetune
```

### The raw LLM output is gibberish

This is expected for a very small model trained on a tiny corpus. Use the Gradio app's grounded final answer as the meaningful output. The optional model draft is included to demonstrate inference, not to serve as the trusted source of truth.

## Demo Guide

A recommended 1-2 minute demo flow:

1. Show the raw CSV in `data/raw/`.
2. Run or show `data.prepare_data` producing `corpus.txt`.
3. Show `tokenizer.json` and explain BPE tokenization.
4. Show `model/gpt.py` and explain the decoder-only transformer.
5. Show `pretrain_best.pt` and a few sample generations.
6. Show instruction fine-tuning and evaluation.
7. Launch the Gradio app.
8. Ask: `What skill should I learn for AI jobs in 2027?`
9. Point out the grounding facts, supporting table, final grounded answer, and optional model draft.

## Team

| Member | Responsibilities |
|---|---|
| James Bridges | Data preparation, tokenization, model architecture, pre-training |
| Aayush Adhikari | Instruction dataset, fine-tuning, evaluation |
| Brailey Sharpe | Trend analysis, grounding, Gradio app, documentation |

## Acknowledgments

The architecture follows GPT-style decoder-only transformer conventions established by Radford et al.'s GPT-2 work. The implementation is also inspired by Sebastian Raschka's *LLMs from Scratch* and Andrej Karpathy's nanoGPT-style educational code patterns.

All project code was written by the team for CS4985 GenAI: Theories & Practices.
