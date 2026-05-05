# Teammate B — Trend Analysis, Grounding & Deployment

Welcome! Your job is the **product layer** — the part the user actually
interacts with. You'll build the trend analysis (descriptive + simple
forecasting), the grounding logic that feeds analysis into the model at
inference time, and the Gradio web app.

---

## What's already done for you

- `data/raw/ai_job_market_dataset.csv` — the raw dataset
- `data/processed/corpus.txt` — templated training corpus
- `tokenizer/tokenizer.json` — trained BPE tokenizer
- `model/gpt.py` — from-scratch GPT, with `load_checkpoint`
- `training/checkpoints/pretrain_best.pt` — the pre-trained model (step 1500, lowest val loss)

---

## Your deliverables

### 1. `analysis/trends.py`

Pandas-based descriptive analysis + simple forecasting. **No LLM needed.**

The dataset has 5 years (2022–2026) × 5 roles × 5 countries × 5 skills.
That's small, which is good — you can compute everything cheaply.

Required functions (these will be called by the Gradio app):

```python
def top_skills_by_year() -> pd.DataFrame:
    """Top skill per year, by frequency."""

def avg_salary_by_role(year: int | None = None) -> pd.DataFrame:
    """Average salary by job title. Optionally filtered to a year."""

def salary_trend(role: str) -> pd.DataFrame:
    """Year-over-year average salary for a given role."""

def skill_growth() -> pd.DataFrame:
    """Each skill's frequency over time, plus a simple linear trend slope."""

def forecast_next_year(role: str, target_year: int = 2027) -> float:
    """Linear regression forecast of average salary for a role.
       Use sklearn.linear_model.LinearRegression on (year, avg_salary).
       Honest about uncertainty — return a single number; mention in the
       README that this is illustrative, not predictive."""

def get_grounding_facts(query: str) -> str:
    """Given a free-text user question, return a short string of relevant
       facts pulled from the analysis. Used to ground the LLM at inference
       time. Keep it under ~200 chars. Simple keyword matching is fine."""
```

Don't over-engineer. A 100-line file is plenty.

### 2. `app/gradio_app.py`

A simple Gradio interface with:

- A text box for the user's question
- A "Generate" button
- An output area for the model's response
- (Optional) a side panel showing the grounding facts that were retrieved

Inference flow:

```python
user_question = "What's a good skill to learn for AI in 2027?"

# 1. Pull facts from analysis
facts = get_grounding_facts(user_question)
# e.g. "Python and PyTorch are the fastest-growing skills 2022-2026.
#       ML Engineer salaries are projected to reach $X in 2027."

# 2. Build a grounded prompt
prompt = f"Facts: {facts}\nQuestion: {user_question}\nAnswer:"

# 3. Generate
ids = tokenizer.encode(prompt).ids
x = torch.tensor([ids], dtype=torch.long, device=device)
out = model.generate(x, max_new_tokens=80, temperature=0.7, top_k=40,
                     eos_token_id=model.cfg.eos_token_id)
response = tokenizer.decode(out[0].tolist())
# Trim the prompt portion off the response
```

Use Gradio's default theme. Don't customize CSS. Function over form.

Loading the model + tokenizer once at startup (not per-request) is
important for responsiveness — put it at module level.

### 3. README polish + demo recording

- Fill in the team names in the main `README.md`
- Add screenshots of the Gradio app to a `docs/screenshots/` directory
- Record a 1–2 minute demo video (OBS, screen recording, anything) and
  link it from the README. Show: data → model → answer to a question.
  Doesn't need to be polished — just demonstrate the pipeline running.

---

## Important notes

### The model's output will be limited

Manage your expectations. The model was pre-trained on ~840 KB of templated
text. It will produce outputs like:

- Plausible-sounding job market sentences
- Sometimes correct numbers (especially after fine-tuning)
- Sometimes invented numbers
- Sometimes degenerate output (repetitive, grammatically off)

This is **expected** for a from-scratch 3M-parameter model on a tiny
corpus. Don't try to fix it by training longer — you'll just overfit
harder. The point is the pipeline.

The grounding facts are what give the answers any real signal. The LLM
is mostly a natural-language wrapper around the actual analysis.

### Be honest in the demo

In your video / README, point out one example where the LLM gives a good
answer (because of grounding) and one where it gives a weird answer (just
the model being small). This reads as honest research, not as failure.

---

## File interfaces

```python
# Loading the model (after Teammate A's fine-tuning is done):
from model.gpt import load_checkpoint
model = load_checkpoint("training/checkpoints/finetune_final.pt", device="cuda")

# Loading the tokenizer:
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
```

If `finetune_final.pt` doesn't exist yet, fall back to
`training/checkpoints/pretrain_best.pt` so you can develop the app in
parallel before Teammate A finishes.

---

## Estimated time

- `analysis/trends.py`: 1–2 hours
- `app/gradio_app.py`: 2 hours
- README + demo video: 1 hour

Total: **4–5 hours** of focused work.

Ping me with questions. Good luck!
