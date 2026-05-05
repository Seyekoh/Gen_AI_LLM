"""
Prepare the AI Job Market dataset for LLM pre-training.

Reads data/raw/ai_job_market_dataset.csv and writes data/processed/corpus.txt
where each row of the CSV is converted into 1-3 natural-language sentences
using templates. Multiple templates per row add lexical diversity.

Run:  python -m data.prepare_data
"""
from __future__ import annotations
import random
from pathlib import Path
import pandas as pd

# Reproducibility
random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "ai_job_market_dataset.csv"
OUT_TXT = PROJECT_ROOT / "data" / "processed" / "corpus.txt"


# Sentence templates. Each row in the CSV gets rendered through
# several of these to produce a more varied corpus.
ROW_TEMPLATES = [
    "In {year}, a {level}-level {role} in {country} working at a {company} company earned ${salary:,}. Their top skill was {skill}. Remote work: {remote}.",
    "A {role} based in {country} at a {company} firm reported a salary of ${salary:,} in {year}. The role required {skill} as the primary skill, at {level} level. Remote: {remote}.",
    "{year} data: {role} ({level}) in {country}, {company} employer, salary ${salary:,}, top skill {skill}, remote {remote}.",
    "The {role} position in {country} paid ${salary:,} in {year} for {level}-level professionals. {skill} was the most important skill. The role was {remote_phrase}.",
    "{level} {role}s in {country} at {company} companies made ${salary:,} on average in {year}, with {skill} listed as the top required skill. Remote work was {remote_phrase}.",
]

# Aggregate-level templates produced once per (year, role) group.
# These give the model some "summary" patterns to learn.
GROUP_TEMPLATES = [
    "In {year}, the average salary for a {role} was ${avg_salary:,.0f} across {n} reported positions.",
    "Across {n} {role} roles in {year}, salaries averaged ${avg_salary:,.0f}.",
    "{year} saw {n} {role} positions reported, with mean compensation of ${avg_salary:,.0f}.",
]


def remote_phrase(remote_flag: str) -> str:
    return "available remotely" if remote_flag.strip().lower() == "yes" else "on-site only"


def render_row(row: pd.Series, n_templates: int = 3) -> list[str]:
    """Render one CSV row through n_templates randomly chosen templates."""
    chosen = random.sample(ROW_TEMPLATES, k=min(n_templates, len(ROW_TEMPLATES)))
    sentences = []
    for tpl in chosen:
        sentences.append(tpl.format(
            year=int(row["Year"]),
            role=row["Job_Title"],
            country=row["Country"],
            company=row["Company_Type"],
            level=row["Experience_Level"],
            salary=int(row["Salary_USD"]),
            skill=row["Top_Skill"],
            remote=row["Remote"],
            remote_phrase=remote_phrase(row["Remote"]),
        ))
    return sentences


def render_groups(df: pd.DataFrame) -> list[str]:
    """Generate aggregate sentences over (year, role) groups."""
    sentences = []
    grouped = df.groupby(["Year", "Job_Title"]).agg(
        avg_salary=("Salary_USD", "mean"),
        n=("Salary_USD", "size"),
    ).reset_index()
    for _, g in grouped.iterrows():
        for tpl in GROUP_TEMPLATES:
            sentences.append(tpl.format(
                year=int(g["Year"]),
                role=g["Job_Title"],
                avg_salary=g["avg_salary"],
                n=int(g["n"]),
            ))
    return sentences


def main() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_CSV}. "
            "Place ai_job_market_dataset.csv in data/raw/ and rerun."
        )

    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Per-row sentences
    all_sentences: list[str] = []
    for _, row in df.iterrows():
        all_sentences.extend(render_row(row, n_templates=3))

    # Aggregate sentences
    all_sentences.extend(render_groups(df))

    # Shuffle so the model doesn't see all rows of the same year together
    random.shuffle(all_sentences)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TXT.open("w", encoding="utf-8") as f:
        for s in all_sentences:
            f.write(s + "\n")

    total_chars = sum(len(s) for s in all_sentences)
    print(f"Wrote {len(all_sentences):,} sentences to {OUT_TXT}")
    print(f"Total characters: {total_chars:,} (~{total_chars/1024:.1f} KB)")
    print(f"Average sentence length: {total_chars/len(all_sentences):.1f} chars")


if __name__ == "__main__":
    main()
