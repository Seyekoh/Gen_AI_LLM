from __future__ import annotations
import pandas as pd
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data/raw/ai_job_market_dataset.csv"
OUT_PATH = PROJECT_ROOT / "instruction_data/instructions.txt"

random.seed(42)


def format_response(text: str) -> str:
    return text.strip().replace("\n", " ")


def main():
    df = pd.read_csv(RAW_CSV)

    examples = []

    # -------------------------------
    # PER-ROW HIGH QUALITY EXAMPLES
    # -------------------------------
    for _, row in df.iterrows():

        role = row["Job_Title"]
        country = row["Country"]
        salary = int(row["Salary_USD"])
        skill = row["Top_Skill"]
        remote = str(row["Remote"]).lower() == "yes"

        base_templates = [
            (
                f"Explain the salary expectations for a {role} in {country}.",
                f"A {role} in {country} typically earns around ${salary:,.0f} annually."
            ),
            (
                f"What is the earning potential of a {role}?",
                f"The average salary for a {role} is approximately ${salary:,.0f}."
            ),
            (
                f"Is working as a {role} financially rewarding?",
                f"Yes. A {role} earns about ${salary:,.0f}, which is considered competitive."
            ),
            (
                f"What is the key skill required for a {role}?",
                f"The most important skill for a {role} is {skill}."
            ),
            (
                f"Does a {role} require remote work capability?",
                f"{'Yes, this role supports remote work.' if remote else 'No, this role is typically on-site.'}"
            ),
        ]

        # slight randomization prevents memorization collapse
        random.shuffle(base_templates)
        examples.extend(base_templates)

    # -------------------------------
    # GLOBAL GENERALIZATION EXAMPLES
    # -------------------------------
    general_examples = [
        ("Are AI jobs high paying?",
         "Yes, AI-related roles are generally high-paying compared to many other fields."),

        ("Is AI a good career path?",
         "Yes, AI is a fast-growing field with strong long-term career opportunities."),

        ("Do AI jobs require programming?",
         "Yes, most AI jobs require strong programming and data skills."),

        ("Can AI jobs be remote?",
         "Many AI roles offer remote or hybrid work options."),

        ("What skills are important in AI jobs?",
         "Common skills include Python, machine learning, data analysis, and deep learning."),
    ]

    examples.extend(general_examples)

    # -------------------------------
    # WRITE FORMATTED SFT FILE
    # -------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for q, a in examples:
            q = format_response(q)
            a = format_response(a)

            f.write(
                "<bos>### Instruction:\n"
                f"{q}\n\n"
                "### Response:\n"
                f"{a}\n"
                "<eos>\n\n"
            )

    print(f"Wrote {len(examples)} instruction-response pairs")


if __name__ == "__main__":
    main()
