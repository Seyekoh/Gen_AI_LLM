"""Expanded Gradio app for grounded AI job-market question answering.

This app uses the from-scratch LLM as part of the pipeline, but it does not
trust the model to compute facts. The final user-facing answer is generated
from deterministic Pandas/sklearn trend analysis. The LLM output is shown as
a separate draft so the demo still demonstrates model inference without letting
unstable text ruin the answer quality.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any

import gradio as gr
import pandas as pd
import torch
from tokenizers import Tokenizer

# Allow `python app/gradio_app.py` as well as `python -m app.gradio_app`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.trends import (  # noqa: E402
    avg_salary_by_role,
    forecast_next_year,
    get_grounding_facts,
    salary_trend,
    skill_growth,
    top_skills_by_year,
)
from model.gpt import load_checkpoint  # noqa: E402


# ---------------------------------------------------------------------
# Paths / model loading
# ---------------------------------------------------------------------

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai_job_market_dataset.csv"

CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "training" / "checkpoints" / "finetune_final.pt",
    PROJECT_ROOT / "training" / "checkpoints" / "finetune_best.pt",
    PROJECT_ROOT / "training" / "checkpoints" / "pretrain_best.pt",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _select_checkpoint() -> Path:
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path

    searched = "\n".join(f"- {p}" for p in CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(f"No checkpoint found. Searched:\n{searched}")


def _load_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Expected data/raw/ai_job_market_dataset.csv."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    df["Year"] = df["Year"].astype(int)
    df["Salary_USD"] = pd.to_numeric(df["Salary_USD"], errors="coerce")
    df = df.dropna(subset=["Salary_USD"])
    return df


if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")

TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
CHECKPOINT_PATH = _select_checkpoint()
MODEL = load_checkpoint(CHECKPOINT_PATH, device=DEVICE)
MODEL.eval()

RAW_DF = _load_raw_data()


# ---------------------------------------------------------------------
# Basic metadata helpers
# ---------------------------------------------------------------------

ROLES = sorted(RAW_DF["Job_Title"].dropna().unique().tolist())
SKILLS = sorted(RAW_DF["Top_Skill"].dropna().unique().tolist())
COUNTRIES = sorted(RAW_DF["Country"].dropna().unique().tolist())
YEARS = sorted(RAW_DF["Year"].dropna().unique().astype(int).tolist())

LATEST_YEAR = max(YEARS)
EARLIEST_YEAR = min(YEARS)


ROLE_ALIASES = {
    "ml": "ML Engineer",
    "machine learning": "ML Engineer",
    "machine-learning": "ML Engineer",
    "ai": "AI Engineer",
    "artificial intelligence": "AI Engineer",
    "nlp": "NLP Engineer",
    "natural language": "NLP Engineer",
    "analyst": "Data Analyst",
    "data analyst": "Data Analyst",
    "scientist": "Data Scientist",
    "data scientist": "Data Scientist",
}

INTENT_KEYWORDS = {
    "forecast": ["forecast", "predict", "project", "projection", "future", "2027", "next year"],
    "salary": ["salary", "pay", "paid", "earn", "earning", "compensation", "income"],
    "skill": ["skill", "learn", "study", "growth", "growing", "fastest", "important"],
    "trend": ["trend", "over time", "year over year", "year-over-year", "increase", "decrease", "changed"],
    "compare": ["compare", "versus", "vs", "difference", "better", "higher"],
    "top": ["top", "best", "highest", "popular", "common", "most"],
    "remote": ["remote", "work from home", "hybrid", "on-site", "onsite"],
    "summary": ["summary", "overview", "dataset", "data", "describe"],
}


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _number(value: float) -> str:
    return f"{value:,.0f}"


def _normalize(text: str) -> str:
    return text.strip().lower()


def _extract_year(query: str) -> int | None:
    years = re.findall(r"\b(20\d{2})\b", query)
    if not years:
        return None
    return int(years[0])


def _extract_roles(query: str) -> list[str]:
    q = _normalize(query)
    found: list[str] = []

    # Exact role-name matching first.
    for role in sorted(ROLES, key=len, reverse=True):
        if role.lower() in q:
            found.append(role)

    # Alias matching second.
    for alias, role in ROLE_ALIASES.items():
        if alias in q and role in ROLES and role not in found:
            found.append(role)

    return found


def _extract_country(query: str) -> str | None:
    q = _normalize(query)

    for country in sorted(COUNTRIES, key=len, reverse=True):
        if country.lower() in q:
            return country

    country_aliases = {
        "united states": "USA",
        "us": "USA",
        "u.s.": "USA",
        "america": "USA",
        "uk": "UK",
        "united kingdom": "UK",
    }

    for alias, country in country_aliases.items():
        if alias in q and country in COUNTRIES:
            return country

    return None


def _detect_intent(query: str) -> str:
    q = _normalize(query)

    # Priority matters. A question like "forecast salary" should route
    # to forecast before generic salary.
    priority = [
        "compare",
        "forecast",
        "trend",
        "skill",
        "salary",
        "remote",
        "top",
        "summary",
    ]

    for intent in priority:
        if any(word in q for word in INTENT_KEYWORDS[intent]):
            return intent

    return "general"


def _safe_table(df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    """Return a display-friendly table with rounded numeric columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.head(max_rows).copy()

    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(2)

    return out


# ---------------------------------------------------------------------
# LLM generation helpers
# ---------------------------------------------------------------------

def _build_prompt(facts: str, question: str) -> str:
    return (
        "<bos>### Instruction:\n"
        "Answer using only the facts provided. Keep the answer short, clear, "
        "and grounded in the AI job-market dataset.\n"
        f"Facts: {facts}\n"
        f"Question: {question}\n\n"
        "### Response:\n"
    )


def _clean_response(text: str) -> str:
    text = text.strip()

    for marker in ["<eos>", "<bos>", "<pad>", "### Instruction:", "### Response:"]:
        if marker in text:
            text = text.split(marker, 1)[0]

    return text.strip()


def _looks_bad(text: str) -> bool:
    """Detect unusable generations from the tiny model."""
    text = text.strip()

    if not text:
        return True

    if len(text) < 20:
        return True

    bad_markers = ["<unk>", "<pad>", "<bos>", "### Instruction", "### Response"]
    if any(marker in text for marker in bad_markers):
        return True

    words = re.findall(r"\w+", text.lower())
    if len(words) < 5:
        return True

    unique_ratio = len(set(words)) / max(1, len(words))
    if unique_ratio < 0.45:
        return True

    # Repeated punctuation or obviously broken output.
    if "::::" in text or "####" in text:
        return True

    return False


def generate_model_draft(question: str, facts: str) -> str:
    """Run the from-scratch LLM and return its draft output.

    This is intentionally separated from the final answer because the model is
    small and can produce unstable text. The final answer is built from the
    deterministic analysis layer.
    """
    prompt = _build_prompt(facts, question)
    ids = TOKENIZER.encode(prompt).ids

    if len(ids) > MODEL.cfg.block_size:
        ids = ids[-MODEL.cfg.block_size:]

    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = MODEL.generate(
            x,
            max_new_tokens=50,
            temperature=0.35,
            top_k=15,
            eos_token_id=MODEL.cfg.eos_token_id,
        )

    generated_ids = out[0, len(ids):].tolist()
    draft = _clean_response(TOKENIZER.decode(generated_ids))

    if _looks_bad(draft):
        return (
            "[Model draft rejected as unstable. The final answer below is produced "
            "from the grounded analysis layer.]"
        )

    return draft


# ---------------------------------------------------------------------
# Deterministic answer builders
# ---------------------------------------------------------------------

def answer_dataset_summary() -> tuple[str, pd.DataFrame]:
    n_rows = len(RAW_DF)
    role_count = RAW_DF["Job_Title"].nunique()
    country_count = RAW_DF["Country"].nunique()
    skill_count = RAW_DF["Top_Skill"].nunique()
    min_salary = RAW_DF["Salary_USD"].min()
    max_salary = RAW_DF["Salary_USD"].max()
    avg_salary = RAW_DF["Salary_USD"].mean()

    role_table = avg_salary_by_role().rename(
        columns={
            "Job_Title": "Role",
            "avg_salary": "Average Salary",
            "count": "Records",
        }
    )

    answer = (
        f"The dataset contains {_number(n_rows)} AI job-market records from "
        f"{EARLIEST_YEAR} through {LATEST_YEAR}. It covers {role_count} roles, "
        f"{country_count} countries, and {skill_count} top-skill categories. "
        f"Across the full dataset, salaries range from {_money(min_salary)} to "
        f"{_money(max_salary)}, with an overall average of {_money(avg_salary)}. "
        f"The table shows the highest-paying roles by average salary."
    )

    return answer, _safe_table(role_table, max_rows=10)


def answer_salary(query: str) -> tuple[str, pd.DataFrame]:
    roles = _extract_roles(query)
    year = _extract_year(query)
    country = _extract_country(query)

    df = RAW_DF.copy()

    if year is not None:
        if year not in YEARS:
            return (
                f"The dataset only covers {EARLIEST_YEAR} through {LATEST_YEAR}, "
                f"so it does not contain salary records for {year}.",
                pd.DataFrame(),
            )
        df = df[df["Year"] == year]

    if country is not None:
        df = df[df["Country"] == country]

    if roles:
        role = roles[0]
        df = df[df["Job_Title"] == role]

        if df.empty:
            return (
                f"I found the role {role}, but there were no matching records "
                f"for the requested filters.",
                pd.DataFrame(),
            )

        avg_salary = df["Salary_USD"].mean()
        min_salary = df["Salary_USD"].min()
        max_salary = df["Salary_USD"].max()
        count = len(df)

        filters = []
        if year is not None:
            filters.append(str(year))
        if country is not None:
            filters.append(country)
        filter_text = f" for {' and '.join(filters)}" if filters else ""

        table = (
            df.groupby(["Job_Title", "Year"])
            .agg(
                Average_Salary=("Salary_USD", "mean"),
                Min_Salary=("Salary_USD", "min"),
                Max_Salary=("Salary_USD", "max"),
                Records=("Salary_USD", "size"),
            )
            .reset_index()
            .sort_values(["Year", "Average_Salary"], ascending=[True, False])
        )

        answer = (
            f"Based on {count} matching records, the average salary for "
            f"{role}{filter_text} is about {_money(avg_salary)}. "
            f"The observed range in the dataset is {_money(min_salary)} to "
            f"{_money(max_salary)}. Because this is a small educational dataset, "
            f"these numbers should be treated as dataset trends rather than real "
            f"market salary guarantees."
        )

        return answer, _safe_table(table, max_rows=10)

    # No specific role: show role ranking.
    grouped = (
        df.groupby("Job_Title")
        .agg(
            Average_Salary=("Salary_USD", "mean"),
            Min_Salary=("Salary_USD", "min"),
            Max_Salary=("Salary_USD", "max"),
            Records=("Salary_USD", "size"),
        )
        .reset_index()
        .sort_values("Average_Salary", ascending=False)
    )

    if grouped.empty:
        return "No matching salary records were found.", pd.DataFrame()

    top = grouped.iloc[0]

    filter_bits = []
    if year is not None:
        filter_bits.append(str(year))
    if country is not None:
        filter_bits.append(country)
    filter_text = f" for {' and '.join(filter_bits)}" if filter_bits else ""

    answer = (
        f"The highest average salary role{filter_text} is {top.Job_Title}, "
        f"with an average salary of about {_money(top.Average_Salary)}. "
        f"The table ranks roles by average salary."
    )

    return answer, _safe_table(grouped, max_rows=10)


def answer_forecast(query: str) -> tuple[str, pd.DataFrame]:
    roles = _extract_roles(query)
    target_year = _extract_year(query) or (LATEST_YEAR + 1)

    if target_year <= LATEST_YEAR:
        target_year = LATEST_YEAR + 1

    if not roles:
        # If the user asks a generic forecast question, forecast the top salary role.
        top_role = avg_salary_by_role().iloc[0]["Job_Title"]
        roles = [top_role]

    role = roles[0]

    try:
        forecast = forecast_next_year(role, target_year)
        trend = salary_trend(role)
    except ValueError as exc:
        return str(exc), pd.DataFrame()

    latest_row = trend[trend["Year"] == LATEST_YEAR]
    latest_salary = None
    if not latest_row.empty:
        latest_salary = float(latest_row.iloc[0]["avg_salary"])

    display = trend.rename(
        columns={
            "Year": "Year",
            "avg_salary": "Average Salary",
            "count": "Records",
        }
    )

    forecast_row = pd.DataFrame(
        [
            {
                "Year": target_year,
                "Average Salary": forecast,
                "Records": "Forecast",
            }
        ]
    )
    display = pd.concat([display, forecast_row], ignore_index=True)

    if latest_salary is not None:
        direction = "higher than" if forecast >= latest_salary else "lower than"
        comparison = (
            f"This is {direction} the {LATEST_YEAR} dataset average of "
            f"{_money(latest_salary)}."
        )
    else:
        comparison = ""

    answer = (
        f"The simple linear trend forecast estimates the average salary for "
        f"{role} in {target_year} at about {_money(forecast)}. {comparison} "
        f"This is an illustrative regression over the dataset years, not a real "
        f"market prediction."
    )

    return answer, _safe_table(display, max_rows=10)


def answer_skill(query: str) -> tuple[str, pd.DataFrame]:
    q = _normalize(query)
    year = _extract_year(query)

    growth = skill_growth()
    skill_summary = (
        growth[["Top_Skill", "slope"]]
        .drop_duplicates()
        .sort_values("slope", ascending=False)
        .reset_index(drop=True)
    )

    top_growth = skill_summary.head(3)

    if year is not None and year in YEARS:
        top_by_year = top_skills_by_year()
        row = top_by_year[top_by_year["Year"] == year]

        if not row.empty:
            top_skill = row.iloc[0]["Top_Skill"]
            freq = int(row.iloc[0]["frequency"])
            answer = (
                f"In {year}, the most common top skill in the dataset is "
                f"{top_skill}, appearing in {freq} records. Across the full "
                f"{EARLIEST_YEAR}-{LATEST_YEAR} range, the fastest-growing skills "
                f"by frequency are {', '.join(top_growth['Top_Skill'].tolist())}."
            )
            return answer, _safe_table(top_growth, max_rows=10)

    if "learn" in q or "study" in q or "career" in q:
        names = top_growth["Top_Skill"].tolist()
        answer = (
            f"Based on the dataset trend, the strongest skills to prioritize are "
            f"{', '.join(names)}. These have the highest positive frequency trend "
            f"from {EARLIEST_YEAR} to {LATEST_YEAR}. For a student or early-career "
            f"candidate, I would treat these as practical learning targets rather "
            f"than guaranteed job requirements."
        )
    else:
        names = top_growth["Top_Skill"].tolist()
        answer = (
            f"The fastest-growing skills by frequency are {', '.join(names)}. "
            f"The slope column shows the simple linear trend in how often each "
            f"skill appears across the dataset years."
        )

    return answer, _safe_table(skill_summary, max_rows=10)


def answer_trend(query: str) -> tuple[str, pd.DataFrame]:
    roles = _extract_roles(query)

    if roles:
        role = roles[0]
        trend = salary_trend(role)

        if trend.empty:
            return f"No salary trend records were found for {role}.", pd.DataFrame()

        first = trend.iloc[0]
        last = trend.iloc[-1]
        change = float(last["avg_salary"] - first["avg_salary"])
        pct = change / float(first["avg_salary"]) * 100 if first["avg_salary"] else 0

        direction = "increased" if change >= 0 else "decreased"

        table = trend.rename(
            columns={
                "avg_salary": "Average Salary",
                "count": "Records",
            }
        )

        answer = (
            f"For {role}, average salary {direction} from "
            f"{_money(first['avg_salary'])} in {int(first['Year'])} to "
            f"{_money(last['avg_salary'])} in {int(last['Year'])}. "
            f"That is a change of {_money(abs(change))}, or about "
            f"{abs(pct):.1f}%. The table shows the year-over-year trend."
        )

        return answer, _safe_table(table, max_rows=10)

    # If no role is given, summarize all role trends.
    rows = []
    for role in ROLES:
        trend = salary_trend(role)
        if len(trend) < 2:
            continue

        first = trend.iloc[0]
        last = trend.iloc[-1]
        change = float(last["avg_salary"] - first["avg_salary"])
        pct = change / float(first["avg_salary"]) * 100 if first["avg_salary"] else 0

        rows.append(
            {
                "Role": role,
                "First Year": int(first["Year"]),
                "First Avg Salary": float(first["avg_salary"]),
                "Latest Year": int(last["Year"]),
                "Latest Avg Salary": float(last["avg_salary"]),
                "Change": change,
                "Percent Change": pct,
            }
        )

    table = pd.DataFrame(rows).sort_values("Change", ascending=False)

    if table.empty:
        return "No trend records were available.", pd.DataFrame()

    top = table.iloc[0]
    answer = (
        f"Across roles, {top.Role} shows the largest average salary increase "
        f"from {EARLIEST_YEAR} to {LATEST_YEAR}, rising by about "
        f"{_money(top.Change)}. The table compares salary changes by role."
    )

    return answer, _safe_table(table, max_rows=10)


def answer_compare(query: str) -> tuple[str, pd.DataFrame]:
    roles = _extract_roles(query)

    if len(roles) < 2:
        salary_table = avg_salary_by_role().rename(
            columns={
                "Job_Title": "Role",
                "avg_salary": "Average Salary",
                "count": "Records",
            }
        )

        answer = (
            "I could not find two specific roles in the question, so I compared "
            "all roles by average salary instead. Ask something like "
            "'Compare ML Engineer and Data Scientist' for a direct comparison."
        )

        return answer, _safe_table(salary_table, max_rows=10)

    selected_roles = roles[:2]
    rows = []

    for role in selected_roles:
        role_df = RAW_DF[RAW_DF["Job_Title"] == role]
        rows.append(
            {
                "Role": role,
                "Average Salary": role_df["Salary_USD"].mean(),
                "Min Salary": role_df["Salary_USD"].min(),
                "Max Salary": role_df["Salary_USD"].max(),
                "Records": len(role_df),
                "Most Common Skill": role_df["Top_Skill"].mode().iloc[0],
                "Remote Rate": (role_df["Remote"].astype(str).str.lower() == "yes").mean() * 100,
            }
        )

    table = pd.DataFrame(rows).sort_values("Average Salary", ascending=False)
    winner = table.iloc[0]
    other = table.iloc[1]
    diff = float(winner["Average Salary"] - other["Average Salary"])

    answer = (
        f"Between {selected_roles[0]} and {selected_roles[1]}, the dataset shows "
        f"{winner.Role} with the higher average salary at "
        f"{_money(winner['Average Salary'])}. That is about {_money(diff)} higher "
        f"than {other.Role}. The table also compares common skills and remote-work "
        f"rates."
    )

    return answer, _safe_table(table, max_rows=10)


def answer_remote(query: str) -> tuple[str, pd.DataFrame]:
    roles = _extract_roles(query)
    df = RAW_DF.copy()

    if roles:
        df = df[df["Job_Title"] == roles[0]]

    if df.empty:
        return "No matching remote-work records were found.", pd.DataFrame()

    table = (
        df.assign(Remote_Flag=df["Remote"].astype(str).str.lower().eq("yes"))
        .groupby("Job_Title")
        .agg(
            Remote_Rate=("Remote_Flag", lambda x: x.mean() * 100),
            Remote_Count=("Remote_Flag", "sum"),
            Records=("Remote_Flag", "size"),
        )
        .reset_index()
        .sort_values("Remote_Rate", ascending=False)
    )

    if roles:
        row = table.iloc[0]
        answer = (
            f"For {row.Job_Title}, about {row.Remote_Rate:.1f}% of records in the "
            f"dataset are marked remote. That means {int(row.Remote_Count)} out of "
            f"{int(row.Records)} matching records support remote work."
        )
    else:
        row = table.iloc[0]
        answer = (
            f"The role with the highest remote-work rate is {row.Job_Title}, with "
            f"about {row.Remote_Rate:.1f}% of records marked remote. The table "
            f"compares remote-work rates across roles."
        )

    return answer, _safe_table(table, max_rows=10)


def answer_top(query: str) -> tuple[str, pd.DataFrame]:
    q = _normalize(query)

    if "skill" in q or "popular" in q or "common" in q:
        table = top_skills_by_year().rename(
            columns={
                "Top_Skill": "Top Skill",
                "frequency": "Frequency",
            }
        )
        latest = table[table["Year"] == LATEST_YEAR].iloc[0]

        answer = (
            f"In the latest dataset year, {LATEST_YEAR}, the top skill is "
            f"{latest['Top Skill']} with {int(latest['Frequency'])} records. "
            f"The table shows the top skill for each year."
        )

        return answer, _safe_table(table, max_rows=10)

    table = avg_salary_by_role().rename(
        columns={
            "Job_Title": "Role",
            "avg_salary": "Average Salary",
            "count": "Records",
        }
    )

    top = table.iloc[0]

    answer = (
        f"The highest-paying role by average salary is {top.Role}, with an "
        f"average salary of about {_money(top['Average Salary'])}. The table "
        f"shows the role ranking."
    )

    return answer, _safe_table(table, max_rows=10)


def answer_general(query: str) -> tuple[str, pd.DataFrame]:
    # A useful default: combine top salary role and fastest-growing skill.
    salary_table = avg_salary_by_role()
    growth_table = skill_growth()[["Top_Skill", "slope"]].drop_duplicates()

    top_role = salary_table.iloc[0]
    top_skill = growth_table.sort_values("slope", ascending=False).iloc[0]

    answer = (
        f"Based on the dataset, the highest average salary role is "
        f"{top_role.Job_Title} at about {_money(top_role.avg_salary)}. "
        f"The fastest-growing skill by frequency is {top_skill.Top_Skill}. "
        f"For more detail, ask about salaries, forecasts, skill growth, remote "
        f"work, or comparisons between roles."
    )

    display = salary_table.rename(
        columns={
            "Job_Title": "Role",
            "avg_salary": "Average Salary",
            "count": "Records",
        }
    )

    return answer, _safe_table(display, max_rows=10)


def build_grounded_answer(question: str) -> tuple[str, str, pd.DataFrame, str]:
    """Route a user question to the best deterministic analysis answer."""
    intent = _detect_intent(question)
    facts = get_grounding_facts(question)

    if intent == "salary":
        answer, table = answer_salary(question)
    elif intent == "forecast":
        answer, table = answer_forecast(question)
    elif intent == "skill":
        answer, table = answer_skill(question)
    elif intent == "trend":
        answer, table = answer_trend(question)
    elif intent == "compare":
        answer, table = answer_compare(question)
    elif intent == "remote":
        answer, table = answer_remote(question)
    elif intent == "top":
        answer, table = answer_top(question)
    elif intent == "summary":
        answer, table = answer_dataset_summary()
    else:
        answer, table = answer_general(question)

    return answer, facts, table, intent


# ---------------------------------------------------------------------
# Main Gradio event function
# ---------------------------------------------------------------------

def generate_answer(question: str, use_model_draft: bool) -> tuple[str, str, str, pd.DataFrame, str]:
    question = question.strip()

    if not question:
        return (
            "Please enter a question.",
            "",
            "",
            pd.DataFrame(),
            "No query entered.",
        )

    grounded_answer, facts, table, intent = build_grounded_answer(question)

    if use_model_draft:
        model_draft = generate_model_draft(question, facts)
    else:
        model_draft = "Model draft disabled. Final answer uses grounded analysis only."

    route_info = (
        f"Detected intent: {intent}\n"
        f"Loaded checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}\n"
        f"Device: {DEVICE}\n"
        f"Dataset years: {EARLIEST_YEAR}-{LATEST_YEAR}\n"
        f"Roles: {', '.join(ROLES)}\n"
        f"Skills: {', '.join(SKILLS)}"
    )

    return grounded_answer, facts, model_draft, table, route_info


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

DESCRIPTION = """
Ask a question about AI job roles, salaries, skills, forecasts, remote work,
or trends. The final answer is grounded in deterministic dataset analysis.
The from-scratch LLM is still loaded and can produce a draft, but unstable
model text is not used as the source of truth.
"""

EXAMPLES = [
    "What is the average salary for an ML Engineer?",
    "Forecast AI Engineer salary in 2027.",
    "Compare ML Engineer and Data Scientist.",
    "Which role has the highest average salary?",
    "How has Data Scientist salary changed over time?",
    "Which skills are growing the fastest?",
    "Are AI jobs remote?",
    "Give me a summary of the dataset.",
]


with gr.Blocks() as demo:
    gr.Markdown("# AI Job Market LLM")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Question",
                placeholder="What's a good skill to learn for AI in 2027?",
                lines=3,
            )

            with gr.Row():
                generate_button = gr.Button("Generate", variant="primary")
                use_model_draft = gr.Checkbox(
                    label="Show from-scratch LLM draft",
                    value=True,
                )

            final_answer = gr.Textbox(
                label="Grounded Final Answer",
                lines=8,
            )

            support_table = gr.Dataframe(
                label="Supporting Analysis Table",
                interactive=False,
                wrap=True,
            )

        with gr.Column(scale=1):
            facts = gr.Textbox(
                label="Retrieved Grounding Facts",
                lines=5,
            )

            model_draft = gr.Textbox(
                label="From-Scratch LLM Draft",
                lines=8,
            )

            route_info = gr.Textbox(
                label="App Diagnostics",
                lines=8,
                interactive=False,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=question,
    )

    generate_button.click(
        generate_answer,
        inputs=[question, use_model_draft],
        outputs=[final_answer, facts, model_draft, support_table, route_info],
    )

    question.submit(
        generate_answer,
        inputs=[question, use_model_draft],
        outputs=[final_answer, facts, model_draft, support_table, route_info],
    )


if __name__ == "__main__":
    demo.launch()