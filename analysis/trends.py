"""
Trend analysis and grounding utilities for the AI job-market LLM app.

This module intentionally keeps the analysis simple and deterministic. The LLM
should not be trusted to compute facts at inference time; instead, the Gradio app
uses these functions to retrieve small, grounded facts from the CSV first.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai_job_market_dataset.csv"


@lru_cache(maxsize=1)
def _load_data() -> pd.DataFrame:
    """Load and lightly validate the raw job-market dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Expected data/raw/ai_job_market_dataset.csv."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"Year", "Job_Title", "Country", "Salary_USD", "Top_Skill", "Remote"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Year"] = df["Year"].astype(int)
    df["Salary_USD"] = pd.to_numeric(df["Salary_USD"], errors="coerce")
    df = df.dropna(subset=["Salary_USD"])
    return df


def _match_role(role: str) -> str:
    """Return the canonical role name from the dataset using simple matching."""
    df = _load_data()
    roles = sorted(df["Job_Title"].dropna().unique())
    query = role.strip().lower()

    for r in roles:
        if r.lower() == query:
            return r

    for r in roles:
        r_lower = r.lower()
        if query in r_lower or r_lower in query:
            return r

    raise ValueError(f"Unknown role: {role!r}. Available roles: {', '.join(roles)}")


def _role_from_query(query: str) -> str | None:
    """Find a dataset role mentioned inside a free-text query."""
    df = _load_data()
    q = query.lower()

    for role in sorted(df["Job_Title"].dropna().unique(), key=len, reverse=True):
        if role.lower() in q:
            return role

    aliases = {
        "ml": "ML Engineer",
        "machine learning": "ML Engineer",
        "ai": "AI Engineer",
        "nlp": "NLP Engineer",
        "analyst": "Data Analyst",
        "scientist": "Data Scientist",
    }
    for key, value in aliases.items():
        if key in q and value in set(df["Job_Title"]):
            return value

    return None


def top_skills_by_year() -> pd.DataFrame:
    """Top skill per year, by frequency."""
    df = _load_data()
    counts = (
        df.groupby(["Year", "Top_Skill"])
        .size()
        .reset_index(name="frequency")
        .sort_values(["Year", "frequency", "Top_Skill"], ascending=[True, False, True])
    )
    return counts.groupby("Year", as_index=False).first()


def avg_salary_by_role(year: int | None = None) -> pd.DataFrame:
    """Average salary by job title. Optionally filtered to a year."""
    df = _load_data()
    if year is not None:
        df = df[df["Year"] == int(year)]

    return (
        df.groupby("Job_Title")
        .agg(avg_salary=("Salary_USD", "mean"), count=("Salary_USD", "size"))
        .reset_index()
        .sort_values("avg_salary", ascending=False)
    )


def salary_trend(role: str) -> pd.DataFrame:
    """Year-over-year average salary for a given role."""
    canonical_role = _match_role(role)
    df = _load_data()
    return (
        df[df["Job_Title"] == canonical_role]
        .groupby("Year")
        .agg(avg_salary=("Salary_USD", "mean"), count=("Salary_USD", "size"))
        .reset_index()
        .sort_values("Year")
    )


def skill_growth() -> pd.DataFrame:
    """Each skill's frequency over time, plus a simple linear trend slope."""
    df = _load_data()
    years = sorted(df["Year"].unique())
    skills = sorted(df["Top_Skill"].dropna().unique())

    counts = (
        df.groupby(["Top_Skill", "Year"])
        .size()
        .reset_index(name="frequency")
    )

    grid = pd.MultiIndex.from_product([skills, years], names=["Top_Skill", "Year"])
    counts = (
        counts.set_index(["Top_Skill", "Year"])
        .reindex(grid, fill_value=0)
        .reset_index()
    )

    slopes: dict[str, float] = {}
    for skill, group in counts.groupby("Top_Skill"):
        x = group[["Year"]].to_numpy()
        y = group["frequency"].to_numpy()
        slopes[skill] = float(LinearRegression().fit(x, y).coef_[0])

    counts["slope"] = counts["Top_Skill"].map(slopes)
    return counts.sort_values(["slope", "Top_Skill", "Year"], ascending=[False, True, True])


def forecast_next_year(role: str, target_year: int = 2027) -> float:
    """Linear regression forecast of average salary for a role."""
    trend = salary_trend(role)
    if len(trend) < 2:
        return float(trend["avg_salary"].iloc[-1])

    x = trend[["Year"]].to_numpy()
    y = trend["avg_salary"].to_numpy()
    model = LinearRegression().fit(x, y)
    prediction = model.predict([[int(target_year)]])[0]
    return float(prediction)


def get_grounding_facts(query: str) -> str:
    """
    Return short relevant facts from the analysis for use in LLM prompting.

    This intentionally uses simple keyword matching and keeps output short so it
    can fit comfortably inside the model's small context window.
    """
    q = query.lower()
    role = _role_from_query(query)

    if any(word in q for word in ["forecast", "project", "projection", "2027", "future"]):
        if role:
            salary = forecast_next_year(role, 2027)
            return f"2027 illustrative forecast: {role} avg salary about ${salary:,.0f}."
        top_role = avg_salary_by_role().iloc[0]
        return f"Highest avg salary role is {top_role.Job_Title} at about ${top_role.avg_salary:,.0f}."

    if any(word in q for word in ["salary", "pay", "earn", "earning", "compensation"]):
        if role:
            latest_year = int(_load_data()["Year"].max())
            salaries = avg_salary_by_role(latest_year)
            row = salaries[salaries["Job_Title"] == role]
            if not row.empty:
                return f"In {latest_year}, {role} avg salary is about ${row.iloc[0].avg_salary:,.0f}."
        top_role = avg_salary_by_role().iloc[0]
        return f"Overall highest avg salary: {top_role.Job_Title}, about ${top_role.avg_salary:,.0f}."

    if any(word in q for word in ["skill", "learn", "growth", "growing", "fastest"]):
        growth = skill_growth()
        top = growth[["Top_Skill", "slope"]].drop_duplicates().head(2)
        names = ", ".join(top["Top_Skill"].tolist())
        return f"Fastest-growing skills by frequency are {names}."

    if any(word in q for word in ["top", "popular", "common"]):
        latest_year = int(_load_data()["Year"].max())
        row = top_skills_by_year().query("Year == @latest_year").iloc[0]
        return f"Top skill in {latest_year}: {row.Top_Skill} ({int(row.frequency)} records)."

    avg = avg_salary_by_role().iloc[0]
    growth = skill_growth()[["Top_Skill", "slope"]].drop_duplicates().iloc[0]
    return f"Top salary role: {avg.Job_Title}; fastest-growing skill: {growth.Top_Skill}."
