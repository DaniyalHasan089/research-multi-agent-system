"""
ResearchState — the single typed dictionary that flows through every
node in the LangGraph pipeline. Each agent reads what it needs and
writes only its own outputs. No shared memory between agents.
"""

from typing import Optional, TypedDict


class ResearchState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    user_query: str           # Raw natural-language question from the user
    year_from: Optional[int]  # Optional: include papers published from this year
    year_to: Optional[int]    # Optional: include papers published up to this year

    # ── Agent 1: Query Planner ───────────────────────────────────────────────
    domain: str               # Classified domain: cs, medicine, physics, etc.
    clarified_focus: Optional[str] # Explicit disambiguated research scope & boundary
    search_queries: list[str] # 5 targeted academic search strings

    # ── Agent 2: Paper Researcher ────────────────────────────────────────────
    clean_papers: list[dict]  # Deduped, abstract-filtered list of candidate papers

    # ── Agent 3: Analyzer & Ranker ───────────────────────────────────────────
    paper_analyses: list[dict] # Top-K papers with extracted insights, sorted by score
    search_stats: dict         # Pipeline counts for PRISMA-style reporting

    # ── Agent 4: Report Writer ───────────────────────────────────────────────
    report_markdown: str      # Full structured Markdown report with [n] citations

    # ── Agent 5: Formatter ───────────────────────────────────────────────────
    report_html: str          # Styled HTML version of the report
    pdf_path: str             # Temp file path of the generated PDF for download
