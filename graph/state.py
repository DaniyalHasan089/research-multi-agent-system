"""
ResearchState — the single typed dictionary that flows through every
node in the LangGraph pipeline. Each agent reads what it needs and
writes only its own outputs. No shared memory between agents.
"""

from typing import TypedDict


class ResearchState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    user_query: str           # Raw natural-language question from the user

    # ── Agent 1: Query Planner ───────────────────────────────────────────────
    domain: str               # Classified domain: cs, medicine, physics, etc.
    search_queries: list[str] # 5 targeted academic search strings

    # ── Agent 2: Paper Researcher ────────────────────────────────────────────
    clean_papers: list[dict]  # Deduped, abstract-filtered list of candidate papers

    # ── Agent 3: Analyzer & Ranker ───────────────────────────────────────────
    paper_analyses: list[dict] # Top-K papers with extracted insights, sorted by score

    # ── Agent 4: Report Writer ───────────────────────────────────────────────
    report_markdown: str      # Full structured Markdown report with [n] citations

    # ── Agent 5: Formatter ───────────────────────────────────────────────────
    report_html: str          # Styled HTML version of the report
    pdf_path: str             # Temp file path of the generated PDF for download
