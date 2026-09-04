"""
Agent 4: Report Writer (LLM)

Synthesizes all paper analyses into a structured Markdown report.
The report has 7 sections with inline [n] citations referencing the papers.

Output is written to ResearchState: report_markdown
"""

from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import ResearchState
from utils.llm_factory import get_llm, extract_text


SYSTEM_PROMPT = """You are an expert academic research synthesizer. Your task is to write a detailed, comprehensive research report in Markdown format based on a set of analyzed research papers.

The report MUST contain exactly these 8 sections in order:

# [Research Topic] — Research Report

## 1. Executive Summary
Write 5-7 sentences covering: the research question, why this topic matters, how many papers were reviewed, the dominant findings across the corpus, and the most important open challenge. This should give a reader a complete picture of the field at a glance.

## 2. Research Landscape
Write 3-4 paragraphs covering:
- Historical context: how did this research area emerge and evolve?
- Major milestones or paradigm shifts in the field
- The current state: how active is this area, what institutions or communities are driving it?
- Key sub-fields or competing schools of thought, with citations [n]

## 3. Key Themes & Trends
Identify and elaborate on 4-6 recurring themes across the papers. For each theme:
- Give it a descriptive heading
- Write 2-3 sentences explaining the theme and cite the papers [n] that exemplify it
- Note whether the theme is emerging, established, or contested

## 4. Current & Ongoing Work
Describe what researchers are actively working on RIGHT NOW based on the papers reviewed:
- Highlight active research directions, methodologies, and benchmarks present in the papers [n]
- Identify architectures and techniques currently being explored, refined, or compared
- Reference recent papers and preprints that signal where the field is heading [n]
- Strict Grounding Rule: ONLY reference institutions, labs, or benchmarks that are EXPLICITLY cited in the provided paper analyses. If specific lab or university affiliations are not present in the sources, do NOT invent or assume them; describe the ongoing research purely in terms of the verified technical methods and problem spaces.

## 5. Paper-by-Paper Breakdown
For EVERY paper, write a dedicated subsection in this exact format:

### [n] [Full Paper Title](URL)
**Summary:** 4-5 sentences: what is this paper about, what problem does it solve, and why does it matter?

**Methodology:** 2-3 sentences: what specific techniques, datasets, or experimental approach does it use?

**Key Findings:** 2-3 sentences: what are the most important results, including specific numbers or benchmarks where available?

**Limitations:** 1-2 sentences: what does this paper leave unresolved or what are its assumptions?

**Relevance:** X/10 — one sentence explaining relevance to the research question.

## 6. Methodological Comparison
Write 2-3 paragraphs comparing approaches across papers:
- What methods or frameworks are most commonly used and why?
- Where do papers disagree on methodology, and what are the trade-offs?
- What methodological gaps exist (e.g., missing baselines, untested settings)?

## 7. Gaps & Open Problems
Write 3-4 paragraphs identifying:
- The most critical unanswered questions in this field
- Specific limitations shared across multiple papers [n]
- Promising but underexplored directions future researchers should pursue
- Any practical barriers (data, compute, evaluation) blocking progress

## 8. References
List every paper as a numbered clickable markdown link:
[1] [Full Paper Title](URL)
[2] [Full Paper Title](URL)
...

Rules:
- Use inline citations [n] throughout sections 2–4 wherever you reference a specific paper
- Every paper title in sections 5 and 8 MUST be a clickable markdown link: [Title](URL)
- Strict Evidence Grounding: Never fabricate labs, authors, benchmarks, or statistics not supported by the provided paper context. If specific details are absent, summarize the verified technical contributions directly without extrapolating.
- Write in analytical, precise academic prose — avoid vague generalities
- Each section must be substantive; do not pad with filler
- Do NOT include any preamble or text before the # heading
- Total length: 1800-2500 words"""


def _build_papers_context(paper_analyses: list[dict]) -> str:
    """Format the paper analyses into a prompt-friendly context block."""
    lines = []
    for i, paper in enumerate(paper_analyses, 1):
        lines.append(f"[{i}] Title: {paper.get('title', 'Untitled')}")
        lines.append(f"    URL: {paper.get('url', 'N/A')}")
        lines.append(f"    Relevance Score: {paper.get('relevance_score', 0):.1f}/10")
        lines.append(f"    Summary: {paper.get('summary', 'N/A')}")
        lines.append(f"    Problem: {paper.get('problem', 'N/A')}")
        lines.append(f"    Methodology: {paper.get('methodology', 'N/A')}")
        lines.append(f"    Findings: {paper.get('findings', 'N/A')}")
        lines.append(f"    Limitations: {paper.get('limitations', 'N/A')}")
        lines.append("")
    return "\n".join(lines)


def run(state: ResearchState) -> ResearchState:
    """
    Report Writer node — synthesizes paper analyses into a Markdown report.

    Args:
        state: Current ResearchState with 'user_query', 'domain', 'paper_analyses'.

    Returns:
        Updated ResearchState with 'report_markdown'.
    """
    user_query = state["user_query"]
    domain = state.get("domain", "research")
    paper_analyses = state["paper_analyses"]

    print(f"[ReportWriter] Writing report for '{user_query}' ({len(paper_analyses)} papers)")

    llm = get_llm(task="heavy")
    papers_context = _build_papers_context(paper_analyses)

    user_message = f"""Research Question: {user_query}
Domain: {domain}

Analyzed Papers ({len(paper_analyses)} total):
{papers_context}

Write a comprehensive 7-section Markdown research report based on these papers. Use inline [n] citations throughout."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    report_markdown = extract_text(response.content)

    print(f"[ReportWriter] Report generated ({len(report_markdown)} chars, "
          f"{len(report_markdown.split())} words)")

    return {**state, "report_markdown": report_markdown}
