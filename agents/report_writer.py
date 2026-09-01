"""
Agent 4: Report Writer (LLM)

Synthesizes all paper analyses into a structured Markdown report.
The report has 7 sections with inline [n] citations referencing the papers.

Output is written to ResearchState: report_markdown
"""

from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import ResearchState
from utils.llm_factory import get_llm, extract_text


SYSTEM_PROMPT = """You are an expert academic research synthesizer. Your task is to write a comprehensive, well-structured research report in Markdown format based on a set of analyzed research papers.

The report MUST contain exactly these 7 sections in order:

# [Research Topic] — Research Report

## 1. Executive Summary
A high-level overview of the research landscape (3-5 sentences). Summarize the state of the field and key takeaways.

## 2. Research Landscape
Describe the broader context: how active is this field, what are the major schools of thought, and how has it evolved?

## 3. Key Themes & Trends
Identify 3-5 recurring themes across the papers. Use bullet points for clarity.

## 4. Paper-by-Paper Breakdown
For each paper, write a short paragraph covering its contribution. Use inline citations like [1], [2], etc. corresponding to the References section. Include the relevance score.

## 5. Methodological Comparison
Compare the approaches used across papers. What methods are most common? What are the trade-offs?

## 6. Gaps & Open Problems
What questions remain unanswered? What are the most promising directions for future research?

## 7. References
Number each paper as [1], [2], etc. Format:
[1] **Title** — URL

Rules:
- Use inline citations [n] throughout sections 2–5 wherever you reference a specific paper
- Be analytical, not just descriptive
- Write in academic but accessible prose
- Do NOT include any preamble or text before the # heading
- Total length: 800-1500 words"""


def _build_papers_context(paper_analyses: list[dict]) -> str:
    """Format the paper analyses into a prompt-friendly context block."""
    lines = []
    for i, paper in enumerate(paper_analyses, 1):
        lines.append(f"[{i}] Title: {paper.get('title', 'Untitled')}")
        lines.append(f"    URL: {paper.get('url', '')}")
        lines.append(f"    Relevance Score: {paper.get('relevance_score', 0):.1f}/10")
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
