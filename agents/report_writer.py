"""
Agent 4: Report Writer (LLM)

Synthesizes all paper analyses into a structured Markdown report.
The report has 8 sections with inline [n] citations referencing the papers.

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
- Major milestones or paradigm shifts in the field (only when grounded in the provided analyses)
- The current state: how active is this area based on the papers reviewed?
- Key sub-fields or competing schools of thought, with citations [n]
- Soften institutional claims: do NOT invent or fill with unnamed "industry labs" or "academic groups". Only name organizations, labs, or institutions if they appear explicitly in the provided paper analyses. Otherwise describe the landscape in terms of methods, problems, and cited papers [n].

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

**Key Findings:** 2-3 sentences: what are the most important results, including specific numbers or benchmarks where available and grounded?

**Limitations:** 1-2 sentences: what does this paper leave unresolved or what are its assumptions?


## 6. Methodological Comparison
Write 2-3 paragraphs comparing approaches across papers:
- What methods or frameworks are most commonly used and why?
- Where do papers disagree on methodology, and what are the trade-offs?
- What methodological gaps exist (e.g., missing baselines, untested settings)?

## 7. Research Gaps & Open Problems
Topic-level synthesis across the corpus — NOT a repeat of each paper's Limitations (those stay in section 5). Use this fixed skeleton:

### 7.1 What's missing in the literature
Identify 3-5 concrete gaps. Each gap MUST include:
(a) one-sentence claim,
(b) citations [n] showing the corpus supports that this is open/missing,
(c) what specifically is absent (benchmark, setting, comparison, theory, modality, etc.).
Prefer formulations like "no paper in this set evaluates X under Y" over grand claims.

### 7.2 Why it matters
For each gap or grouped gaps: state practical stakes (deployment, safety, sample efficiency, generalization, eval, etc.) grounded in the analyses — no vague "important for AGI" filler.

### 7.3 Promising directions
Give 2-4 next-step research angles that follow from those gaps, still grounded in the corpus. Do not invent labs, datasets, or papers not in the analyses.

### 7.4 Barriers
List practical blockers shared across papers (data, compute, eval protocol, sim-to-real, etc.) — only if supported by the paper analyses, with [n].

Rules for this section:
- Every gap needs at least one [n]
- No filler: "more research is needed", "further investigation", unnamed industry labs
- This is topic-level synthesis, not a per-paper Limitations dump

## 8. References
List every paper as a numbered clickable markdown link. Prefer this format:
[1] [Clean Title](URL) (arXiv:ID; DOI:…)
Include arXiv ID and/or DOI in parentheses after the link when present in the analysis metadata. Omit empty identity fields.
[2] [Clean Title](URL)
...

Rules:
- Use inline citations [n] throughout sections 2–7 wherever you reference a specific paper
- Every paper title in sections 5 and 8 MUST be a clickable markdown link: [Title](URL) using the analysis title and url
- Quantitative grounding: Every quantitative claim (speedups, accuracies, dataset sizes, percentages, etc.) MUST have an inline [n] citation AND must appear in that paper's analysis fields `key_numbers`, `speedup_claimed`, or `findings`. If a number is not grounded there, omit it or explicitly say that quantitative evidence was not extracted from the provided analysis.
- Do not invent lab, institution, university, or company names
- Strict Evidence Grounding: Never fabricate labs, authors, benchmarks, or statistics not supported by the provided paper context. If specific details are absent, summarize the verified technical contributions directly without extrapolating.
- Write in analytical, precise academic prose — avoid vague generalities and unnamed "industry/academic" filler
- Each section must be substantive; do not pad with filler
- Do NOT include any preamble or text before the # heading
- Total length: 1800-2500 words"""


def _build_papers_context(paper_analyses: list[dict]) -> str:
    """Format the paper analyses into a prompt-friendly context block."""
    lines = []
    for i, paper in enumerate(paper_analyses, 1):
        lines.append(f"[{i}] Title: {paper.get('title', 'Untitled')}")
        lines.append(f"    URL: {paper.get('url', 'N/A')}")
        if paper.get("arxiv_id"):
            lines.append(f"    arXiv ID: {paper['arxiv_id']}")
        if paper.get("doi"):
            lines.append(f"    DOI: {paper['doi']}")
        if paper.get("year") is not None:
            lines.append(f"    Year: {paper['year']}")
        if paper.get("venue"):
            lines.append(f"    Venue: {paper['venue']}")
        lines.append(f"    Summary: {paper.get('summary', 'N/A')}")
        lines.append(f"    Problem: {paper.get('problem', 'N/A')}")
        lines.append(f"    Methodology: {paper.get('methodology', 'N/A')}")
        lines.append(f"    Findings: {paper.get('findings', 'N/A')}")
        lines.append(f"    Limitations: {paper.get('limitations', 'N/A')}")
        if paper.get("speedup_claimed") is not None:
            lines.append(f"    Speedup Claimed: {paper['speedup_claimed']}")
        key_numbers = paper.get("key_numbers") or []
        if key_numbers:
            lines.append(f"    Key Numbers: {key_numbers}")
        unsupported = paper.get("unsupported_claims") or []
        if unsupported:
            lines.append(f"    Unsupported Claims (do not use as facts): {unsupported}")
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

Write a comprehensive 8-section Markdown research report based on these papers (section 7 must be Research Gaps & Open Problems with the 7.1–7.4 skeleton). Use inline [n] citations throughout. Ground every quantitative claim in key_numbers / speedup_claimed / findings for the cited paper; otherwise omit or note that evidence was not extracted. Do not invent lab or institution names."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    report_markdown = extract_text(response.content)

    print(f"[ReportWriter] Report generated ({len(report_markdown)} chars, "
          f"{len(report_markdown.split())} words)")

    return {**state, "report_markdown": report_markdown}
