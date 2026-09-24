"""
Agent 4: Report Writer (LLM)

Synthesizes all paper analyses into a structured Markdown literature review.
The report has 10 sections with inline [n] citations referencing the papers.

Sections 2 (Search Strategy) and 3 (Inclusion Criteria) use PRISMA-style
pipeline statistics computed by the Analyzer & Ranker.

Output is written to ResearchState: report_markdown
"""

from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import ResearchState
from utils.llm_factory import get_llm, extract_text


SYSTEM_PROMPT = """You are an expert academic literature review writer. Your task is to write a detailed, comprehensive literature review in Markdown format based on a set of analyzed research papers and the provided search statistics.

The report MUST contain exactly these 10 sections in order:

# [Research Topic] — Literature Review

## 1. Executive Summary
Write 5-7 sentences covering: the research question, why this topic matters, how many papers were reviewed, the dominant findings across the corpus, and the most important open challenge. This should give a reader a complete picture of the field at a glance.

## 2. Search Strategy & Methodology
Using the SEARCH STATISTICS block provided below, write 2-3 sentences documenting:
- The academic sources and domain searched
- The exact pipeline counts: papers retrieved → after deduplication → after year filtering → successfully analyzed → final included
- Any date range constraints applied
Use the EXACT numbers from the SEARCH STATISTICS block — do not invent or round them. Present this as a methodological note (PRISMA-style).

## 3. Inclusion Criteria
Write 3-4 sentences documenting:
- Papers were selected based on AI-assessed relevance scoring (0–10) against the research question
- Only the top-K highest scoring papers entered the final synthesis
- Any year range constraint applied to scope the literature temporally
- What types of sources were targeted (preprints, peer-reviewed venues, etc.)

## 4. Background & Context
Write 3-4 paragraphs covering:
- Historical context: how did this research area emerge and evolve?
- Major milestones or paradigm shifts in the field (only when grounded in the provided analyses)
- The current state: how active is this area based on the papers reviewed?
- Key sub-fields or competing schools of thought, with citations [n]
- Soften institutional claims: do NOT invent or fill with unnamed "industry labs" or "academic groups". Only name organizations, labs, or institutions if they appear explicitly in the provided paper analyses. Otherwise describe the landscape in terms of methods, problems, and cited papers [n].

## 5. Thematic Synthesis
Identify and elaborate on 4-6 recurring themes across the papers. For each theme:
- Give it a descriptive heading
- Write 2-3 sentences explaining the theme and cite the papers [n] that exemplify it
- Note whether the theme is emerging, established, or contested

## 6. Current & Ongoing Work
Describe what researchers are actively working on RIGHT NOW based on the papers reviewed:
- Highlight active research directions, methodologies, and benchmarks present in the papers [n]
- Identify architectures and techniques currently being explored, refined, or compared
- Reference recent papers and preprints that signal where the field is heading [n]
- Strict Grounding Rule: ONLY reference institutions, labs, or benchmarks that are EXPLICITLY cited in the provided paper analyses. If specific lab or university affiliations are not present in the sources, do NOT invent or assume them; describe the ongoing research purely in terms of the verified technical methods and problem spaces.

## 7. Paper-by-Paper Breakdown
For EVERY paper, write a dedicated subsection in this exact format:

### [n] [Full Paper Title](URL)
**Summary:** 4-5 sentences: what is this paper about, what problem does it solve, and why does it matter?

**Methodology:** 2-3 sentences: what specific techniques, datasets, or experimental approach does it use?

**Key Findings:** 2-3 sentences: what are the most important results, including specific numbers or benchmarks where available and grounded?

**Limitations:** 1-2 sentences: what does this paper leave unresolved or what are its assumptions?


## 8. Methodological Comparison
Write 2-3 paragraphs comparing approaches across papers:
- What methods or frameworks are most commonly used and why?
- Where do papers disagree on methodology, and what are the trade-offs?
- What methodological gaps exist (e.g., missing baselines, untested settings)?

## 9. Gaps & Future Directions
Topic-level synthesis across the corpus — NOT a repeat of each paper's Limitations (those stay in section 7). Use this fixed skeleton:

### 9.1 What's missing in the literature
Identify 3-5 concrete gaps. Each gap MUST include:
(a) one-sentence claim,
(b) citations [n] showing the corpus supports that this is open/missing,
(c) what specifically is absent (benchmark, setting, comparison, theory, modality, etc.).
Prefer formulations like "no paper in this set evaluates X under Y" over grand claims.

### 9.2 Why it matters
For each gap or grouped gaps: state practical stakes (deployment, safety, sample efficiency, generalization, eval, etc.) grounded in the analyses — no vague "important for AGI" filler.

### 9.3 Promising directions
Give 2-4 next-step research angles that follow from those gaps, still grounded in the corpus. Do not invent labs, datasets, or papers not in the analyses.

### 9.4 Barriers
List practical blockers shared across papers (data, compute, eval protocol, sim-to-real, etc.) — only if supported by the paper analyses, with [n].

Rules for this section:
- Every gap needs at least one [n]
- No filler: "more research is needed", "further investigation", unnamed industry labs
- This is topic-level synthesis, not a per-paper Limitations dump

## 10. References
List every paper as a numbered clickable markdown link. Prefer this format:
[1] [Clean Title](URL) (arXiv:ID; DOI:...)
Include arXiv ID and/or DOI in parentheses after the link when present in the analysis metadata. Omit empty identity fields.
[2] [Clean Title](URL)
...

Rules:
- Use inline citations [n] throughout sections 4-9 wherever you reference a specific paper
- Every paper title in sections 7 and 10 MUST be a clickable markdown link: [Title](URL) using the analysis title and url
- Quantitative grounding: Every quantitative claim (speedups, accuracies, dataset sizes, percentages, etc.) MUST have an inline [n] citation AND must appear in that paper's analysis fields `key_numbers`, `speedup_claimed`, or `findings`. If a number is not grounded there, omit it or explicitly say that quantitative evidence was not extracted from the provided analysis.
- Do not invent lab, institution, university, or company names
- Strict Evidence Grounding: Never fabricate labs, authors, benchmarks, or statistics not supported by the provided paper context. If specific details are absent, summarize the verified technical contributions directly without extrapolating.
- Write in analytical, precise academic prose — avoid vague generalities and unnamed "industry/academic" filler
- Each section must be substantive; do not pad with filler
- Do NOT include any preamble or text before the # heading
- Total length: 2200-3000 words"""


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


def _build_stats_block(search_stats: dict, search_queries: list[str], domain: str) -> str:
    """Format search_stats into a readable PRISMA-style block for the LLM."""
    year_from = search_stats.get("year_from")
    year_to = search_stats.get("year_to")
    date_range = (
        f"{year_from or 'any'} - {year_to or 'any'}"
        if (year_from or year_to)
        else "No date restriction applied"
    )

    duplicates = search_stats.get("duplicates_removed", 0)
    excluded_year = search_stats.get("excluded_by_year", 0)
    failed = search_stats.get("total_failed_analysis", 0)

    lines = [
        "=== SEARCH STATISTICS (use verbatim in Section 2) ===",
        f"Domain classified        : {domain.upper()}",
        f"Search queries issued    : {len(search_queries)}",
        f"Date range filter        : {date_range}",
        f"Papers retrieved (raw)   : {search_stats.get('total_retrieved', 'N/A')}",
        f"After deduplication      : {search_stats.get('total_after_dedup', 'N/A')} "
        f"({duplicates} duplicate(s) removed)",
        f"After year filter        : {search_stats.get('total_after_year_filter', 'N/A')} "
        f"({excluded_year} excluded outside date range)",
        f"Successfully analyzed    : {search_stats.get('total_analyzed', 'N/A')} "
        f"({failed} failed LLM analysis)",
        f"Final included in review : {search_stats.get('total_included', 'N/A')} "
        f"(top-{search_stats.get('top_k', 'N/A')} by relevance score)",
        "",
        "Search queries used:",
    ]
    for i, q in enumerate(search_queries, 1):
        lines.append(f"  {i}. {q}")
    lines.append("=== END SEARCH STATISTICS ===")
    return "\n".join(lines)


def run(state: ResearchState) -> ResearchState:
    """
    Report Writer node -- synthesizes paper analyses into a Markdown literature review.

    Args:
        state: Current ResearchState with 'user_query', 'domain', 'paper_analyses',
               'search_stats', and 'search_queries'.

    Returns:
        Updated ResearchState with 'report_markdown'.
    """
    user_query = state["user_query"]
    domain = state.get("domain", "research")
    paper_analyses = state["paper_analyses"]
    search_stats = state.get("search_stats", {})
    search_queries = state.get("search_queries", [])

    print(f"[ReportWriter] Writing literature review for '{user_query}' ({len(paper_analyses)} papers)")

    llm = get_llm(task="heavy")
    papers_context = _build_papers_context(paper_analyses)
    stats_block = _build_stats_block(search_stats, search_queries, domain)

    user_message = f"""Research Question: {user_query}
Domain: {domain}

{stats_block}

Analyzed Papers ({len(paper_analyses)} total):
{papers_context}

Write a comprehensive 10-section Markdown literature review. Section 2 must use the exact numbers from the SEARCH STATISTICS block above. Section 9 must use the Gaps & Future Directions skeleton (9.1-9.4). Use inline [n] citations throughout. Ground every quantitative claim in key_numbers / speedup_claimed / findings for the cited paper; otherwise omit or note that evidence was not extracted. Do not invent lab or institution names."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    report_markdown = extract_text(response.content)

    print(f"[ReportWriter] Literature review generated ({len(report_markdown)} chars, "
          f"{len(report_markdown.split())} words)")

    return {**state, "report_markdown": report_markdown}
