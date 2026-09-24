"""
Agent 3: Analyzer & Ranker (LLM)

Takes the clean paper list from the Paper Researcher and:
  1. Deduplicates papers by arxiv_id (primary) then normalized title (fallback)
  2. Filters papers outside the requested year range (papers with unknown year kept)
  3. Concurrently analyzes remaining candidates using asyncio.gather + Semaphore(5)
  4. For each paper, extracts structured insights:
     (summary, problem, methodology, findings, limitations, relevance_score,
      relevance_rationale, speedup_claimed, key_numbers, unsupported_claims)
  5. Soft-fails individual papers
  6. Sorts papers by relevance_score descending (title as stable tie-break)
  7. Returns top-K papers (scores kept internal for ranking only)
  8. Computes search_stats dict for PRISMA-style reporting

Output is written to ResearchState: paper_analyses, search_stats
"""

import asyncio
import json
import os
import re
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import ResearchState
from utils.llm_factory import get_llm, extract_text
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K_PAPERS"))


# ── Deduplication Helpers ────────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for title comparison."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)  # strip punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _dedup_papers(papers: list[dict]) -> list[dict]:
    """
    Deduplicate papers by:
      1. arxiv_id (primary) — same paper from arXiv + SemanticScholar
      2. Normalized title (fallback) — catches identical papers with slightly different URLs
    First occurrence wins (maintains round-robin order from PaperResearcher).
    """
    seen_arxiv_ids: set[str] = set()
    seen_norm_titles: set[str] = set()
    deduped: list[dict] = []

    for paper in papers:
        arxiv_id = (paper.get("arxiv_id") or "").strip()
        norm_title = _normalize_title(paper.get("title") or "")

        # Primary dedup: arxiv_id
        if arxiv_id:
            if arxiv_id in seen_arxiv_ids:
                continue
            seen_arxiv_ids.add(arxiv_id)

        # Fallback dedup: normalized title (skip very short/generic titles)
        if norm_title and len(norm_title) > 10:
            if norm_title in seen_norm_titles:
                continue
            seen_norm_titles.add(norm_title)

        deduped.append(paper)

    return deduped


def _filter_by_year(
    papers: list[dict],
    year_from: int | None,
    year_to: int | None,
) -> list[dict]:
    """
    Remove papers whose year is known and falls outside [year_from, year_to].
    Papers with year=None are always kept (unknown publication year).
    """
    if not year_from and not year_to:
        return papers

    kept = []
    for paper in papers:
        year = paper.get("year")
        if year is None:
            kept.append(paper)  # unknown year — keep
            continue
        if year_from and year < year_from:
            continue
        if year_to and year > year_to:
            continue
        kept.append(paper)
    return kept

SYSTEM_PROMPT = """You are an expert academic paper analyst. You will be given the title, URL, identity metadata (when available), and abstract/content of a research paper. Your job is to extract detailed structured insights and rate the paper's relevance to a given research question.

Respond ONLY with a valid JSON object in this exact format:
{
  "summary": "<4-5 sentence plain-language summary of what this paper is about, its core contributions, and why it matters>",
  "problem": "<2-3 sentences: what specific problem or gap in the literature does this paper address? Include context on why this problem is important.>",
  "methodology": "<2-3 sentences: what specific methods, models, datasets, or experimental setups does it use? Be precise about techniques.>",
  "findings": "<3-4 sentences: what are the key quantitative or qualitative results? Include specific numbers or benchmarks ONLY when they appear in the provided text.>",
  "limitations": "<2 sentences: what are the main limitations, assumptions, or open questions left by this paper?>",
  "relevance_score": <number from 0 to 10, where 10 is perfectly relevant to the query>,
  "relevance_rationale": "<1-2 sentences explaining why this score was assigned, tied to the research question>",
  "speedup_claimed": <number or null — ONLY if an explicit speedup/factor/improvement number appears in the provided text; otherwise null>,
  "key_numbers": ["<strings copied or tightly paraphrased from the provided text>", "..."] or [],
  "unsupported_claims": ["<things you wanted to say but could not ground in the provided text>", "..."] or []
}

Scoring rules:
- Assign differentiated relevance scores; identical 10s across the batch are not allowed.
- Use the full 0–10 scale; do NOT default everything to 10 or pile scores at the top.
- Prefer relative discrimination: similar papers should receive clearly different scores when possible.
- Scores ≥ 9 require a strong rationale explicitly tied to the research question.
- If evidence is thin, score lower and list what you could not ground in unsupported_claims.

Be thorough, accurate, and objective. Use specific details from the paper content. Do not invent numbers, labs, or claims absent from the provided text. Do not include text outside the JSON."""


def _rank_key(analysis: dict) -> tuple:
    """Sort key: higher relevance first; title ascending for stable ties."""
    score = float(analysis.get("relevance_score", 0) or 0)
    title = (analysis.get("title") or "").lower()
    return (-score, title)


async def _analyze_paper_async(
    llm,
    paper: dict,
    user_query: str,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Concurrently analyze a single paper under semaphore control."""
    async with semaphore:
        title = paper.get("title", "Untitled")
        url = paper.get("url", "")
        content = paper.get("content", "")
        arxiv_id = paper.get("arxiv_id") or ""
        doi = paper.get("doi") or ""
        year = paper.get("year")
        venue = paper.get("venue") or ""

        # Truncate content to avoid exceeding context window
        max_content_chars = 5000
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "... [truncated]"

        print(f"[AnalyzerRanker] Analyzing paper {index}/{total}: '{title[:60]}...'")

        identity_lines = []
        if arxiv_id:
            identity_lines.append(f"arXiv ID: {arxiv_id}")
        if doi:
            identity_lines.append(f"DOI: {doi}")
        if year is not None:
            identity_lines.append(f"Year: {year}")
        if venue:
            identity_lines.append(f"Venue: {venue}")
        identity_block = ("\n".join(identity_lines) + "\n") if identity_lines else ""

        user_message = f"""Research question: {user_query}

Paper title: {title}
Paper URL: {url}
{identity_block}
Abstract / Content:
{content}

Please analyze this paper and return your assessment as JSON."""

        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
            response = await llm.ainvoke(messages)
            raw = extract_text(response.content)

            # Extract JSON (handles markdown fences)
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                print(f"[AnalyzerRanker]   Could not parse JSON for '{title}'")
                return None

            analysis = json.loads(json_match.group())
            # Attach identity from input paper — do not invent
            analysis["title"] = title
            analysis["url"] = url
            analysis["arxiv_id"] = arxiv_id
            analysis["doi"] = doi
            analysis["year"] = year
            analysis["venue"] = venue
            analysis["relevance_score"] = float(analysis.get("relevance_score", 0))
            analysis["relevance_rationale"] = analysis.get("relevance_rationale") or ""
            # speedup_claimed: number or null only
            sc = analysis.get("speedup_claimed", None)
            if sc is not None:
                try:
                    analysis["speedup_claimed"] = float(sc)
                except (TypeError, ValueError):
                    analysis["speedup_claimed"] = None
            else:
                analysis["speedup_claimed"] = None
            analysis["key_numbers"] = analysis.get("key_numbers") or []
            if not isinstance(analysis["key_numbers"], list):
                analysis["key_numbers"] = []
            analysis["unsupported_claims"] = analysis.get("unsupported_claims") or []
            if not isinstance(analysis["unsupported_claims"], list):
                analysis["unsupported_claims"] = []
            if not analysis.get("summary"):
                analysis["summary"] = analysis.get("findings", "No summary available.")
            return analysis

        except Exception as e:
            print(f"[AnalyzerRanker]   Failed to analyze '{title}': {e}")
            return None


async def run(state: ResearchState) -> ResearchState:
    """
    Analyzer & Ranker node — deduplicates, year-filters, asynchronously extracts
    insights from each paper under concurrency control, and returns top-K.

    Args:
        state: Current ResearchState containing 'clean_papers', 'user_query',
               and optionally 'year_from'/'year_to'.

    Returns:
        Updated ResearchState with 'paper_analyses' and 'search_stats'.
    """
    clean_papers = state.get("clean_papers", [])
    user_query = state.get("user_query", "")
    year_from = state.get("year_from")
    year_to = state.get("year_to")
    total_retrieved = len(clean_papers)

    # ── Step 1: Deduplicate by arxiv_id then normalized title ────────────────
    deduped_papers = _dedup_papers(clean_papers)
    total_after_dedup = len(deduped_papers)
    removed_by_dedup = total_retrieved - total_after_dedup
    if removed_by_dedup:
        print(f"[AnalyzerRanker] Deduplication removed {removed_by_dedup} duplicate(s) "
              f"({total_retrieved} → {total_after_dedup})")
    else:
        print(f"[AnalyzerRanker] Deduplication: no duplicates found ({total_retrieved} papers)")

    # ── Step 2: Year range filter ────────────────────────────────────────────
    filtered_papers = _filter_by_year(deduped_papers, year_from, year_to)
    total_after_year_filter = len(filtered_papers)
    removed_by_year = total_after_dedup - total_after_year_filter
    if (year_from or year_to) and removed_by_year:
        print(f"[AnalyzerRanker] Year filter ({year_from or 'any'}–{year_to or 'any'}) "
              f"removed {removed_by_year} paper(s) ({total_after_dedup} → {total_after_year_filter})")
    elif year_from or year_to:
        print(f"[AnalyzerRanker] Year filter applied — all papers within range")

    total = len(filtered_papers)
    print(f"[AnalyzerRanker] Concurrently analyzing {total} papers (top-K = {TOP_K}, concurrency = 5)")

    llm = get_llm(task="heavy")
    semaphore = asyncio.Semaphore(5)

    tasks = [
        _analyze_paper_async(llm, paper, user_query, i, total, semaphore)
        for i, paper in enumerate(filtered_papers, 1)
    ]

    results = await asyncio.gather(*tasks)
    analyses = [r for r in results if r is not None]

    # Sort by model relevance_score desc (title tie-break); take top-K as-is.
    # Scores stay internal for ranking only — not surfaced in the user report.
    analyses.sort(key=_rank_key)
    top_k = analyses[:TOP_K]

    print(f"[AnalyzerRanker] Successfully analyzed: {len(analyses)}/{total}")
    print(f"[AnalyzerRanker] Returning top-{len(top_k)} papers")
    for i, a in enumerate(top_k, 1):
        print(
            f"[AnalyzerRanker]   #{i} (score {a['relevance_score']:.1f}): "
            f"{a['title'][:60]}"
        )

    # ── Compute PRISMA-style pipeline stats for the report ───────────────────
    search_stats = {
        "total_retrieved": total_retrieved,
        "total_after_dedup": total_after_dedup,
        "duplicates_removed": removed_by_dedup,
        "total_after_year_filter": total_after_year_filter,
        "excluded_by_year": removed_by_year,
        "total_analyzed": len(analyses),
        "total_failed_analysis": total - len(analyses),
        "total_included": len(top_k),
        "year_from": year_from,
        "year_to": year_to,
        "top_k": TOP_K,
    }

    return {**state, "paper_analyses": top_k, "search_stats": search_stats}

