"""
Agent 3: Analyzer & Ranker (LLM)

Takes the clean paper list from the Paper Researcher and:
  1. Concurrently analyzes candidate papers using asyncio.gather + Semaphore(5)
  2. For each paper, extracts structured insights:
     (summary, problem, methodology, findings, limitations, relevance_score,
      relevance_rationale, speedup_claimed, key_numbers, unsupported_claims)
  3. Soft-fails individual papers
  4. Sorts papers by relevance_score descending (title as stable tie-break)
  5. Returns top-K papers (scores kept internal for ranking only)

Output is written to ResearchState: paper_analyses
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
    Analyzer & Ranker node — asynchronously extracts insights from each paper
    under concurrency control and returns top-K.

    Args:
        state: Current ResearchState containing 'clean_papers' and 'user_query'.

    Returns:
        Updated ResearchState with 'paper_analyses'.
    """
    clean_papers = state.get("clean_papers", [])
    user_query = state.get("user_query", "")
    total = len(clean_papers)

    print(f"[AnalyzerRanker] Concurrently analyzing {total} papers (top-K = {TOP_K}, concurrency = 5)")

    llm = get_llm(task="heavy")
    semaphore = asyncio.Semaphore(5)

    tasks = [
        _analyze_paper_async(llm, paper, user_query, i, total, semaphore)
        for i, paper in enumerate(clean_papers, 1)
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

    return {**state, "paper_analyses": top_k}
