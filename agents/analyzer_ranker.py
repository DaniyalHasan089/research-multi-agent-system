"""
Agent 3: Analyzer & Ranker (LLM)

Takes the clean paper list from the Paper Researcher and:
  1. Concurrently analyzes candidate papers using asyncio.gather + Semaphore(5)
  2. For each paper, extracts structured insights:
     (summary, problem, methodology, findings, limitations, relevance_score)
  3. Sorts papers by relevance_score descending
  4. Returns top-K papers

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

SYSTEM_PROMPT = """You are an expert academic paper analyst. You will be given the title, URL, and abstract/content of a research paper. Your job is to extract detailed structured insights and rate the paper's relevance to a given research question.

Respond ONLY with a valid JSON object in this exact format:
{
  "summary": "<4-5 sentence plain-language summary of what this paper is about, its core contributions, and why it matters>",
  "problem": "<2-3 sentences: what specific problem or gap in the literature does this paper address? Include context on why this problem is important.>",
  "methodology": "<2-3 sentences: what specific methods, models, datasets, or experimental setups does it use? Be precise about techniques.>",
  "findings": "<3-4 sentences: what are the key quantitative or qualitative results? Include specific numbers or benchmarks where available.>",
  "limitations": "<2 sentences: what are the main limitations, assumptions, or open questions left by this paper?>",
  "relevance_score": <integer from 0 to 10, where 10 is perfectly relevant to the query>
}

Be thorough, accurate, and objective. Use specific details from the paper content. Do not include text outside the JSON."""


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

        # Truncate content to avoid exceeding context window
        max_content_chars = 5000
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "... [truncated]"

        print(f"[AnalyzerRanker] Analyzing paper {index}/{total}: '{title[:60]}...'")

        user_message = f"""Research question: {user_query}

Paper title: {title}
Paper URL: {url}

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
                print(f"[AnalyzerRanker]   ⚠ Could not parse JSON for '{title}'")
                return None

            analysis = json.loads(json_match.group())
            analysis["title"] = title
            analysis["url"] = url
            analysis["relevance_score"] = float(analysis.get("relevance_score", 0))
            if not analysis.get("summary"):
                analysis["summary"] = analysis.get("findings", "No summary available.")
            return analysis

        except Exception as e:
            print(f"[AnalyzerRanker]   ⚠ Failed to analyze '{title}': {e}")
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

    # Sort by relevance score descending and take top-K
    analyses.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    top_k = analyses[:TOP_K]

    print(f"[AnalyzerRanker] Successfully analyzed: {len(analyses)}/{total}")
    print(f"[AnalyzerRanker] Returning top-{len(top_k)} papers")
    for i, a in enumerate(top_k, 1):
        print(f"[AnalyzerRanker]   #{i} (score {a['relevance_score']:.1f}): {a['title'][:60]}")

    return {**state, "paper_analyses": top_k}
