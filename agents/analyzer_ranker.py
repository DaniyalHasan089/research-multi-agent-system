"""
Agent 3: Analyzer & Ranker (LLM)

Takes the clean paper list from the Paper Researcher and:
  1. For each paper, prompts the LLM to extract structured insights
     (problem, methodology, findings, limitations, relevance_score)
  2. Sorts papers by relevance_score descending
  3. Returns top-K papers

Output is written to ResearchState: paper_analyses

Note: This is the most expensive step. v2 will parallelize these LLM calls.
"""

import json
import os
import re
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import ResearchState
from utils.llm_factory import get_llm, extract_text
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K_PAPERS", "10"))

SYSTEM_PROMPT = """You are an expert academic paper analyst. You will be given the title, URL, and abstract/content of a research paper. Your job is to extract structured insights and rate the paper's relevance to a given research question.

Respond ONLY with a valid JSON object in this exact format:
{
  "problem": "<1-2 sentences: what problem does this paper address?>",
  "methodology": "<1-2 sentences: what methods or approaches does it use?>",
  "findings": "<2-3 sentences: what are the key results or contributions?>",
  "limitations": "<1 sentence: what are the main limitations or gaps?>",
  "relevance_score": <integer from 0 to 10, where 10 is perfectly relevant to the query>
}

Be concise, accurate, and objective. Do not include text outside the JSON."""


def _analyze_paper(llm, paper: dict, user_query: str, index: int, total: int) -> dict | None:
    """Analyze a single paper and return structured insights, or None on failure."""
    title = paper.get("title", "Untitled")
    url = paper.get("url", "")
    content = paper.get("content", "")

    # Truncate very long content to stay within context limits
    max_content_chars = 3000
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
        response = llm.invoke(messages)
        raw = extract_text(response.content)

        # Extract JSON (handles markdown fences)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            print(f"[AnalyzerRanker]   ⚠ Could not parse JSON for '{title}'")
            return None

        analysis = json.loads(json_match.group())
        analysis["title"] = title
        analysis["url"] = url
        # Ensure relevance_score is numeric
        analysis["relevance_score"] = float(analysis.get("relevance_score", 0))
        return analysis

    except Exception as e:
        print(f"[AnalyzerRanker]   ⚠ Failed to analyze '{title}': {e}")
        return None


def run(state: ResearchState) -> ResearchState:
    """
    Analyzer & Ranker node — extracts insights from each paper and returns top-K.

    Args:
        state: Current ResearchState containing 'clean_papers' and 'user_query'.

    Returns:
        Updated ResearchState with 'paper_analyses'.
    """
    clean_papers = state["clean_papers"]
    user_query = state["user_query"]
    total = len(clean_papers)

    print(f"[AnalyzerRanker] Analyzing {total} papers (top-K = {TOP_K})")

    llm = get_llm(task="heavy")
    analyses: list[dict] = []

    for i, paper in enumerate(clean_papers, 1):
        result = _analyze_paper(llm, paper, user_query, i, total)
        if result is not None:
            analyses.append(result)

    # Sort by relevance score descending and take top-K
    analyses.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    top_k = analyses[:TOP_K]

    print(f"[AnalyzerRanker] Successfully analyzed: {len(analyses)}/{total}")
    print(f"[AnalyzerRanker] Returning top-{len(top_k)} papers")
    for i, a in enumerate(top_k, 1):
        print(f"[AnalyzerRanker]   #{i} (score {a['relevance_score']:.1f}): {a['title'][:60]}")

    return {**state, "paper_analyses": top_k}
