"""
Agent 2: Paper Researcher (Tavily — no LLM)

Takes the 5 search queries from the Query Planner and:
  1. Calls Tavily search across academic sources for each query
  2. Merges all results into a flat list
  3. Deduplicates by URL
  4. Drops results with no abstract/snippet content
  5. Returns a clean, capped list of candidate papers

Output is written to ResearchState: clean_papers
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient
from graph.state import ResearchState

load_dotenv()

# Academic sources to bias results toward
ACADEMIC_DOMAINS = [
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "semanticscholar.org",
    "nature.com",
    "science.org",
    "ieee.org",
    "acm.org",
    "springer.com",
    "journals.plos.org",
    "biorxiv.org",
    "medrxiv.org",
    "ssrn.com",
    "ncbi.nlm.nih.gov",
    "dl.acm.org",
]

# Max raw candidates passed to the Analyzer (before top-K filtering)
MAX_CANDIDATES = 30


def run(state: ResearchState) -> ResearchState:
    """
    Paper Researcher node — searches Tavily and returns a clean paper list.

    Args:
        state: Current ResearchState containing 'search_queries'.

    Returns:
        Updated ResearchState with 'clean_papers'.
    """
    search_queries = state["search_queries"]
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY is not set in the environment.")

    client = TavilyClient(api_key=api_key)
    raw_results: list[dict] = []

    for i, query in enumerate(search_queries, 1):
        print(f"[PaperResearcher] Searching ({i}/5): '{query}'")
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=10,
                include_domains=ACADEMIC_DOMAINS,
            )
            results = response.get("results", [])
            print(f"[PaperResearcher]   → {len(results)} results returned")
            raw_results.extend(results)
        except Exception as e:
            print(f"[PaperResearcher]   ⚠ Search failed for query '{query}': {e}")
            continue

    print(f"[PaperResearcher] Total raw results: {len(raw_results)}")

    # ── Deduplication by URL ─────────────────────────────────────────────────
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    for result in raw_results:
        url = result.get("url", "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(result)

    print(f"[PaperResearcher] After dedup: {len(deduped)}")

    # ── Filter: require non-empty content/snippet ────────────────────────────
    clean: list[dict] = []
    for result in deduped:
        content = (result.get("content") or result.get("snippet") or "").strip()
        if len(content) > 80:  # Minimum meaningful abstract length
            clean.append({
                "title": result.get("title", "Untitled").strip(),
                "url": result.get("url", ""),
                "content": content,
                "score": result.get("score", 0.0),
            })

    # ── Cap at MAX_CANDIDATES ────────────────────────────────────────────────
    clean = clean[:MAX_CANDIDATES]
    print(f"[PaperResearcher] Clean papers (passed filters): {len(clean)}")

    return {**state, "clean_papers": clean}
