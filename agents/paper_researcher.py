"""
Agent 2: Paper Researcher (Tavily — no LLM)

Takes the 5 search queries and domain classification from the Query Planner and:
  1. Selects a tight, domain-specific academic allowlist (5-6 domains)
  2. Calls Tavily search for each query
  3. Pre-filters low-content/missing snippets (>80 chars)
  4. Merges candidates using round-robin interleaving across all 5 queries
     (ensuring Query 5 is fairly represented alongside Query 1)
  5. Deduplicates by URL and caps at MAX_CANDIDATES

Output is written to ResearchState: clean_papers
"""

import os
from dotenv import load_dotenv
from tavily import TavilyClient
from graph.state import ResearchState

load_dotenv()

# Curated, tight domain allowlists (5-6 high-authority sources max per domain)
# to avoid search dilution in Tavily.
DOMAIN_ALLOWLISTS = {
    "cs": [
        "arxiv.org",
        "semanticscholar.org",
        "ieee.org",
        "dl.acm.org",
        "paperswithcode.com",
    ],
    "medicine": [
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "biorxiv.org",
        "medrxiv.org",
        "nature.com",
    ],
    "biology": [
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "biorxiv.org",
        "nature.com",
        "science.org",
    ],
    "physics": [
        "arxiv.org",
        "nature.com",
        "science.org",
        "journals.aps.org",
        "sciencedirect.com",
    ],
    "economics": [
        "ssrn.com",
        "nber.org",
        "ideas.repec.org",
        "sciencedirect.com",
        "tandfonline.com",
    ],
    "default": [
        "arxiv.org",
        "semanticscholar.org",
        "nature.com",
        "science.org",
        "sciencedirect.com",
    ],
}

# Max raw candidates passed to the Analyzer (balanced across queries)
MAX_CANDIDATES = 30


def _get_allowlist_for_domain(domain: str) -> list[str]:
    """Return a tight, domain-specific allowlist for Tavily search."""
    d = (domain or "").lower().strip()
    if any(k in d for k in ["cs", "computer", "ai", "machine learning", "engineering", "software"]):
        return DOMAIN_ALLOWLISTS["cs"]
    elif any(k in d for k in ["med", "health", "clinical"]):
        return DOMAIN_ALLOWLISTS["medicine"]
    elif any(k in d for k in ["bio", "genomic"]):
        return DOMAIN_ALLOWLISTS["biology"]
    elif any(k in d for k in ["phys", "math", "astro", "quantum"]):
        return DOMAIN_ALLOWLISTS["physics"]
    elif any(k in d for k in ["econ", "finance", "business"]):
        return DOMAIN_ALLOWLISTS["economics"]
    return DOMAIN_ALLOWLISTS["default"]


def run(state: ResearchState) -> ResearchState:
    """
    Paper Researcher node — searches Tavily with domain allowlist and returns
    a balanced candidate list via round-robin query interleaving.

    Args:
        state: Current ResearchState containing 'search_queries' and 'domain'.

    Returns:
        Updated ResearchState with 'clean_papers'.
    """
    search_queries = state["search_queries"]
    domain = state.get("domain", "other")
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY is not set in the environment.")

    client = TavilyClient(api_key=api_key)
    allowlist = _get_allowlist_for_domain(domain)
    print(f"[PaperResearcher] Domain '{domain}' mapped to allowlist: {allowlist}")

    # Collect filtered candidates per query: list of lists
    query_candidates: list[list[dict]] = []

    for i, query in enumerate(search_queries, 1):
        print(f"[PaperResearcher] Searching ({i}/5): '{query}'")
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=10,
                include_domains=allowlist,
            )
            raw_results = response.get("results", [])
            print(f"[PaperResearcher]   → {len(raw_results)} results returned")

            filtered = []
            for r in raw_results:
                content = (r.get("content") or r.get("snippet") or "").strip()
                url = r.get("url", "").strip()
                if len(content) > 80 and url:
                    filtered.append({
                        "title": r.get("title", "Untitled").strip(),
                        "url": url,
                        "content": content,
                        "score": float(r.get("score", 0.0)),
                    })
            query_candidates.append(filtered)

        except Exception as e:
            print(f"[PaperResearcher]   ⚠ Search failed for query '{query}': {e}")
            query_candidates.append([])

    # ── Round-Robin Interleaved Selection across all 5 queries ───────────────
    # Guarantees Query 5 has equal opportunity to contribute as Query 1
    clean_papers: list[dict] = []
    seen_urls: set[str] = set()

    max_depth = max((len(papers) for papers in query_candidates), default=0)
    for rank in range(max_depth):
        for q_idx, papers in enumerate(query_candidates):
            if rank < len(papers):
                candidate = papers[rank]
                if candidate["url"] not in seen_urls:
                    seen_urls.add(candidate["url"])
                    clean_papers.append(candidate)
                    if len(clean_papers) >= MAX_CANDIDATES:
                        break
        if len(clean_papers) >= MAX_CANDIDATES:
            break

    print(f"[PaperResearcher] Round-robin selection complete: {len(clean_papers)} papers")
    return {**state, "clean_papers": clean_papers}
