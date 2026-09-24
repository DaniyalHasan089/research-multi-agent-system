"""
Agent 2: Paper Researcher (Tavily — no LLM)

Takes the 5 search queries and domain classification from the Query Planner and:
  1. Selects a tight, domain-specific academic allowlist (5-6 domains)
  2. Calls Tavily search for each query
  3. Pre-filters low-content/missing snippets (>80 chars)
  4. Merges candidates using round-robin interleaving across all 5 queries
     (ensuring Query 5 is fairly represented alongside Query 1)
  5. Deduplicates by URL and caps at MAX_CANDIDATES
  6. Resolves canonical titles from the arXiv API when arxiv_id is present

Output is written to ResearchState: clean_papers
"""

import os
import re
import time
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from tavily import TavilyClient
from graph.state import ResearchState

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is in requirements.txt
    httpx = None

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

_ARXIV_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_ARXIV_BATCH_SIZE = 20
_ARXIV_TIMEOUT_S = 8.0
_ARXIV_BATCH_SLEEP_S = 0.2


def normalize_title(title: str) -> str:
    """Strip common title chrome and collapse whitespace."""
    t = (title or "").strip()
    if not t:
        return "Untitled"

    # Strip leading "Figure N from " (case insensitive)
    t = re.sub(r"^Figure\s+\d+\s+from\s+", "", t, flags=re.IGNORECASE)

    # Strip trailing " | Semantic Scholar" / " | Proceedings of..." chrome
    t = re.sub(r"\s*\|\s*Semantic Scholar\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\|\s*Proceedings of.*$", "", t, flags=re.IGNORECASE)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Optional: fix missing spaces only where safe (lowercaseLetterUppercase)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)

    return t or "Untitled"


from utils.arxiv_utils import extract_arxiv_id, batch_fetch_arxiv_metadata, format_authors

_DOMAIN_VENUE_MAP = {
    "arxiv.org": "arXiv Preprint",
    "nature.com": "Nature",
    "science.org": "Science",
    "ieee.org": "IEEE Xplore",
    "dl.acm.org": "ACM Digital Library",
    "acm.org": "ACM",
    "biorxiv.org": "bioRxiv",
    "medrxiv.org": "medRxiv",
    "sciencedirect.com": "ScienceDirect",
    "pubmed.ncbi.nlm.nih.gov": "PubMed",
    "ncbi.nlm.nih.gov": "NCBI",
    "semanticscholar.org": "Semantic Scholar",
    "paperswithcode.com": "Papers With Code",
    "ssrn.com": "SSRN",
    "nber.org": "NBER",
    "springer.com": "Springer",
    "wiley.com": "Wiley",
    "cell.com": "Cell Press",
    "thelancet.com": "The Lancet",
    "frontiersin.org": "Frontiers",
    "mdpi.com": "MDPI",
    "plos.org": "PLOS",
    "pnas.org": "PNAS",
}


def extract_identity(url: str, title: str, content: str = "") -> dict:
    """Extract arxiv_id / doi / canonical_url / year / venue / author from URL, title, and content."""
    url = (url or "").strip()
    arxiv_id = extract_arxiv_id(url) or extract_arxiv_id(title) or extract_arxiv_id(content[:400])
    doi = None
    year = None
    venue = None
    author = None

    if arxiv_id:
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        canonical_url = f"https://arxiv.org/abs/{base_id}"
        # Extract publication year from modern arXiv ID (YYMM)
        ym_match = re.match(r"^([0-9]{2})[0-9]{2}", base_id)
        if ym_match:
            yy = int(ym_match.group(1))
            year = 2000 + yy if yy < 50 else 1900 + yy
        venue = f"arXiv:{base_id}"
    else:
        canonical_url = url

    # DOI: doi.org/ or dx.doi.org/
    doi_match = re.search(
        r"(?:dx\.)?doi\.org/(10\.\S+)",
        url,
        flags=re.IGNORECASE,
    )
    if doi_match:
        doi = doi_match.group(1).rstrip("/")
        if not arxiv_id:
            canonical_url = f"https://doi.org/{doi}"

    # Extract year fallback from URL or title
    if year is None:
        y_url = re.search(r"/(?:20|19)(\d{2})[/-]|(?:20|19)(\d{2})", url)
        if y_url:
            matched_year = int(f"20{y_url.group(1) or y_url.group(2)}")
            if 1990 <= matched_year <= 2030:
                year = matched_year
    if year is None and title:
        y_title = re.search(r"\b(20[12]\d)\b", title)
        if y_title:
            year = int(y_title.group(1))

    # Extract venue fallback from domain
    if not venue:
        for domain, domain_venue in _DOMAIN_VENUE_MAP.items():
            if domain in url.lower():
                venue = domain_venue
                break

    # Extract author hint from content if explicitly formatted
    if content:
        auth_match = re.search(
            r"(?:authors?|by)\s*:\s*([A-Z][a-zA-Z\s,.-]+?)(?:\.|\n|;|\band\b|Abstract)",
            content[:600],
            flags=re.IGNORECASE,
        )
        if auth_match:
            cand = auth_match.group(1).strip()
            if 3 < len(cand) < 60 and not any(bad in cand.lower() for bad in ["abstract", "introduction", "download", "pdf", "table", "figure"]):
                author = cand

    return {
        "arxiv_id": arxiv_id,
        "doi": doi,
        "canonical_url": canonical_url or url,
        "year": year,
        "venue": venue,
        "author": author,
    }


def resolve_arxiv_metadata(papers: list[dict]) -> list[dict]:
    """
    Upgrade title, author(s), year, and venue from arXiv API / HTML when arxiv_id is present.
    Ensures at least 2 authors are named when multiple authors exist.
    """
    if not papers:
        return papers

    ids: list[str] = []
    seen: set[str] = set()
    for p in papers:
        aid = (p.get("arxiv_id") or "").strip()
        if aid and aid not in seen:
            seen.add(aid)
            ids.append(aid)

    if not ids:
        return papers

    meta_map = batch_fetch_arxiv_metadata(ids)

    upgraded = 0
    for p in papers:
        aid = (p.get("arxiv_id") or "").strip()
        if not aid:
            continue
        base = re.sub(r"v\d+$", "", aid)
        meta = meta_map.get(aid) or meta_map.get(base)
        if not meta:
            continue

        if meta.get("title"):
            p["title"] = meta["title"]
        if meta.get("author"):
            p["author"] = meta["author"]
        if meta.get("year") and not p.get("year"):
            p["year"] = meta["year"]
        if meta.get("venue") and (not p.get("venue") or p["venue"].lower() in ["unknown", "n/a", "none"]):
            p["venue"] = meta["venue"]
        upgraded += 1

    print(
        f"[PaperResearcher] arXiv metadata resolve: "
        f"{upgraded}/{len(ids)} papers enriched with canonical author & metadata"
    )
    return papers



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
                    raw_title = r.get("title", "Untitled").strip()
                    clean_title = normalize_title(raw_title)
                    identity = extract_identity(url, clean_title, content)
                    canonical = identity.get("canonical_url") or url
                    filtered.append({
                        "title": clean_title,
                        "url": canonical,
                        "content": content,
                        "score": float(r.get("score", 0.0)),
                        "arxiv_id": identity.get("arxiv_id") or "",
                        "doi": identity.get("doi") or "",
                        "year": identity.get("year"),
                        "venue": identity.get("venue") or "",
                        "author": identity.get("author") or "",
                    })

            # Prefer higher-scoring hits within each query before round-robin
            filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            query_candidates.append(filtered)

        except Exception as e:
            print(f"[PaperResearcher]   ✗ Search failed for query '{query}': {e}")
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

    # After final clean_papers: enrich with canonical arXiv metadata when possible
    clean_papers = resolve_arxiv_metadata(clean_papers)

    return {**state, "clean_papers": clean_papers}
