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


def extract_identity(url: str, title: str) -> dict:
    """Extract arxiv_id / doi / canonical_url from a paper URL (and title if needed)."""
    url = (url or "").strip()
    arxiv_id = None
    doi = None
    year = None
    venue = None

    # arXiv: /abs/ID or /pdf/ID.pdf
    arxiv_match = re.search(
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        url,
        flags=re.IGNORECASE,
    )
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        # Prefer versionless abs URL as canonical
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        canonical_url = f"https://arxiv.org/abs/{base_id}"
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
        # Prefer doi.org canonical if we have no arxiv canonical
        if not arxiv_id:
            canonical_url = f"https://doi.org/{doi}"

    # Year/venue left as None unless trivially parseable — do not invent
    return {
        "arxiv_id": arxiv_id,
        "doi": doi,
        "canonical_url": canonical_url or url,
        "year": year,
        "venue": venue,
    }


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _parse_arxiv_atom_titles(xml_text: str) -> dict[str, str]:
    """Parse arXiv Atom API XML into {arxiv_id: title} for each entry."""
    mapping: dict[str, str] = {}
    if not xml_text:
        return mapping

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return mapping

    for entry in root.findall("atom:entry", _ARXIV_ATOM_NS):
        title_el = entry.find("atom:title", _ARXIV_ATOM_NS)
        id_el = entry.find("atom:id", _ARXIV_ATOM_NS)
        if title_el is None or id_el is None or not (id_el.text or "").strip():
            continue

        # id looks like http://arxiv.org/abs/1234.5678v1
        id_text = id_el.text.strip()
        m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", id_text, re.I)
        if not m:
            continue
        aid = m.group(1)
        title = _collapse_ws("".join(title_el.itertext()))
        if title:
            mapping[aid] = title
            # Also index versionless form for easier lookup
            base = re.sub(r"v\d+$", "", aid)
            mapping[base] = title

    return mapping


def _fetch_arxiv_titles_batch(arxiv_ids: list[str]) -> dict[str, str]:
    """Fetch canonical titles for a batch of arXiv IDs. On failure return {}."""
    if not arxiv_ids:
        return {}

    id_list = ",".join(arxiv_ids)
    url = f"https://export.arxiv.org/api/query?id_list={id_list}"

    try:
        if httpx is not None:
            with httpx.Client(timeout=_ARXIV_TIMEOUT_S) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return _parse_arxiv_atom_titles(resp.text)
        else:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "research-multi-agent-system/1.0"})
            with urllib.request.urlopen(req, timeout=_ARXIV_TIMEOUT_S) as resp:
                return _parse_arxiv_atom_titles(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"[PaperResearcher] arXiv title batch failed ({len(arxiv_ids)} ids): {e}")
        return {}


def resolve_arxiv_titles(papers: list[dict]) -> list[dict]:
    """
    Replace truncated/normalized titles with canonical arXiv titles when arxiv_id
    is present. Batches id_list requests; on failure keeps the existing title.
    """
    if not papers:
        return papers

    # Preserve order of first occurrence; prefer id as stored
    ids: list[str] = []
    seen: set[str] = set()
    for p in papers:
        aid = (p.get("arxiv_id") or "").strip()
        if aid and aid not in seen:
            seen.add(aid)
            ids.append(aid)

    if not ids:
        return papers

    title_map: dict[str, str] = {}
    batches = [
        ids[i : i + _ARXIV_BATCH_SIZE]
        for i in range(0, len(ids), _ARXIV_BATCH_SIZE)
    ]
    for bi, batch in enumerate(batches):
        if bi > 0:
            time.sleep(_ARXIV_BATCH_SLEEP_S)
        title_map.update(_fetch_arxiv_titles_batch(batch))

    upgraded = 0
    for p in papers:
        aid = (p.get("arxiv_id") or "").strip()
        if not aid:
            continue
        base = re.sub(r"v\d+$", "", aid)
        new_title = title_map.get(aid) or title_map.get(base)
        if not new_title:
            continue
        old = (p.get("title") or "").strip()
        if new_title != old:
            p["title"] = new_title
            upgraded += 1

    print(
        f"[PaperResearcher] arXiv title resolve: "
        f"{upgraded}/{len(ids)} titles upgraded "
        f"({len(batches)} batch request(s))"
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
                    identity = extract_identity(url, clean_title)
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

    # After final clean_papers: resolve canonical titles from arXiv when possible
    clean_papers = resolve_arxiv_titles(clean_papers)

    return {**state, "clean_papers": clean_papers}
