"""
Shared arXiv metadata utility.
Provides reliable extraction, batch querying, HTML fallbacks, and multi-author formatting.
"""

import re
import ssl
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import httpx
except ImportError:
    httpx = None

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl._create_unverified_context()

_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

_TIMEOUT_S = 8.0        # API query timeout (reduced from 15s to fail fast)
_HTML_TIMEOUT_S = 5.0   # HTML scrape fallback timeout
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def extract_arxiv_id(text: str) -> str | None:
    """Extract an authentic arXiv ID from a URL, title, or snippet string."""
    if not text:
        return None
    
    # 1. URL or reference with arXiv domain or arxiv: prefix
    # Matches: arxiv.org/abs/2311.02462, arxiv.org/html/2311.02462, arxiv:2311.02462
    m_url = re.search(
        r"(?:arxiv(?:\.org)?/(?:abs|pdf|html|ps|format)/|arxiv:\s*)([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        text,
        re.I,
    )
    if m_url:
        return m_url.group(1).rstrip("/.")

    # 2. Classic category format with explicit arxiv.org or arxiv: prefix
    # e.g. arxiv.org/abs/cs/0612082 or arxiv:math/0602001
    m_old = re.search(
        r"(?:arxiv(?:\.org)?/(?:abs|pdf|html)/|arxiv:\s*)([a-z\-]+(?:\.[a-z]{2})?/[0-9]{7}(?:v\d+)?)",
        text,
        re.I,
    )
    if m_old:
        return m_old.group(1).rstrip("/.")

    # 3. If 'arxiv' is explicitly mentioned in the text, match standalone YYMM.NNNNN
    if "arxiv" in text.lower():
        m_gen = re.search(r"\b([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b", text)
        if m_gen:
            cand = m_gen.group(1)
            try:
                mm = int(cand[2:4])
                if 1 <= mm <= 12:
                    return cand
            except Exception:
                pass

    return None


def format_authors(authors: list[str]) -> str:
    """
    Format author list ensuring at least 2 authors are named when multiple exist.
    - 1 author  -> Author 1
    - 2 authors -> Author 1, Author 2
    - 3 authors -> Author 1, Author 2, Author 3
    - 4 authors -> Author 1, Author 2, Author 3, Author 4
    - 5+ authors -> Author 1, Author 2 et al.
    """
    cleaned = []
    for a in authors:
        a_clean = re.sub(r"\s+", " ", (a or "").strip())
        if a_clean and a_clean not in cleaned:
            cleaned.append(a_clean)

    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    elif len(cleaned) == 2:
        return f"{cleaned[0]}, {cleaned[1]}"
    elif len(cleaned) == 3:
        return f"{cleaned[0]}, {cleaned[1]}, {cleaned[2]}"
    elif len(cleaned) == 4:
        return f"{cleaned[0]}, {cleaned[1]}, {cleaned[2]}, {cleaned[3]}"
    else:
        # 5 or more authors: explicitly list the first 2 authors followed by et al.
        return f"{cleaned[0]}, {cleaned[1]} et al."


def _parse_atom_xml(xml_text: str) -> dict[str, dict]:
    """Parse arXiv Atom XML response into mapping: id -> metadata."""
    mapping: dict[str, dict] = {}
    if not xml_text:
        return mapping

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return mapping

    for entry in root.findall("atom:entry", _ATOM_NS):
        id_el = entry.find("atom:id", _ATOM_NS)
        if id_el is None or not (id_el.text or "").strip():
            continue

        id_text = id_el.text.strip()
        aid = extract_arxiv_id(id_text)
        if not aid:
            continue
        base = re.sub(r"v\d+$", "", aid)

        # Title
        title_el = entry.find("atom:title", _ATOM_NS)
        title = re.sub(r"\s+", " ", "".join(title_el.itertext())).strip() if title_el is not None else ""

        # Authors
        authors = []
        for a_el in entry.findall("atom:author", _ATOM_NS):
            name_el = a_el.find("atom:name", _ATOM_NS)
            if name_el is not None and (name_el.text or "").strip():
                authors.append(re.sub(r"\s+", " ", name_el.text).strip())

        author_str = format_authors(authors)

        # Published Year
        year = None
        pub_el = entry.find("atom:published", _ATOM_NS)
        if pub_el is not None and pub_el.text:
            ym = re.search(r"^(\d{4})", pub_el.text.strip())
            if ym:
                year = int(ym.group(1))

        # Venue / Journal
        venue = ""
        jref_el = entry.find("arxiv:journal_ref", _ATOM_NS)
        if jref_el is not None and (jref_el.text or "").strip():
            venue = re.sub(r"\s+", " ", jref_el.text).strip()
        if not venue:
            comm_el = entry.find("arxiv:comment", _ATOM_NS)
            if comm_el is not None and (comm_el.text or "").strip():
                comm_text = re.sub(r"\s+", " ", comm_el.text).strip()
                cm = re.search(r"(accepted (?:to|at|in)|published in|proceedings of)\s+([^,;.]+)", comm_text, re.I)
                if cm:
                    venue = cm.group(0).strip()
        if not venue:
            venue = f"arXiv:{base}"

        meta = {
            "title": title,
            "authors": authors,
            "author": author_str,
            "year": year,
            "venue": venue,
        }
        mapping[aid] = meta
        mapping[base] = meta

    return mapping


def _fetch_html_authors(base_id: str) -> dict:
    """Fallback: fetch authors and title directly from arxiv.org/abs/{id} HTML page."""
    url = f"https://arxiv.org/abs/{base_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_HTML_TIMEOUT_S, context=_SSL_CONTEXT) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Extract title
        title = ""
        m_title = re.search(r'<h1 class="title mathjax"[^>]*><span class="descriptor">Title:</span>(.*?)</h1>', html, re.DOTALL)
        if m_title:
            title = re.sub(r"\s+", " ", m_title.group(1)).strip()

        # Extract authors
        authors = []
        m_auth = re.search(r'<div class="authors"[^>]*>(.*?)</div>', html, re.DOTALL)
        if m_auth:
            names = re.findall(r'<a href="[^"]*search/arxiv[^"]*">([^<]+)</a>', m_auth.group(1))
            for n in names:
                n_clean = re.sub(r"\s+", " ", n).strip()
                if n_clean and n_clean not in authors:
                    authors.append(n_clean)

        # Extract year from dateline e.g. [Submitted on 6 Nov 2023]
        year = None
        m_year = re.search(r'Submitted on \d+\s+\w+\s+(\d{4})', html)
        if m_year:
            year = int(m_year.group(1))

        if authors:
            return {
                "title": title,
                "authors": authors,
                "author": format_authors(authors),
                "year": year,
                "venue": f"arXiv:{base_id}",
            }
    except Exception as e:
        print(f"[arXivUtils] HTML scrape fallback failed for {base_id}: {e}")

    return {}


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Fetch metadata for a single arXiv ID with API + HTML fallback."""
    if not arxiv_id:
        return {}
    base = re.sub(r"v\d+$", "", arxiv_id).strip()
    url = f"https://export.arxiv.org/api/query?id_list={base}"

    try:
        if httpx is not None:
            with httpx.Client(timeout=_TIMEOUT_S, headers={"User-Agent": _USER_AGENT}) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    res = _parse_atom_xml(resp.text)
                    if res.get(base) and res[base].get("author"):
                        return res[base]
        else:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S, context=_SSL_CONTEXT) as resp:
                xml_text = resp.read().decode("utf-8", errors="replace")
                res = _parse_atom_xml(xml_text)
                if res.get(base) and res[base].get("author"):
                    return res[base]
    except Exception as e:
        print(f"[arXivUtils] API query failed for {base}: {e}")

    # Fallback to HTML
    return _fetch_html_authors(base)


def batch_fetch_arxiv_metadata(arxiv_ids: list[str]) -> dict[str, dict]:
    """
    Fetch metadata for a list of arXiv IDs using small chunks to avoid timeouts,
    with automatic fallbacks for missing papers.
    """
    if not arxiv_ids:
        return {}

    clean_ids = []
    seen = set()
    for aid in arxiv_ids:
        b = re.sub(r"v\d+$", "", (aid or "").strip())
        if b and b not in seen:
            seen.add(b)
            clean_ids.append(b)

    results: dict[str, dict] = {}
    chunk_size = 6  # Small chunks respond quickly and reliably

    # Reuse a single httpx.Client across all chunks to avoid repeated TLS handshakes
    if httpx is not None:
        with httpx.Client(timeout=_TIMEOUT_S, headers={"User-Agent": _USER_AGENT}) as client:
            for i in range(0, len(clean_ids), chunk_size):
                chunk = clean_ids[i : i + chunk_size]
                if i > 0:
                    time.sleep(0.3)  # Respect polite request pacing
                id_str = ",".join(chunk)
                url = f"https://export.arxiv.org/api/query?id_list={id_str}"
                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        parsed = _parse_atom_xml(resp.text)
                        results.update(parsed)
                except Exception as e:
                    print(f"[arXivUtils] Batch query failed for chunk ({len(chunk)} ids): {e}")
    else:
        for i in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[i : i + chunk_size]
            if i > 0:
                time.sleep(0.3)  # Respect polite request pacing
            id_str = ",".join(chunk)
            url = f"https://export.arxiv.org/api/query?id_list={id_str}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
                with urllib.request.urlopen(req, timeout=_TIMEOUT_S, context=_SSL_CONTEXT) as resp:
                    xml_text = resp.read().decode("utf-8", errors="replace")
                    parsed = _parse_atom_xml(xml_text)
                    results.update(parsed)
            except Exception as e:
                print(f"[arXivUtils] Batch query failed for chunk ({len(chunk)} ids): {e}")

    # Concurrent HTML fallback for papers still missing authors after the batch
    missing = [b for b in clean_ids if not results.get(b) or not results[b].get("author")]
    if missing:
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_id = {executor.submit(_fetch_html_authors, b): b for b in missing}
            for future in as_completed(future_to_id):
                b = future_to_id[future]
                try:
                    fb = future.result()
                    if fb:
                        results[b] = fb
                except Exception:
                    pass

    return results
