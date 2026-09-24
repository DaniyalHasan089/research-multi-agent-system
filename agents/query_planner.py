"""
Agent 1: Query Planner (LLM)

Takes the user's natural-language query and:
  1. Classifies the research domain (cs, medicine, physics, etc.)
  2. Expands the query into 5 targeted academic search strings

Includes resilient clamp/pad and deterministic fallback logic to ensure
the pipeline never hard-crashes on query count or minor JSON malformations.

Output is written to ResearchState: domain, search_queries
"""

import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import ResearchState
from utils.llm_factory import get_llm, extract_text


SYSTEM_PROMPT = """You are an expert academic research strategist specializing in scientific query formulation, conceptual disambiguation, and precision search retrieval.

Your task is to analyze a user's research question and:
1. Disambiguate polysemous, metaphorical, or dual-meaning terms:
   - In computer science, artificial intelligence, and software engineering, words like "harness", "agent", "transformer", "diffusion", "pipeline", "container", "prompt", and "alignment" have precise computational meanings.
   - For example, "harness engineering" in modern computer science refers to test harnesses, evaluation frameworks, AI agent runtime substrates, and benchmark harnesses. It MUST NEVER be confused with physical wearable gear (e.g. fall-arrest straps, climbing harnesses, dog harnesses, or automotive wiring).
   - Unless the user explicitly asks for physical, mechanical, or wearable equipment, computational and AI topics must be strictly classified under "cs" and anchored with software/AI terminology.
2. Define a "clarified_focus" (1 concise, unambiguous sentence) explicitly stating the conceptual boundary of the research (e.g., "AI and software agent harness engineering, including evaluation substrates, runtime execution environments, and benchmark frameworks").
3. Classify the primary research domain:
   - cs (Computer Science, Artificial Intelligence, Software Engineering, Machine Learning)
   - medicine (Clinical Medicine, Pharmacology, Healthcare)
   - biology (Biological Sciences, Genetics, Bioinformatics)
   - physics (Physics, Quantum, Astronomy, Materials Science)
   - economics (Economics, Quantitative Finance, Business)
   - engineering (Mechanical, Civil, Structural, Aerospace Engineering ONLY when physical equipment/structures are explicitly requested)
   - other (Other academic fields)
4. Generate exactly 5 targeted, high-precision academic search strings optimized for scientific databases (arXiv, Semantic Scholar, IEEE, PubMed).
   - CRITICAL REQUIREMENT: Every single search string MUST be anchored with unambiguous domain keywords to prevent cross-domain contamination (e.g., for AI/software harness: include terms like 'AI agent', 'LLM runtime', 'software benchmark', 'test harness architecture', 'evaluation substrate').
   - NEVER generate queries for physical wearable equipment, ergonomics, fall protection, or strap dynamics when the inquiry is about software or AI systems.

Respond ONLY with a valid JSON object in this exact format:
{
  "domain": "<cs | medicine | physics | biology | economics | engineering | other>",
  "clarified_focus": "<1 concise sentence defining the explicit conceptual boundary and intended domain>",
  "queries": [
    "<search string 1>",
    "<search string 2>",
    "<search string 3>",
    "<search string 4>",
    "<search string 5>"
  ]
}
Do not include any text outside the JSON object."""


def _fallback_queries(user_query: str, domain: str = "cs") -> list[str]:
    """Generate 5 deterministic academic queries anchored in the intended domain."""
    base = user_query.strip()
    # Check if query is software/AI harness engineering
    if re.search(r"\bharness\b", base, re.I) and not re.search(r"\b(fall|climbing|safety belt|strap|body|dog)\b", base, re.I):
        return [
            f"{base} AI agent runtime substrate",
            f"{base} LLM benchmark evaluation framework",
            f"{base} software architecture test automation",
            f"{base} observability autonomous systems",
            f"{base} foundation models empirical evaluation",
        ]
    return [
        f"{base} survey review",
        f"{base} methodology architecture",
        f"{base} state of the art benchmarks",
        f"{base} empirical analysis evaluation",
        f"{base} open challenges future directions",
    ]


def run(state: ResearchState) -> ResearchState:
    """
    Query Planner node — expands user_query into domain, clarified_focus, and 5 search strings.

    Args:
        state: Current ResearchState containing 'user_query', optionally 'year_from'/'year_to'.

    Returns:
        Updated ResearchState with 'domain', 'clarified_focus', and 'search_queries'.
    """
    user_query = state["user_query"].strip()
    year_from = state.get("year_from")
    year_to = state.get("year_to")
    print(f"[QueryPlanner] Planning queries for: '{user_query}'")
    if year_from or year_to:
        date_note = f" (date filter: {year_from or 'any'}–{year_to or 'any'})"
        print(f"[QueryPlanner] Date filter applied{date_note}")

    domain = "cs" if re.search(r"\b(harness|agent|llm|model|software|ai|prompt)\b", user_query, re.I) else "other"
    clarified_focus = ""
    queries: list[str] = []

    try:
        llm = get_llm(task="light")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Research question: {user_query}"),
        ]

        response = llm.invoke(messages)
        raw = extract_text(response.content)

        # Extract JSON from response (handling markdown fences)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            parsed_domain = str(parsed.get("domain", "")).strip().lower()
            if parsed_domain:
                domain = parsed_domain
            clarified_focus = str(parsed.get("clarified_focus", "")).strip()
            raw_queries = parsed.get("queries", [])
            if isinstance(raw_queries, list):
                for q in raw_queries:
                    clean_q = str(q).strip()
                    if clean_q and clean_q not in queries:
                        queries.append(clean_q)
    except Exception as e:
        print(f"[QueryPlanner] ⚠ LLM generation or parsing warning: {e}. Applying fallback.")

    # ── Establish default clarified_focus if absent ─────────────────────────
    if not clarified_focus:
        if re.search(r"\bharness\b", user_query, re.I) and not re.search(r"\b(fall|climbing|safety belt|strap|body|dog)\b", user_query, re.I):
            domain = "cs"
            clarified_focus = "AI Agent & Software Test Harness Engineering (Runtime Substrates, Benchmark Evaluators, Execution Environments)"
        else:
            clarified_focus = f"Academic literature review on {user_query} in {domain.upper()}"

    # ── Resilient Clamp & Pad to exactly 5 queries ───────────────────────────
    if not queries:
        queries = _fallback_queries(user_query, domain)

    fallback_pool = _fallback_queries(user_query, domain)
    for fb in fallback_pool:
        if len(queries) >= 5:
            break
        if fb not in queries:
            queries.append(fb)

    # Clamp to top 5
    queries = queries[:5]

    # ── Embed date range into each query string when provided ───────────────────────
    if year_from or year_to:
        date_suffix = " ".join(
            [str(year_from) if year_from else "", str(year_to) if year_to else ""]
        ).strip()
        queries = [f"{q} {date_suffix}" for q in queries]
        print(f"[QueryPlanner] Appended date range '{date_suffix}' to all queries")

    print(f"[QueryPlanner] Domain classified: '{domain}'")
    print(f"[QueryPlanner] Clarified Focus: '{clarified_focus}'")
    for i, q in enumerate(queries, 1):
        print(f"[QueryPlanner]   Query {i}: {q}")

    return {**state, "domain": domain, "clarified_focus": clarified_focus, "search_queries": queries}
