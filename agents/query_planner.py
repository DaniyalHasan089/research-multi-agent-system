"""
Agent 1: Query Planner (LLM)

Takes the user's natural-language query and:
  1. Classifies the research domain (cs, medicine, physics, etc.)
  2. Expands the query into 5 targeted academic search strings

Output is written to ResearchState: domain, search_queries
"""

import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import ResearchState
from utils.llm_factory import get_llm, extract_text


SYSTEM_PROMPT = """You are an expert academic research assistant specializing in query formulation.
Your task is to analyze a user's research question and:
1. Classify the primary research domain.
2. Generate 5 distinct, targeted academic search strings optimized for databases like arXiv, PubMed, and Semantic Scholar.

Each search string should:
- Target a different angle or sub-topic of the main question
- Use academic terminology and keywords
- Be concise (5-10 words) but specific enough to return relevant papers

Respond ONLY with a valid JSON object in this exact format:
{
  "domain": "<single domain: cs | medicine | physics | biology | economics | psychology | engineering | other>",
  "queries": [
    "<search string 1>",
    "<search string 2>",
    "<search string 3>",
    "<search string 4>",
    "<search string 5>"
  ]
}
Do not include any text outside the JSON object."""


def run(state: ResearchState) -> ResearchState:
    """
    Query Planner node — expands user_query into domain + 5 search strings.

    Args:
        state: Current ResearchState containing 'user_query'.

    Returns:
        Updated ResearchState with 'domain' and 'search_queries'.
    """
    user_query = state["user_query"]
    print(f"[QueryPlanner] Planning queries for: '{user_query}'")

    llm = get_llm(task="light")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research question: {user_query}"),
    ]

    response = llm.invoke(messages)
    raw = extract_text(response.content)

    # Extract JSON from response (handles cases where LLM adds markdown fences)
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"[QueryPlanner] Could not parse JSON from LLM response:\n{raw}")

    parsed = json.loads(json_match.group())

    domain = parsed.get("domain", "other")
    queries = parsed.get("queries", [])

    if len(queries) != 5:
        raise ValueError(f"[QueryPlanner] Expected 5 search queries, got {len(queries)}")

    print(f"[QueryPlanner] Domain: {domain}")
    for i, q in enumerate(queries, 1):
        print(f"[QueryPlanner]   Query {i}: {q}")

    return {**state, "domain": domain, "search_queries": queries}
