"""
LangGraph Pipeline — defines the linear 5-node StateGraph DAG.

Flow:
  START → query_planner → paper_researcher → analyzer_ranker → report_writer → formatter → END

Each node is a thin wrapper that calls the corresponding agent's run() function.
No cycles in v1. Critic loop and parallel analysis are planned for v2.
"""

from langgraph.graph import StateGraph, START, END
from graph.state import ResearchState
import agents.query_planner as query_planner
import agents.paper_researcher as paper_researcher
import agents.analyzer_ranker as analyzer_ranker
import agents.report_writer as report_writer
import agents.formatter as formatter


def _node_query_planner(state: ResearchState) -> ResearchState:
    """Node 1: Expand user query into domain + 5 search strings."""
    return query_planner.run(state)


def _node_paper_researcher(state: ResearchState) -> ResearchState:
    """Node 2: Search Tavily, deduplicate, filter, return clean paper list."""
    return paper_researcher.run(state)


def _node_analyzer_ranker(state: ResearchState) -> ResearchState:
    """Node 3: Analyze each paper with LLM, score, return top-K."""
    return analyzer_ranker.run(state)


def _node_report_writer(state: ResearchState) -> ResearchState:
    """Node 4: Synthesize analyses into a 7-section Markdown report."""
    return report_writer.run(state)


def _node_formatter(state: ResearchState) -> ResearchState:
    """Node 5: Convert Markdown → HTML → PDF."""
    return formatter.run(state)


def build_pipeline() -> StateGraph:
    """
    Build and compile the research pipeline StateGraph.

    Returns:
        A compiled LangGraph app ready to invoke.
    """
    graph = StateGraph(ResearchState)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("query_planner", _node_query_planner)
    graph.add_node("paper_researcher", _node_paper_researcher)
    graph.add_node("analyzer_ranker", _node_analyzer_ranker)
    graph.add_node("report_writer", _node_report_writer)
    graph.add_node("formatter", _node_formatter)

    # ── Wire edges (linear DAG) ──────────────────────────────────────────────
    graph.add_edge(START, "query_planner")
    graph.add_edge("query_planner", "paper_researcher")
    graph.add_edge("paper_researcher", "analyzer_ranker")
    graph.add_edge("analyzer_ranker", "report_writer")
    graph.add_edge("report_writer", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()


# Singleton pipeline instance — compiled once and reused across requests
pipeline = build_pipeline()
