import asyncio
import json
import os
import re
import uuid
import uvicorn
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv()

app = FastAPI(
    title="AI Researcher Agent",
    description="Multi-agent academic research pipeline",
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_STATIC_DIR = _ROOT / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"


# ── Request / Response Models ─────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str
    year_from: Optional[int] = None   # e.g. 2020 — include papers from this year
    year_to: Optional[int] = None     # e.g. 2024 — include papers up to this year


# ── SSE Helper ────────────────────────────────────────────────────────────────
def _sse_event(stage: str, progress: int, message: str, data: dict = None) -> str:
    """Serialize a progress event to a JSON string for SSE."""
    payload = {"stage": stage, "progress": progress, "message": message}
    if data:
        payload["data"] = data
    return json.dumps(payload)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-file frontend."""
    if not _INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend not found. Ensure static/index.html exists.")
    return HTMLResponse(content=_INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/research")
async def run_research(request: ResearchRequest):
    """
    Run the full 5-agent research pipeline and stream SSE progress events.

    Uses LangGraph's native astream() with stream_mode="updates". As each
    node in the DAG completes, an SSE update is yielded immediately to the UI.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator():
        # Import pipeline lazily inside the generator to avoid startup cost
        from graph.pipeline import pipeline
        from graph.state import ResearchState
        from agents.formatter import get_pdf_dir

        state: ResearchState = {
            "user_query": query,
            "year_from": request.year_from,
            "year_to": request.year_to,
        }
        accumulated_state: dict = dict(state)

        try:
            # Emit initial planning event
            yield _sse_event(
                "planning", 5, f"Planning research strategy for: \"{query[:60]}\""
            )
            await asyncio.sleep(0)

            # Native LangGraph async streaming execution
            async for update in pipeline.astream(state, stream_mode="updates"):
                for node_name, node_output in update.items():
                    accumulated_state.update(node_output)

                    if node_name == "query_planner":
                        domain = accumulated_state.get("domain", "other")
                        queries = accumulated_state.get("search_queries", [])
                        clarified_focus = accumulated_state.get("clarified_focus", "")
                        focus_str = f" • Focus: {clarified_focus[:65]}..." if clarified_focus else ""
                        yield _sse_event(
                            "query_planned", 15,
                            f"Domain: {domain.upper()}{focus_str} — Generated {len(queries)} search angles",
                            data={"domain": domain, "clarified_focus": clarified_focus, "queries": queries}
                        )
                        await asyncio.sleep(0)
                        yield _sse_event("searching", 20, f"Searching {domain} academic repositories via Tavily...")
                        await asyncio.sleep(0)

                    elif node_name == "paper_researcher":
                        papers = accumulated_state.get("clean_papers", [])
                        yield _sse_event(
                            "papers_found", 35,
                            f"Selected {len(papers)} balanced candidate papers across queries"
                        )
                        await asyncio.sleep(0)
                        yield _sse_event("analyzing", 40, f"Concurrently analyzing {len(papers)} papers with AI...")
                        await asyncio.sleep(0)

                    elif node_name == "analyzer_ranker":
                        analyses = accumulated_state.get("paper_analyses", [])
                        yield _sse_event(
                            "analyzed", 70,
                            f"Analyzed and ranked top {len(analyses)} papers by relevance"
                        )
                        await asyncio.sleep(0)
                        yield _sse_event("writing", 75, "Synthesizing research dossier with strict evidence grounding...")
                        await asyncio.sleep(0)

                    elif node_name == "report_writer":
                        report_md = accumulated_state.get("report_markdown", "")
                        words = len(report_md.split())
                        yield _sse_event(
                            "report_written", 90,
                            f"Report synthesized ({words} words across 8 sections)"
                        )
                        await asyncio.sleep(0)
                        yield _sse_event("formatting", 93, "Generating PDF & HTML dossier...")
                        await asyncio.sleep(0)

                    elif node_name == "formatter":
                        pdf_path = accumulated_state.get("pdf_path", "")
                        pdf_id = Path(pdf_path).stem if pdf_path else ""
                        analyzed_count = len(accumulated_state.get("paper_analyses", []))

                        yield _sse_event(
                            "complete", 100,
                            "Research complete! Your report is ready.",
                            data={
                                "report_html": accumulated_state.get("report_html", ""),
                                "report_markdown": accumulated_state.get("report_markdown", ""),
                                "paper_analyses": accumulated_state.get("paper_analyses", []),
                                "pdf_id": pdf_id,
                                "domain": accumulated_state.get("domain", ""),
                                "clarified_focus": accumulated_state.get("clarified_focus", ""),
                                "paper_count": analyzed_count,
                                "query": query,
                            }
                        )
                        await asyncio.sleep(0)

        except Exception as e:
            yield _sse_event(
                "error", 0,
                f"Pipeline failed: {str(e)}"
            )

    return EventSourceResponse(event_generator())


def _format_pdf_filename(topic: Optional[str]) -> str:
    """Format a safe, clean filename reflecting the research topic."""
    if not topic:
        return "research_report.pdf"
    # Keep only alphanumeric characters, spaces, hyphens, and underscores
    clean = re.sub(r"[^\w\s-]", "", topic).strip()
    clean = re.sub(r"[-\s]+", "_", clean)
    clean = clean[:60].strip("_")
    if not clean:
        return "research_report.pdf"
    if not clean.lower().endswith("report"):
        clean = f"{clean}_report"
    return f"{clean}.pdf"


@app.get("/download/{pdf_id}")
async def download_pdf(pdf_id: str, topic: Optional[str] = Query(default=None)):
    """
    Serve a previously generated PDF by ID with a topic-reflective filename.

    The pdf_id is the UUID stem of the file (no extension).
    Files are stored in the system temp directory.
    """
    from agents.formatter import get_pdf_dir

    # Sanitize: only allow UUID-like names
    if not all(c in "0123456789abcdef-" for c in pdf_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid PDF ID.")

    pdf_path = get_pdf_dir() / f"{pdf_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found. It may have expired.")

    # Determine topic from query param or sidecar metadata
    resolved_topic = topic
    if not resolved_topic:
        meta_path = get_pdf_dir() / f"{pdf_id}.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                resolved_topic = meta.get("topic")
            except Exception:
                pass

    filename = _format_pdf_filename(resolved_topic)

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
    )

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          
        reload_dirs=["."],    
    )