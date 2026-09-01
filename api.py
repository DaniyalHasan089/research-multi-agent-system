"""
FastAPI Application — API endpoints + static file serving.

Endpoints:
  GET  /                    → Serves static/index.html
  POST /research            → Runs the full pipeline; streams SSE progress events
  GET  /download/{pdf_id}   → Serves a generated PDF for download

SSE Event Schema:
  { "stage": str, "progress": int, "message": str, "data": dict }
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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

    Each event is a JSON object:
      { stage, progress, message, data? }

    The final event at progress=100 includes:
      data: { report_html, pdf_id, domain, paper_count }
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator():
        # Import pipeline lazily inside the generator to avoid startup cost
        from graph.pipeline import pipeline
        from graph.state import ResearchState
        from agents.formatter import get_pdf_dir

        state: ResearchState = {"user_query": query}

        try:
            # ── Stage 1: Query Planner ────────────────────────────────────────
            yield _sse_event(
                "planning", 5, f"Planning search strategy for: \"{query[:60]}\""
            )
            await asyncio.sleep(0)  # Yield control to event loop

            # Run the pipeline node by node so we can emit progress between nodes
            import agents.query_planner as qp
            state = await asyncio.to_thread(qp.run, state)

            yield _sse_event(
                "query_planned", 10,
                f"Domain: {state['domain']} — Generated {len(state['search_queries'])} search queries"
            )
            await asyncio.sleep(0)

            # ── Stage 2: Paper Researcher ─────────────────────────────────────
            yield _sse_event("searching", 15, "Searching academic sources via Tavily...")
            await asyncio.sleep(0)

            import agents.paper_researcher as pr
            state = await asyncio.to_thread(pr.run, state)

            paper_count = len(state.get("clean_papers", []))
            yield _sse_event(
                "papers_found", 30,
                f"Found {paper_count} relevant papers after filtering and deduplication"
            )
            await asyncio.sleep(0)

            # ── Stage 3: Analyzer & Ranker ────────────────────────────────────
            yield _sse_event(
                "analyzing", 35,
                f"Analyzing {paper_count} papers with AI — this is the longest step..."
            )
            await asyncio.sleep(0)

            import agents.analyzer_ranker as ar
            state = await asyncio.to_thread(ar.run, state)

            analyzed_count = len(state.get("paper_analyses", []))
            yield _sse_event(
                "analyzed", 65,
                f"Analyzed and ranked {analyzed_count} papers by relevance"
            )
            await asyncio.sleep(0)

            # ── Stage 4: Report Writer ────────────────────────────────────────
            yield _sse_event("writing", 70, "Synthesizing findings into a structured report...")
            await asyncio.sleep(0)

            import agents.report_writer as rw
            state = await asyncio.to_thread(rw.run, state)

            yield _sse_event(
                "report_written", 90,
                f"Report written ({len(state['report_markdown'].split())} words)"
            )
            await asyncio.sleep(0)

            # ── Stage 5: Formatter ────────────────────────────────────────────
            yield _sse_event("formatting", 92, "Generating PDF...")
            await asyncio.sleep(0)

            import agents.formatter as fmt
            state = await asyncio.to_thread(fmt.run, state)

            # Extract PDF ID from path
            pdf_path = state.get("pdf_path", "")
            pdf_id = Path(pdf_path).stem if pdf_path else ""

            yield _sse_event(
                "complete", 100,
                "Research complete! Your report is ready.",
                data={
                    "report_html": state.get("report_html", ""),
                    "pdf_id": pdf_id,
                    "domain": state.get("domain", ""),
                    "paper_count": analyzed_count,
                    "query": query,
                }
            )

        except Exception as e:
            yield _sse_event(
                "error", 0,
                f"Pipeline failed: {str(e)}"
            )

    return EventSourceResponse(event_generator())


@app.get("/download/{pdf_id}")
async def download_pdf(pdf_id: str):
    """
    Serve a previously generated PDF by ID.

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

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="research_report.pdf",
    )
