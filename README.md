# AI Researcher Agent

A multi-agent academic research pipeline that takes a natural-language question, autonomously discovers and analyzes relevant papers, and synthesizes everything into a structured report — delivered as an interactive web view and a downloadable PDF.

## Architecture

```
User Query
    │
    ▼
[1] Query Planner      (LLM)    — Expands query into 5 academic search strings + classifies domain
    │
    ▼
[2] Paper Researcher   (Tavily) — Searches academic sources, deduplicates, filters by content quality
    │
    ▼
[3] Analyzer & Ranker  (LLM)    — Extracts insights per paper, scores relevance, returns top-K
    │
    ▼
[4] Report Writer      (LLM)    — Synthesizes analyses into a 7-section Markdown report with citations
    │
    ▼
[5] Formatter          (Python) — Converts Markdown → Styled HTML → PDF via WeasyPrint
    │
    ▼
Interactive Web Report + PDF Download
```

**Stack:** LangGraph · FastAPI · Tavily · Gemini Flash / OpenAI · WeasyPrint · Vanilla JS

## Quick Start

### 1. Clone & create virtual environment

```bash
git clone https://github.com/your-username/research-multi-agent-system.git
cd research-multi-agent-system

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on WeasyPrint:** WeasyPrint requires system-level libraries.
> - **Windows:** Install [GTK for Windows](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases)
> - **macOS:** `brew install pango`
> - **Linux (Ubuntu/Debian):** `sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0`

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```
LLM_PROVIDER=gemini           # or openai

GOOGLE_API_KEY=...            # required if LLM_PROVIDER=gemini
OPENAI_API_KEY=...            # required if LLM_PROVIDER=openai

TAVILY_API_KEY=...            # always required — get one at https://tavily.com

TOP_K_PAPERS=10               # number of papers to analyze and include in report
```

### 4. Run

```bash
python main.py
```

Open **http://localhost:8000** in your browser.

## Project Structure

```
research-multi-agent-system/
├── agents/
│   ├── query_planner.py        # Agent 1 — LLM query expansion
│   ├── paper_researcher.py     # Agent 2 — Tavily search + filtering
│   ├── analyzer_ranker.py      # Agent 3 — LLM paper analysis + ranking
│   ├── report_writer.py        # Agent 4 — LLM report synthesis
│   └── formatter.py            # Agent 5 — Markdown → HTML → PDF
├── graph/
│   ├── state.py                # ResearchState TypedDict (shared schema)
│   └── pipeline.py             # LangGraph DAG definition
├── utils/
│   ├── llm_factory.py          # Provider-agnostic LLM init
│   └── pdf_styles.css          # WeasyPrint PDF stylesheet
├── static/
│   └── index.html              # Single-file frontend (HTML + CSS + JS)
├── api.py                      # FastAPI app (serves UI + SSE + PDF download)
├── main.py                     # Entry point — runs uvicorn
├── requirements.txt
└── .env.example
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend (`static/index.html`) |
| `POST` | `/research` | Runs the pipeline; streams SSE progress events |
| `GET` | `/download/{pdf_id}` | Downloads a generated PDF by ID |
| `GET` | `/docs` | FastAPI auto-generated API documentation |

### SSE Event Format

```json
{ "stage": "query_planned", "progress": 10, "message": "...", "data": {} }
```

Progress checkpoints: `5% → 10% → 15% → 30% → 35% → 65% → 70% → 90% → 92% → 100%`

## LLM Provider

Switch providers by changing `LLM_PROVIDER` in `.env`:

| Provider | Env var | Models used |
|---|---|---|
| Gemini (default) | `LLM_PROVIDER=gemini` | `gemini-1.5-flash` for all tasks |
| OpenAI | `LLM_PROVIDER=openai` | `gpt-4o-mini` (light) · `gpt-4o` (heavy) |

## Roadmap (v2)

- [ ] Parallel paper analysis (~60% speed reduction)
- [ ] Critic loop — LLM reviews and improves the report
- [ ] Citation link resolution — fetch DOIs, verify paper metadata
- [ ] Domain-specific source lists (e.g., PubMed for medicine)
- [ ] Query history and report caching