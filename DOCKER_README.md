# Docker Setup — AI Researcher Agent

A Docker Compose setup that runs the **FastAPI AI Researcher Agent** behind an **Nginx reverse proxy**, with full SSE streaming support for the `/research` endpoint and reliable PDF generation.

---

## Prerequisites

| Tool | Min version |
|------|-------------|
| Docker Desktop (Windows) / Docker Engine (Ubuntu) | 24+ |
| Docker Compose plugin | v2.20+ |

---

## 1 · First-time setup

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY (or OPENAI_API_KEY) and TAVILY_API_KEY
```

> **Never commit `.env` to version control.** It is excluded by `.dockerignore` and `.gitignore`.

---

## 2 · Build & Start

```bash
# Build the image and start all services in the background
docker compose up --build -d
```

The app is available at **http://localhost** once both containers are healthy (usually ~60 s on first run while dependencies install).

---

## 3 · Stop

```bash
docker compose down
```

> Generated PDFs are stored in the `pdf_storage` named volume and survive `down`. To also remove the volume: `docker compose down -v`

---

## 4 · Check Logs

```bash
# Live logs from all services
docker compose logs -f

# FastAPI only
docker compose logs -f api

# Nginx only
docker compose logs -f nginx
```

---

## 5 · Rebuild after code changes

```bash
docker compose up --build -d
```

Docker caches the dependency layer, so rebuilds are fast when only Python source files change.

---

## 6 · Useful commands

| Task | Command |
|------|---------|
| Check container health | `docker compose ps` |
| Open a shell in the API container | `docker compose exec api bash` |
| Remove containers + volumes | `docker compose down -v` |
| Pull the latest base images | `docker compose pull` |

---

## Architecture

```
Browser / Client
      │  HTTP :80
      ▼
  ┌─────────┐
  │  Nginx  │  reverse proxy, SSE-safe (no buffering on /research)
  └────┬────┘
       │ HTTP :8000 (internal)
       ▼
  ┌──────────┐
  │ FastAPI  │  LangGraph pipeline · Gemini/OpenAI · Tavily · xhtml2pdf
  └──────────┘
       │
  pdf_storage  ← named Docker volume (persists PDFs across restarts)
```

---

## Environment variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `gemini` or `openai` |
| `GOOGLE_API_KEY` | Required when `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` |
| `TAVILY_API_KEY` | Required for web search |
| `TOP_K_PAPERS` | Max papers analysed (default `15`) |

---

## Production notes

- **HTTPS / TLS**: Add a Certbot sidecar or terminate TLS at your load balancer / cloud gateway.  
- **Rate limiting**: Add `limit_req_zone` directives to `nginx/nginx.conf`.  
- **Scaling**: Increase `--workers` in the `CMD` inside `Dockerfile` or add replicas in `compose.yaml`.  
- **Secrets management**: Replace `.env` with Docker Secrets or a vault solution (e.g. AWS Secrets Manager, HashiCorp Vault).

