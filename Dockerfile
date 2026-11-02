# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install Python dependencies in a separate stage to keep the final image lean.
FROM python:3.11-slim AS builder

WORKDIR /app

# Retry apt on transient network failures (common on some cloud/on-prem servers)
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries

# Install build tools needed to compile some Python packages (e.g. cffi, cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first so Docker can cache this layer
COPY requirements.txt .

# Install all Python dependencies into a dedicated prefix so we can copy them cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Retry apt on transient network failures
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries

# ── System libraries required by xhtml2pdf / ReportLab for PDF generation ────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info \
    fonts-dejavu-core \
    fonts-liberation \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code (no venv, no .env, no __pycache__)
COPY agents/      ./agents/
COPY graph/       ./graph/
COPY utils/       ./utils/
COPY static/      ./static/
COPY api.py       .

# The application loads secrets from environment variables at runtime.
# No .env file is baked into the image.

# Expose the FastAPI port
EXPOSE 8000

# Health-check: hit the root endpoint every 30 s, allow 60 s startup time.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run uvicorn without --reload in production (reload=True is for dev only).
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

