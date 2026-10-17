# ---------------------------------------------------------------------------
# RetailPulse dashboard image  (used by Render, Docker, Kubernetes)
# Lean: only runtime deps. Pre-computed artifacts (data/processed, models) are
# baked in, so the container starts serving instantly.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .streamlit ./.streamlit
COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY data ./data
COPY models ./models
COPY reports ./reports

# Run as an unprivileged user (data/ stays writable for the first-run fallback)
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health', timeout=4)" || exit 1

# Render injects $PORT (default 10000); locally it falls back to 8501.
CMD ["sh", "-c", "exec streamlit run dashboard/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
