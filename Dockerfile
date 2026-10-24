# Bank churn scoring API. Build:  docker build -t bank-churn-api:local .
# Run:                            docker run --rm -p 8000:8000 bank-churn-api:local

# --- build stage: install pinned dependencies into a virtualenv (wheels only; nothing is compiled) ---
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements-serve.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --only-binary=:all: -r requirements-serve.txt

# --- runtime stage ---
FROM python:3.11-slim
# libgomp1: OpenMP runtime used by xgboost for multithreaded prediction
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/artifacts/champion

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# only what the API needs at runtime: no training code, notebooks, tests, data or tracking store
COPY src/__init__.py src/__init__.py
COPY src/features src/features
COPY src/api src/api
COPY src/config src/config
COPY artifacts/champion artifacts/champion

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
