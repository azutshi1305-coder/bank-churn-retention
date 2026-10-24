# Bank churn scoring API.
#   Build:  docker build -t bank-churn-api:local .
#   Run:    docker run --rm -p 8000:8000 bank-churn-api:local
#
# Verified locally: builds; runs as uid 10001 (non-root); HEALTHCHECK reports healthy; runs with --read-only,
# --cap-drop ALL and --security-opt no-new-privileges; passes scripts/smoke_test.py; predictions identical to the
# Windows venv on 50 test customers (scripts/parity_check.py). Size is ~900 MB as reported by Docker.
#
# Dependencies are installed in several separate layers, all pinned by requirements-serve.txt (used as a pip
# constraints file). One 557 MB layer made `docker push` to ECR time out on a slow uplink; smaller layers upload
# independently and a retry only repeats the piece that failed.
#
# CPU limits: Python's os.cpu_count() reports the HOST's CPUs, not a container's limit, so XGBoost would start too
# many threads under a CPU cap. Set XGB_NTHREADS to the vCPUs the container gets, e.g.
#   docker run --rm --cpus 2 -e XGB_NTHREADS=2 -p 8000:8000 bank-churn-api:local

FROM python:3.11-slim

# libgomp1: OpenMP runtime used by xgboost for multithreaded prediction
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/artifacts/champion

RUN python -m venv /opt/venv
COPY requirements-serve.txt /tmp/requirements-serve.txt
# one heavy package group per layer (install order respects dependencies); wheels only, nothing is compiled
RUN pip install --only-binary=:all: -c /tmp/requirements-serve.txt numpy
RUN pip install --only-binary=:all: -c /tmp/requirements-serve.txt scipy
RUN pip install --only-binary=:all: -c /tmp/requirements-serve.txt pandas
RUN pip install --only-binary=:all: -c /tmp/requirements-serve.txt scikit-learn
RUN pip install --only-binary=:all: -c /tmp/requirements-serve.txt xgboost-cpu skops
# everything else, and a final check that every pin in requirements-serve.txt is satisfied
RUN pip install --only-binary=:all: -r /tmp/requirements-serve.txt && rm /tmp/requirements-serve.txt

WORKDIR /app
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
