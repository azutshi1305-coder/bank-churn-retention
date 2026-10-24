"""FastAPI serving for the churn champion.

Run from the repo root:  uvicorn src.api.main:app
Environment: MODEL_PATH = directory written by src/models/export_champion.py (default: artifacts/champion).
The API needs neither mlflow.db nor mlruns/.
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.bands import BAND_DESCRIPTIONS, PROB_CLIP
from src.api.schemas import BatchRequest, BatchResponse, Customer, Health, ModelInfo, PredictResponse
from src.api.service import ChampionService

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "champion"

logger = logging.getLogger("churn_api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

PORTFOLIO_NOTE = ("Risk and expected benefit only, not a contact decision. How many customers to contact is a portfolio-level "
                  "choice limited by team capacity: score the whole customer list, rank it by expected_net_benefit, and contact "
                  "from the top, up to capacity and only while expected_net_benefit is positive.")

DESCRIPTION = f"""
Scores bank customers for churn risk with the calibrated champion model (no Gender input).

**Risk bands** on the calibrated churn probability: **low** = {BAND_DESCRIPTIONS['low']}; **medium** = {BAND_DESCRIPTIONS['medium']};
**high** = {BAND_DESCRIPTIONS['high']}. `GET /model-info` reports how each band behaved on the held-out test split.

**Probabilities are clipped** to [{PROB_CLIP[0]}, {PROB_CLIP[1]}] by the API (the registered model is unchanged): isotonic calibration can
return exactly 0 or 1, which overstates certainty about one customer. Risk bands and expected net benefit use the clipped value;
`GET /model-info` reports the test metrics before and after clipping.

**Not a contact decision.** {PORTFOLIO_NOTE}

**Money figures are illustrative.** Value and expected net benefit use the *base* scenario in `src/config/retention_assumptions.yaml`;
the retention success rate is an assumption because there is no treatment data.

**Privacy.** Each request logs only a request id, path, status, latency and the model version. Input values are never logged.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ChampionService(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))  # loaded once, at startup
    logger.info("model loaded: version=%s", app.state.service.version)
    yield


app = FastAPI(title="Bank churn scoring API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """422 with only where/what/why. FastAPI's default echoes the rejected input, which breaks on NaN (a 500) and returns values."""
    return JSONResponse(status_code=422, content={"detail": [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    service = getattr(request.app.state, "service", None)
    version = service.version if service else "none"
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # log the type only: messages and tracebacks could echo input values
        logger.error("request_id=%s method=%s path=%s status=500 latency_ms=%.1f model_version=%s error=%s",
                     request_id, request.method, request.url.path, (time.perf_counter() - start) * 1000, version, type(exc).__name__)
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%d latency_ms=%.1f model_version=%s",
                request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - start) * 1000, version)
    return response


def _frame(customers: list[Customer]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in customers])


@app.get("/health", response_model=Health)
def health(request: Request):
    """Liveness check; also reports the loaded model version."""
    return {"status": "ok", "model_version": request.app.state.service.version}


@app.get("/model-info", response_model=ModelInfo)
def model_info(request: Request):
    """Version, training date, calibration method, features, held-out test metrics, and how each risk band behaved on test."""
    md = request.app.state.service.metadata
    return {"model_name": md["model_name"], "model_version": md["model_version"], "run_id": md["run_id"],
            "training_date": md["training_date"], "calibration_method": md.get("calibration_method"),
            "n_features": md["n_features"], "features": request.app.state.service.features,
            "metrics": md["metrics"], "metrics_note": md["metrics_note"], "probability_clip": md["probability_clip"],
            "risk_bands": md["risk_bands"]}


PREDICT_DESCRIPTION = f"""
Returns the calibrated churn probability, a risk band, the customer's estimated annual value, the expected net benefit of
contacting them under the base assumptions, and the top 3 drivers (XGBoost contributions in log-odds, averaged over the
5 fold models). Drivers explain the model's score, not causes of churn.

**This is not a contact decision.** {PORTFOLIO_NOTE}
"""

BATCH_DESCRIPTION = f"""
Scores up to 1,000 customers; predictions come back in request order. Each prediction has the same fields as `/predict`.
To plan a campaign, rank the whole list by `expected_net_benefit`.

`top_drivers` is **off by default** here (`null`) because exact per-customer contributions are the expensive part; add
`?include_drivers=true` to get them (a few seconds for 1,000 customers).

**This is not a contact decision.** {PORTFOLIO_NOTE}
"""


@app.post("/predict", response_model=PredictResponse, summary="Score one customer", description=PREDICT_DESCRIPTION)
def predict(customer: Customer, request: Request):
    service = request.app.state.service
    return {"request_id": request.state.request_id, "model_version": service.version, "assumptions": service.assumptions,
            "note": PORTFOLIO_NOTE, **service.score(_frame([customer]))[0]}


@app.post("/predict/batch", response_model=BatchResponse, summary="Score up to 1,000 customers", description=BATCH_DESCRIPTION)
def predict_batch(body: BatchRequest, request: Request, include_drivers: bool = False):
    service = request.app.state.service
    return {"request_id": request.state.request_id, "model_version": service.version, "count": len(body.customers),
            "assumptions": service.assumptions, "note": PORTFOLIO_NOTE,
            "predictions": service.score(_frame(body.customers), drivers=include_drivers)}
