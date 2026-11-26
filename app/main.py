"""FastAPI prediction service for the Lucknow Rooftop Solar Cooperative.

Endpoints
  GET  /health   -> {"status": "ok", "model_loaded": true}
  POST /predict  -> {"prediction": 31.42, "unit": "kWh"}

The client's two rules live in the request schema, so a bad request gets
HTTP 422 from pydantic and never reaches the model:
  1. panel_age_yr above 30 is impossible  -> Field(le=30)
  2. temp_c below 6 is a typing mistake    -> Field(ge=6)

The model is the @champion pipeline saved as models/model.joblib (override
with the MODEL_PATH environment variable). It does its own preprocessing, so
the eight request fields go straight into it.

Run locally (repo root):  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # find solar_ml without setting PYTHONPATH
    sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from solar_ml.config import load_params, resolve  # noqa: E402
from solar_ml.logging_utils import get_logger  # noqa: E402

PARAMS = load_params()
FEATURES = list(PARAMS["columns"]["numeric"]) + list(PARAMS["columns"]["categorical"])
logger = get_logger("mp003-api")


class SolarRequest(BaseModel):
    """Exactly the eight fields the client sends. Unknown fields are rejected."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "panel_area_m2": 34.5, "sunshine_hours": 7.2, "temp_c": 31.0,
            "dust_days": 12, "panel_age_yr": 6.5, "cloud_cover_pct": 40,
            "roof_type": "flat", "area": "Old City"}},
    )

    panel_area_m2: float = Field(..., description="Panel area, square metres")
    sunshine_hours: float = Field(..., description="Hours of bright sunshine")
    temp_c: float = Field(..., ge=6, description="Maximum temperature, deg C (client rule: below 6 is a typo)")
    dust_days: float = Field(..., description="Days since the panels were cleaned")
    panel_age_yr: float = Field(..., le=30, description="Panel age, years (client rule: above 30 is impossible)")
    cloud_cover_pct: float = Field(..., description="Cloud cover at noon, percent")
    roof_type: str = Field(..., min_length=1, description="flat, sloped or tin")
    area: str = Field(..., min_length=1, description="Old City, New Colony or Industrial")


class PredictionResponse(BaseModel):
    prediction: float
    unit: str = "kWh"


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH") or resolve(PARAMS["model"]["output_path"]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path()
    try:
        app.state.model = joblib.load(path)
        logger.info("model loaded", extra={"fields": {"event": "startup", "model_path": str(path)}})
    except FileNotFoundError:
        app.state.model = None
        logger.error("model file not found", extra={"fields": {"event": "startup", "model_path": str(path)}})
    yield


app = FastAPI(title="MP003 Rooftop Solar Energy API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": getattr(app.state, "model", None) is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: SolarRequest) -> PredictionResponse:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    start = time.perf_counter()
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    value = round(float(model.predict(row)[0]), 3)
    logger.info("prediction", extra={"fields": {
        "event": "prediction",
        "prediction_kwh": value,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "roof_type": req.roof_type,
        "area": req.area,
        "panel_area_m2": req.panel_area_m2,
        "panel_age_yr": req.panel_age_yr,
    }})
    return PredictionResponse(prediction=value)
