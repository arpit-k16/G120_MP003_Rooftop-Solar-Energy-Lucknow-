"""API tests with FastAPI's TestClient. No CSVs or MLflow needed: a tiny model
is trained inline and saved to a temp folder, and MODEL_PATH points the app at it.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `app`

from app.main import FEATURES, SolarRequest, app  # noqa: E402
from solar_ml.config import load_params  # noqa: E402
from solar_ml.model import build_pipeline  # noqa: E402

GOOD = {
    "panel_area_m2": 34.5, "sunshine_hours": 7.2, "temp_c": 31.0, "dust_days": 12,
    "panel_age_yr": 6.5, "cloud_cover_pct": 40, "roof_type": "flat", "area": "Old City",
}


def tiny_model():
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({
        "panel_area_m2": rng.uniform(8, 60, n), "sunshine_hours": rng.uniform(2, 11, n),
        "temp_c": rng.uniform(12, 46, n), "dust_days": rng.integers(0, 31, n),
        "panel_age_yr": rng.uniform(0, 15, n), "cloud_cover_pct": rng.integers(0, 91, n),
        "roof_type": rng.choice(["flat", "sloped", "tin"], n),
        "area": rng.choice(["Old City", "New Colony", "Industrial"], n),
    })
    y = 40 - 1.2 * df.panel_age_yr - 0.15 * df.cloud_cover_pct
    return build_pipeline("ridge", {"alpha": 1.0}, load_params()).fit(df, y)


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(tiny_model(), path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    with TestClient(app) as c:  # `with` runs the startup that loads the model
        yield c


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_predict_returns_a_number(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["prediction"], float) and body["unit"] == "kWh"
    assert 0 < body["prediction"] < 60


def test_schema_has_exactly_the_eight_fields():
    assert set(SolarRequest.model_fields) == set(FEATURES) == set(GOOD)


@pytest.mark.parametrize("field,bad,edge", [
    ("panel_age_yr", 30.1, 30),   # rule 1: above 30 is impossible
    ("temp_c", 5.9, 6),           # rule 2: below 6 is a typing mistake
])
def test_client_rules_give_422(client, field, bad, edge):
    assert client.post("/predict", json={**GOOD, field: bad}).status_code == 422
    assert client.post("/predict", json={**GOOD, field: edge}).status_code == 200


def test_missing_or_extra_field_gives_422(client):
    missing = {k: v for k, v in GOOD.items() if k != "area"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**GOOD, "energy_kwh": 30}).status_code == 422
    assert client.post("/predict", json={**GOOD, "temp_c": "hot"}).status_code == 422


def test_prediction_writes_one_json_log_line(client, capsys):
    capsys.readouterr()  # drop startup logs
    client.post("/predict", json=GOOD)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "prediction" and record["level"] == "INFO"
    assert isinstance(record["prediction_kwh"], float) and "latency_ms" in record


def test_missing_model_gives_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert c.post("/predict", json=GOOD).status_code == 503
