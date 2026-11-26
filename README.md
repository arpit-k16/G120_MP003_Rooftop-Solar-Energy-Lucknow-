# Rooftop Solar Energy Forecasting — Lucknow

**Group G120 · Problem MP003 · SCSE3040 MLOps Mini Project**

An end-to-end MLOps pipeline that predicts the daily energy output (kWh) of
rooftop solar panels for the **Lucknow Rooftop Solar Cooperative**, so that
members can schedule heavy loads for sunny hours and spot under-performing
panels.

![Python](https://img.shields.io/badge/python-3.12-blue)
![pandas](https://img.shields.io/badge/pandas-2.2.3-150458)
![DVC](https://img.shields.io/badge/data-DVC-945DD6)
![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Docker](https://img.shields.io/badge/container-Docker-2496ED)
![Kubernetes](https://img.shields.io/badge/deploy-Kubernetes-326CE5)
![Stage](https://img.shields.io/badge/stage%203-complete-brightgreen)

---

## Contents

- [Overview](#overview)
- [Team](#team)
<!-- - [Project status](#project-status) -->
- [Pipeline](#pipeline)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Getting started](#getting-started)
- [Data quality findings](#data-quality-findings)
- [Cleaned-data contract](#cleaned-data-contract)
- [Model training and MLflow](#model-training-and-mlflow)
- [Configuration](#configuration)
- [Testing](#testing)
- [Prediction service](#prediction-service)
- [Monitoring and retraining plan](#monitoring-and-retraining-plan)
- [Contributing](#contributing)

---

## Overview

| | |
| --- | --- |
| **Client** | Lucknow Rooftop Solar Cooperative (fictional) |
| **Task** | Regression: predict `energy_kwh`, the energy produced in a day |
| **Metric** | MAE (Mean Absolute Error, in kWh) |
| **Training data** | `train.csv`, January–June 2026, 1,032 rows, intentionally messy |
| **Drift data** | `later.csv`, July–August 2026, 300 clean rows |
| **Holdout** | `holdout_features.csv`, September 2026, 250 rows, no target |

Full problem statement: [`PROBLEM.md`](PROBLEM.md). Course rules and marking:
[`STUDENT_HANDBOOK.md`](STUDENT_HANDBOOK.md).

## Team

| Enrolment no. | Member | Role | Owns |
| --- | --- | --- | --- |
| S24CSEU1793 | Arpit Kumar | Data lead | DVC, data checks, cleaning, data findings — [details](ARPITKUMAR_OWNED_FILES.md) |
| S24CSEU2215 | Rahul Nagilla | Model lead | Training, MLflow, champion model, holdout predictions — [details](Rahul_Nagilla_OWNED_FILES.md) |
| S24CSEU1783 | Alya Bathla | Service lead | FastAPI service, tests, Docker — [details](ALYA_OWNED_FILES.md) |
| S24CSEU1771 | Aman Bansal | Ops lead | CI, Kubernetes, logging, drift monitoring, retraining — [details](Aman_Bansal_OWNED_FILES.md) |

Hand-off order between roles: [`TEAM_OWNERSHIP.md`](TEAM_OWNERSHIP.md).

<!--
## Project status

| Stage | Scope | Owner | Status |
| --- | --- | --- | --- |
| 1 | Repository, DVC, seven data-quality checks, data part of `findings.yaml` | Arpit | ✅ Complete |
| 2 | Training with MLflow tracking, 3+ runs, registered `@champion` model | Rahul | ⏳ In progress |
| 3 | FastAPI service, Dockerfile, GitHub Actions CI, Kubernetes manifests, JSON logging | Alya, Aman | ⏳ In progress |
| 4 | PSI drift, per-group MAE, retrain v2, promotion decision, holdout predictions | Aman, Rahul | 🔜 Planned |
-->

## Pipeline

```mermaid
flowchart LR
    A[train.csv<br/>DVC-tracked] --> B[data_checks.py<br/>7 checks]
    B --> C[reports/findings.yaml]
    A --> D[cleaning.py]
    B --> D
    D --> E[train_clean.csv<br/>1,000 rows]
    E --> F[Training + MLflow<br/>Stage 2]
    F --> G[models/model.joblib<br/>@champion]
    G --> H[FastAPI /predict<br/>Stage 3]
    H --> I[Docker + Kubernetes<br/>Stage 3]
    J[later.csv] --> K[PSI drift + retrain v2<br/>Stage 4]
    G --> K
    K --> L[holdout_predictions.csv]
```

## Repository structure

```text
.
├── README.md                     # this file
├── PROBLEM.md                    # problem statement (from the course)
├── STUDENT_HANDBOOK.md           # course rules and marking
├── GROUP.md                      # group and members
├── TEAM_OWNERSHIP.md             # roles and hand-off order
├── *_OWNED_FILES.md              # each member's files and responsibilities
├── params.yaml                   # all paths, columns and thresholds
├── requirements.txt              # pinned dependencies
├── .dvc/  .dvcignore             # DVC configuration
├── data/
│   └── raw/
│       ├── train.csv.dvc         # DVC pointer (CSV itself is not in git)
│       ├── later.csv.dvc         # DVC pointer
│       └── holdout_features.csv  # committed directly (no target)
├── src/
│   └── solar_ml/
│       ├── __init__.py
│       ├── config.py             # loads params.yaml and data files
│       ├── data_checks.py        # detects the seven data problems
│       ├── cleaning.py           # fixes them, writes the cleaned dataset
│       ├── model.py              # preprocessing + regressor pipeline
│       ├── train.py              # MLflow runs, registers v1 as @champion
│       ├── evaluate.py           # MAE, MAE per area, model loading
│       └── logging_utils.py      # one JSON log line per record, to stdout
├── app/
│   ├── __init__.py
│   └── main.py                   # FastAPI: GET /health, POST /predict
├── tests/
│   ├── conftest.py
│   ├── test_data_checks.py       # data checks and cleaning
│   ├── test_model.py             # pipeline, MAE and training logic
│   └── test_api.py               # TestClient: endpoints, client rules, logs
├── Dockerfile                    # python:3.12-slim image mp003-api:1.0
├── .dockerignore
├── k8s/
│   ├── deployment.yaml           # mp003-api, 3 replicas, probes on /health
│   └── service.yaml              # mp003-svc, port 80 -> 8000
├── mlflow.db                     # MLflow tracking + model registry (SQLite)
├── models/
│   └── model.joblib              # the @champion pipeline the API loads
├── .github/workflows/ci.yml      # runs pytest on every push
└── reports/
    ├── data_checks.json          # raw output of data_checks.py
    ├── train_summary.json        # every candidate's scores and the chosen one
    └── findings.yaml             # the form read by the marking tool
```

Still to come in Stage 4: `src/solar_ml/monitoring.py` and
`predictions/holdout_predictions.csv`.

## Dataset

| Column | Unit | Description |
| --- | --- | --- |
| `record_id` | text | Unique id of the record |
| `date` | YYYY-MM-DD | Date of the record |
| `panel_area_m2` | m² | Panel area |
| `sunshine_hours` | hours | Hours of bright sunshine |
| `temp_c` | °C | Maximum temperature |
| `dust_days` | days | Days since the panels were last cleaned |
| `panel_age_yr` | years | Age of the panels |
| `cloud_cover_pct` | % | Cloud cover at noon |
| `roof_type` | text | `flat`, `sloped` or `tin` |
| `area` | text | `Old City`, `New Colony` or `Industrial` |
| `energy_kwh` | kWh | **Target**: energy produced in the day |

The raw CSVs are versioned with **DVC**, not git. `train.csv.dvc` records the
MD5 of the original file (`d8d15d2aea399d09c28acaeb7f52332a`); `dvc status`
confirms a local copy is unchanged. Never open and re-save `train.csv` in
Excel — it changes the hash.

## Getting started

### Prerequisites

- Python 3.12 (3.10+ works for the data code)
- Git, and DVC for checking the data files

### Setup

```bash
git clone https://github.com/arpit-k16/G120_MP003_Rooftop-Solar-Energy-Lucknow-.git
cd G120_MP003_Rooftop-Solar-Energy-Lucknow-
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

No DVC remote is configured, so copy `train.csv` and `later.csv` from the
course zip into `data/raw/`, then verify them against the committed hashes:

```bash
dvc status        # "Data and pipelines are up to date." means the files match
```

### Run the data checks and cleaning

```bash
export PYTHONPATH=src                       # Windows PowerShell: $env:PYTHONPATH="src"
python -m solar_ml.data_checks              # prints the 7 results, writes reports/data_checks.json
python -m solar_ml.cleaning                 # writes data/processed/train_clean.csv (1,000 rows)
```

## Data quality findings

Every finding below is produced by `src/solar_ml/data_checks.py` and recorded in
[`reports/findings.yaml`](reports/findings.yaml). Counts are on `train.csv` as
given (1,032 rows).

| Problem | Present | Finding | How it was detected |
| --- | :---: | --- | --- |
| Leaking column | No | — | Every feature exists in the holdout file; max \|correlation\| with the target is 0.19 |
| Unit mix-up | Yes | `panel_area_m2` in `New Colony` is in cm², factor **10000** | Group median ≈ 358,000 vs ≈ 33.5 elsewhere; snapped to the exact m² → cm² conversion |
| Sentinel value | Yes | `sunshine_hours` = **999** in **63** rows | `value_counts()` against known sentinels; real range is 2–11 h |
| Duplicate rows | Yes | **32** extra copies | `df.duplicated().sum()` |
| Stuck sensor | No | — | Sorted by date, no value repeats across 7+ days (longest run: 4 rows in one day) |
| Target typos | Yes | **13** rows of `energy_kwh` typed 10× too big | Largest gap in the sorted target: 49.78 → 131.2 |
| Spelling variants | Yes | `roof_type`, **3** true levels | 18 raw spellings normalised to `flat`, `sloped`, `tin` (as in `later.csv`) |

## Cleaned-data contract

`src/solar_ml/cleaning.py` applies the fixes in this order and writes
`data/processed/train_clean.csv` (not committed; regenerate it with the
command above):

1. Drop duplicate rows (1,032 → 1,000).
2. Map `roof_type` spellings to `flat` / `sloped` / `tin`.
3. Divide New Colony's `panel_area_m2` by 10000.
4. Replace `sunshine_hours` = 999 with the median of the real readings (6.2 h).
5. Divide the typo targets by 10.

**For model training:**

| | Columns |
| --- | --- |
| Numeric features | `panel_area_m2`, `sunshine_hours`, `temp_c`, `dust_days`, `panel_age_yr`, `cloud_cover_pct` |
| Categorical features | `roof_type`, `area` |
| Target | `energy_kwh` |
| Not features | `record_id`, `date` |

MLflow runs must log `data_sha256`, the SHA-256 of the original `train.csv`:
`5c46ec426e4f50f0469d970dee5d671889d90f06b97538f99b5f0221c8f8c3a5`
(`solar_ml.config.file_sha256()` computes it).

## Model training and MLflow

```bash
python -m solar_ml.train        # run once; PYTHONPATH=src as above
python -m solar_ml.evaluate     # MAE of models/model.joblib, overall and per area
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs
```

`train.py` cleans `train.csv`, holds back 20% of the cleaned rows, and logs one
MLflow run per candidate setting in `params.yaml` to the experiment
`mp003-solar-energy`. Each run logs `data_sha256`, `cv_mae` (5-fold MAE on the
other 80%) and `mae_val` (MAE on the held-back 20%).

| Run | Model | cv_mae (kWh) | mae_val (kWh) |
| --- | --- | ---: | ---: |
| `ridge_a0.1` | Ridge, alpha 0.1 | 1.425 | 1.555 |
| **`ridge_a1`** | **Ridge, alpha 1.0** | **1.424** | **1.555** |
| `ridge_a10` | Ridge, alpha 10 | 1.428 | 1.558 |
| `random_forest` | 300 trees, depth 10 | 2.138 | 2.176 |
| `gradient_boosting` | 300 trees, lr 0.05, depth 3 | 1.690 | 1.730 |

The relationship is close to linear, so Ridge wins. The best setting is refit
on all 1,000 cleaned rows as run `v1_ridge_a1`, registered as
**`mp003-solar-energy` version 1** with the alias **`@champion`**, and saved to
`models/model.joblib`. The run id is written to the `tracking` block of
`reports/findings.yaml`. Version 2 is reserved for the Stage 4 retrain
(`train.py` refuses to register again without `--force`).

The pipeline scales the numeric columns and one-hot encodes `roof_type` and
`area`, so the model takes the eight raw API fields directly.

## Configuration

All settings live in [`params.yaml`](params.yaml) and are read with
`yaml.safe_load` in `src/solar_ml/config.py`:

| Section | Holds |
| --- | --- |
| `data` | Paths to the raw, holdout and cleaned files |
| `columns` | Id, date, target, group, numeric and categorical columns |
| `data_checks` | Thresholds: sentinel candidates, unit factors, minimum stuck run, leak correlation, typo gap |
| `reports` | Output paths for the data-check results and findings |
| `mlflow` | Tracking URI, experiment, registered model name, champion alias |
| `model` | Validation split, CV folds, seed, and the candidate settings |

Stage 4 adds a monitoring section to the same file.

## Testing

```bash
pytest -q
```

The suite builds small synthetic datasets inside each test, so it runs without
the CSVs (as CI requires). It covers all seven detectors, including the two
problems absent from our data, an end-to-end cleaning run, the model
pipeline, the MAE helpers, candidate selection, and the API through FastAPI's
`TestClient` (a tiny model is trained inside the test). GitHub Actions runs it
on every push.

## Prediction service

`app/main.py` serves the `@champion` model from `models/model.joblib`
(override with the `MODEL_PATH` environment variable).

| Endpoint | Returns |
| --- | --- |
| `GET /health` | `{"status": "ok", "model_loaded": true}` |
| `POST /predict` | `{"prediction": 27.365, "unit": "kWh"}` |

The request body must have exactly these eight fields: `panel_area_m2`,
`sunshine_hours`, `temp_c`, `dust_days`, `panel_age_yr`, `cloud_cover_pct`,
`roof_type`, `area`. The client's two rules sit in the pydantic schema, so a bad
request gets **HTTP 422** and never reaches the model:

| Rule | Schema |
| --- | --- |
| `panel_age_yr` above 30 is impossible | `Field(..., le=30)` |
| `temp_c` below 6 is a typing mistake | `Field(..., ge=6)` |

Missing fields, extra fields (`extra="forbid"`) and wrong types also give 422.
If the model file is missing, `/predict` answers 503.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000      # open http://localhost:8000/docs
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"panel_area_m2":34.5,"sunshine_hours":7.2,"temp_c":31,"dust_days":12,"panel_age_yr":6.5,"cloud_cover_pct":40,"roof_type":"flat","area":"Old City"}'
```

### Logging

`src/solar_ml/logging_utils.py` writes **one JSON line per prediction** to
stdout (never to a file), with the prediction, latency and non-personal panel
details:

```json
{"timestamp": "2026-11-26T16:04:45.132+00:00", "level": "INFO", "logger": "mp003-api", "message": "prediction", "event": "prediction", "prediction_kwh": 27.365, "latency_ms": 5.72, "roof_type": "flat", "area": "Old City", "panel_area_m2": 34.5, "panel_age_yr": 6.5}
```

### Docker

```bash
docker build -t mp003-api:1.0 .
docker run -p 8000:8000 mp003-api:1.0
```

`python:3.12-slim`; `requirements.txt` is copied and installed **before** the
code so the dependency layer is cached; runs as a non-root user; `EXPOSE 8000`;
uvicorn on `0.0.0.0:8000`. `.dockerignore` keeps data, MLflow files, tests and
git history out of the image.

### Kubernetes

```bash
kubectl apply -f k8s/
kubectl get pods -l app=mp003-api        # 3 pods
kubectl port-forward svc/mp003-svc 8080:80
```

| Item | Value |
| --- | --- |
| Deployment | `mp003-api`, **3** replicas, image `mp003-api:1.0`, container port 8000 |
| Service | `mp003-svc`, port 80 → targetPort 8000, selector `app: mp003-api` (matches the pod labels) |
| Readiness probe | `GET /health`, starts after 5 s, every 10 s |
| Liveness probe | `GET /health`, starts after 20 s (waits longer), every 20 s |

### CI

`.github/workflows/ci.yml` runs on every push: Python 3.12, `pip install -r
requirements.txt`, `pytest`.

## Monitoring and retraining plan

To be implemented in Stage 4:

1. PSI (10 bins) for every numeric column, cleaned `train.csv` vs `later.csv`.
2. v1's MAE on `later.csv` for each `area`.
3. Retrain v2 on cleaned train + the first 70% of `later.csv` (by date); compare on the last 30%.
4. Promote v2 to `@champion` only if its MAE is at least 15% lower than v1's.
5. Predict `holdout_features.csv` with the champion → `predictions/holdout_predictions.csv`.

## Contributing

- Every member commits their own work from their **own GitHub account**.
- Work in the files listed in your `*_OWNED_FILES.md`; ask for review before
  changing another member's files.
- Never commit raw CSVs (except `holdout_features.csv`) or anything in `data/processed/`.
- Keep every library in `requirements.txt` pinned with `==`.
- Run `pytest` before pushing.
- Pull before you start (`git pull --rebase origin main`) to avoid conflicts.

---

*Academic project for SCSE3040 MLOps · Faculty: Pulkit Dwivedi · Batch 59.
The client and data are fictional.*
