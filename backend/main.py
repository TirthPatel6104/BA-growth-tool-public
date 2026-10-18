"""
backend/main.py
================
Extended FastAPI application that exposes the RGR training pipeline
and prediction service alongside the existing endpoints.

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import math
import os
import json
import copy
import sys
import threading
import time
import uuid
import glob
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Ensure backend root is on path
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from api.deps import DbSession
from db.bootstrap import init_db
from db.config import get_dialect, redact_url
from db.engine import get_server_version, list_tables, ping
from training.pipeline import RGRTrainingPipeline
from prediction.predictor import RGRPredictor
from data.extractors import SPECIES_CONFIG, FOREST_UNITS, VALID_COMBINATIONS, BAGrowthDataExtractor
from models.coefficient_store import CoefficientStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Start-up: make sure the configured database has the current schema and,
    on the zero-config SQLite path, the synthetic demo dataset
    (see db/bootstrap.py – controlled by BAGT_AUTO_MIGRATE / BAGT_AUTO_SEED).
    """
    try:
        summary = init_db()
        logger.info(
            "Database ready: %s (%s) migrated=%s seeded=%s",
            summary["url"], summary["dialect"], summary["migrated"], summary["seeded"],
        )
    except Exception as exc:  # noqa: BLE001 – keep the API up; /health reports the DB state
        logger.error("Database initialisation failed: %s", exc)
    yield


app = FastAPI(title="BA Growth Tool API", version="1.0.0", lifespan=lifespan)


def _cors_origins_from_env() -> List[str]:
    raw = os.getenv("BAGT_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

_ALLOWED_CORS_ORIGINS = list(dict.fromkeys(_DEFAULT_CORS_ORIGINS + _cors_origins_from_env()))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()

@app.middleware("http")
async def add_telemetry_headers(request, call_next):
    request_id = str(uuid.uuid4())
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.4f}s"
    return response

@app.get("/health")
def health_check(db: DbSession) -> Dict[str, Any]:
    """Enterprise telemetry and health check endpoint."""
    db_status = "HEALTHY" if ping(db) else "UNAVAILABLE"

    uptime_seconds = int(time.time() - _START_TIME)
    return {
        "status": "UP",
        "service": "Basal Area Growth Rate API",
        "version": "1.0.0-DEMO",
        "environment": os.getenv("ENV", "production-demo"),
        "uptime_seconds": uptime_seconds,
        "database_connection": db_status,
        "database": {"dialect": get_dialect(), "url": redact_url()},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def _species_display_name(species_abbrev: str) -> str:
    name = str(SPECIES_CONFIG[species_abbrev]["name"])
    if "(" in name:
        name = name.split("(", 1)[0].strip()
    return name.replace(" ", "")


def _forest_unit_display_name(forest_unit: str) -> str:
    return f"FU_{forest_unit}"


_SPECIES_ALIAS_TO_ABBREV: Dict[str, str] = {}
for _abbrev, _cfg in SPECIES_CONFIG.items():
    _name = str(_cfg["name"])
    _name_no_paren = _name.split("(", 1)[0].strip()
    _candidates = {
        _abbrev,
        _abbrev.lower(),
        _name,
        _name.lower(),
        _name_no_paren,
        _name_no_paren.lower(),
        _name_no_paren.replace(" ", ""),
        _name_no_paren.replace(" ", "").lower(),
        _species_display_name(_abbrev),
        _species_display_name(_abbrev).lower(),
    }
    for _candidate in _candidates:
        _SPECIES_ALIAS_TO_ABBREV[_candidate] = _abbrev

_FOREST_UNIT_ALIAS_TO_CODE: Dict[str, str] = {}
for _fu in FOREST_UNITS:
    _candidates = {
        _fu,
        _fu.lower(),
        f"FU_{_fu}",
        f"fu_{_fu.lower()}",
    }
    for _candidate in _candidates:
        _FOREST_UNIT_ALIAS_TO_CODE[_candidate] = _fu

_SPEC_CODE_TO_ABBREV: Dict[int, str] = {
    int(cfg["code"]): abbrev
    for abbrev, cfg in SPECIES_CONFIG.items()
    if cfg.get("code") is not None and int(cfg["code"]) != 999
}


def _resolve_species_abbrev(species_group: str) -> str:
    resolved = _SPECIES_ALIAS_TO_ABBREV.get(species_group)
    if resolved:
        return resolved
    lowered = _SPECIES_ALIAS_TO_ABBREV.get(species_group.lower())
    if lowered:
        return lowered
    compact = _SPECIES_ALIAS_TO_ABBREV.get(species_group.replace(" ", ""))
    if compact:
        return compact
    compact_lower = _SPECIES_ALIAS_TO_ABBREV.get(species_group.replace(" ", "").lower())
    if compact_lower:
        return compact_lower
    valid = sorted({_species_display_name(abbrev) for abbrev in SPECIES_CONFIG})
    raise HTTPException(status_code=400, detail=f"Unknown species_group '{species_group}'. Valid options: {valid}")


def _resolve_model_group(model_group: str) -> Dict[str, Optional[str]]:
    species_abbrev = _SPECIES_ALIAS_TO_ABBREV.get(model_group) or _SPECIES_ALIAS_TO_ABBREV.get(model_group.lower())
    if species_abbrev:
        return {
            "kind": "species",
            "species_abbrev": species_abbrev,
            "forest_unit": None,
            "display": _species_display_name(species_abbrev),
        }

    forest_unit = _FOREST_UNIT_ALIAS_TO_CODE.get(model_group) or _FOREST_UNIT_ALIAS_TO_CODE.get(model_group.lower())
    if forest_unit:
        return {
            "kind": "forest_unit",
            "species_abbrev": None,
            "forest_unit": forest_unit,
            "display": _forest_unit_display_name(forest_unit),
        }

    valid_species = sorted({_species_display_name(abbrev) for abbrev in SPECIES_CONFIG})
    valid_fus = sorted({_forest_unit_display_name(forest_unit) for forest_unit in FOREST_UNITS})
    valid = valid_species + valid_fus
    raise HTTPException(status_code=400, detail=f"Unknown model_group '{model_group}'. Valid options: {valid}")


def _extract_species_data(species_abbrev: str, db: Session):
    return _extract_model_group_data(species_abbrev, None, db)


def _extract_model_group_data(species_abbrev: Optional[str], forest_unit: Optional[str], db: Session):
    cached_df = _get_cached_model_data(species_abbrev, forest_unit)
    if cached_df is not None:
        return cached_df

    extractor = BAGrowthDataExtractor(db, populate_tcr=True)
    df = extractor.extract_training_data(species_abbrev, forest_unit, since_year=0)

    _set_cached_model_data(species_abbrev, forest_unit, df)
    return df.copy(deep=True)


def _summary_stats_for_column(df, column_name: str) -> Optional[Dict[str, float]]:
    import pandas as pd

    if column_name not in df.columns:
        return None
    values = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _two_sigma_outlier_stats(df, column_name: str) -> Optional[Dict[str, Any]]:
    import pandas as pd

    if column_name not in df.columns:
        return None

    values = pd.to_numeric(df[column_name], errors="coerce").dropna()
    sample_size = int(len(values))
    if sample_size == 0:
        return None

    mean = float(values.mean())
    std_dev = float(values.std(ddof=1)) if sample_size > 1 else 0.0
    lower_bound = mean - (2.0 * std_dev)
    upper_bound = mean + (2.0 * std_dev)

    if std_dev <= 0:
        outside_count = 0
    else:
        outside_count = int(((values < lower_bound) | (values > upper_bound)).sum())

    outside_percent = float((outside_count / sample_size) * 100.0)

    return {
        "sample_size": sample_size,
        "mean": mean,
        "std_dev": std_dev,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outside_count": outside_count,
        "outside_percent": outside_percent,
    }


def _model_info_for_species(spec_code: int, model_version: str, db: Session) -> Dict[str, Any]:
    store = CoefficientStore(db)
    coefs = store.load_coefficients(spec_code, None, model_version=model_version)
    meta = store.get_model_info(spec_code, None, model_version=model_version)

    return {
        "coefficient_id": int(meta["coefficient_id"]) if meta and meta.get("coefficient_id") is not None else None,
        "intercept": float(coefs["intercept"]),
        "coefficients": {
            "BA": float(coefs["BA"]),
            "BA2": float(coefs.get("BA2", 0.0)),
            "AGSfrac": float(coefs["AGSfrac"]),
            "TCR": float(coefs["TCR"]),
            "DQ": float(coefs["DQ"]),
            "q": float(coefs["q"]),
            "Cut": float(coefs["Cut"]),
        },
        "cv_metrics": {
            "rmse_mean": float(coefs["rmse"]) if coefs.get("rmse") is not None else None,
            "rmse_std": None,
            "r2_mean": float(coefs["r_squared"]) if coefs.get("r_squared") is not None else None,
            "r2_std": None,
        },
        "n_training_rows": int(coefs["n_samples"]) if coefs.get("n_samples") is not None else 0,
        "fitted_at": meta["trained_date"] if meta and meta.get("trained_date") else "",
    }


def _model_info_for_group(
    spec_code: Optional[int], forest_unit: Optional[str], model_version: str, db: Session
) -> Dict[str, Any]:
    store = CoefficientStore(db)
    coefs = store.load_coefficients(spec_code, forest_unit, model_version=model_version)
    meta = store.get_model_info(spec_code, forest_unit, model_version=model_version)

    return {
        "coefficient_id": int(meta["coefficient_id"]) if meta and meta.get("coefficient_id") is not None else None,
        "intercept": float(coefs["intercept"]),
        "coefficients": {
            "BA": float(coefs["BA"]),
            "BA2": float(coefs.get("BA2", 0.0)),
            "AGSfrac": float(coefs["AGSfrac"]),
            "TCR": float(coefs["TCR"]),
            "DQ": float(coefs["DQ"]),
            "q": float(coefs["q"]),
            "Cut": float(coefs["Cut"]),
        },
        "cv_metrics": {
            "rmse_mean": float(coefs["rmse"]) if coefs.get("rmse") is not None else None,
            "rmse_std": None,
            "r2_mean": float(coefs["r_squared"]) if coefs.get("r_squared") is not None else None,
            "r2_std": None,
        },
        "n_training_rows": int(coefs["n_samples"]) if coefs.get("n_samples") is not None else 0,
        "fitted_at": meta["trained_date"] if meta and meta.get("trained_date") else "",
    }


def _load_fallback_coefficients() -> Dict[str, Any]:
    fallback_path = os.path.join(_BACKEND_DIR, "artifacts", "models", "model_coefficients.json")
    if not os.path.exists(fallback_path):
        raise ValueError("No stored species coefficients and no fallback model_coefficients.json found")

    with open(fallback_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    coeffs = payload.get("coefficients", {})
    model_meta = payload.get("model_info", {})

    return {
        "intercept": float(payload.get("intercept", 0.0)),
        "coefficients": {
            "BA": float(coeffs.get("BA", 0.0)),
            "BA2": float(coeffs.get("BA2", 0.0)),
            "AGSfrac": float(coeffs.get("AGSfrac", 0.0)),
            "TCR": float(coeffs.get("TCR", 0.0)),
            "DQ": float(coeffs.get("DQ", 0.0)),
            "q": float(coeffs.get("q", 0.0)),
            "Cut": float(coeffs.get("Cut", 0.0)),
        },
        "cv_metrics": {
            "rmse_mean": float(model_meta.get("rmse")) if model_meta.get("rmse") is not None else None,
            "rmse_std": None,
            "r2_mean": float(model_meta.get("r2_score")) if model_meta.get("r2_score") is not None else None,
            "r2_std": None,
        },
        "n_training_rows": int(model_meta.get("n_samples", 0)),
        "fitted_at": str(model_meta.get("training_date", "")),
    }


def _predict_rgr_from_coefficients(
    model_info: Dict[str, Any],
    BA: float,
    AGSfrac: float,
    TCR: int,
    DQ: float,
    q: float,
    Cut: int,
) -> float:
    coefs = model_info["coefficients"]
    return float(
        model_info["intercept"]
        + coefs.get("BA", 0.0) * BA
        + coefs.get("BA2", 0.0) * (BA ** 2)
        + coefs.get("AGSfrac", 0.0) * AGSfrac
        + coefs.get("TCR", 0.0) * TCR
        + coefs.get("DQ", 0.0) * DQ
        + coefs.get("q", 0.0) * q
        + coefs.get("Cut", 0.0) * Cut
    )


def _non_negative_growth(ba_growth: float, rgr: float) -> tuple[float, float]:
    if ba_growth < 0:
        return 0.0, 0.0
    return float(ba_growth), float(rgr)


def _cut_positive_growth_floor(df: Any) -> float:
    import pandas as pd

    if df is None or len(df) == 0 or "BAgrowth" not in df.columns:
        return 0.0

    growth = pd.to_numeric(df["BAgrowth"], errors="coerce")
    if "Cut" in df.columns:
        cut = pd.to_numeric(df["Cut"], errors="coerce").fillna(0)
    else:
        cut = pd.Series(0.0, index=growth.index)

    cut_positive = growth[(cut >= 1) & (growth > 0)].dropna()
    if not cut_positive.empty:
        return float(cut_positive.median())

    any_positive = growth[growth > 0].dropna()
    if not any_positive.empty:
        return float(any_positive.median())

    return 0.0


def _apply_cut_growth_floor(
    ba_growth: float,
    rgr: float,
    cut: int,
    ba_current: float,
    cut_growth_floor: float,
) -> tuple[float, float]:
    adjusted_growth, adjusted_rgr = _non_negative_growth(ba_growth, rgr)

    if int(cut) >= 1 and adjusted_growth <= 0 and cut_growth_floor > 0:
        adjusted_growth = float(cut_growth_floor)
        adjusted_rgr = float(cut_growth_floor / ba_current) if ba_current > 0 else float(adjusted_rgr)

    return adjusted_growth, adjusted_rgr


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    species_abbrev: Optional[str] = Field(None, description="Train one species model, e.g. 'Mh', 'Cedar'")
    forest_unit:    Optional[str] = Field(None, description="Train one forest-unit model, e.g. 'TH1', 'HDSL2'")
    train_all:      bool          = Field(False, description="Train all configured models")
    save_to_db:     bool          = Field(True,  description="Persist coefficients")
    populate_tcr:   bool          = Field(True,  description="Enrich TCR from shapefile")
    model_version:  str           = Field("1.0")


# ---------------------------------------------------------------------------
# In-memory training job registry
# ---------------------------------------------------------------------------

_job_lock:     threading.Lock         = threading.Lock()
_job_registry: Dict[str, Dict]        = {}   # job_id → job dict


_response_cache_lock: threading.Lock = threading.Lock()
_response_cache: Dict[str, Dict[str, Any]] = {}
_RESPONSE_CACHE_VERSION = "v4_cut_positive_fallback"
_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ENTRIES = 256

_model_data_cache_lock: threading.Lock = threading.Lock()
_model_data_cache: Dict[str, Dict[str, Any]] = {}
_MODEL_DATA_CACHE_TTL_SECONDS = 300
_MODEL_DATA_CACHE_MAX_ENTRIES = 64


def _make_cache_key(namespace: str, **params: Any) -> str:
    normalized = "|".join(f"{key}={params[key]}" for key in sorted(params.keys()))
    return f"{_RESPONSE_CACHE_VERSION}|{namespace}|{normalized}"


def _prune_response_cache_locked(now_ts: float) -> None:
    expired_keys = [
        key for key, entry in _response_cache.items()
        if float(entry.get("expires_at", 0.0)) <= now_ts
    ]
    for key in expired_keys:
        _response_cache.pop(key, None)

    if len(_response_cache) <= _CACHE_MAX_ENTRIES:
        return

    by_oldest = sorted(_response_cache.items(), key=lambda item: float(item[1].get("created_at", 0.0)))
    to_remove = len(_response_cache) - _CACHE_MAX_ENTRIES
    for key, _ in by_oldest[:to_remove]:
        _response_cache.pop(key, None)


def _get_cached_response(cache_key: str) -> Optional[Dict[str, Any]]:
    now_ts = time.time()
    with _response_cache_lock:
        entry = _response_cache.get(cache_key)
        if not entry:
            return None
        if float(entry.get("expires_at", 0.0)) <= now_ts:
            _response_cache.pop(cache_key, None)
            return None
        return copy.deepcopy(entry.get("value"))


def _set_cached_response(cache_key: str, payload: Dict[str, Any], ttl_seconds: int = _CACHE_TTL_SECONDS) -> None:
    now_ts = time.time()
    with _response_cache_lock:
        _response_cache[cache_key] = {
            "created_at": now_ts,
            "expires_at": now_ts + max(1, int(ttl_seconds)),
            "value": copy.deepcopy(payload),
        }
        _prune_response_cache_locked(now_ts)


def _model_data_cache_key(species_abbrev: Optional[str], forest_unit: Optional[str]) -> str:
    return f"species={species_abbrev or '__NONE__'}|fu={forest_unit or '__NONE__'}"


def _prune_model_data_cache_locked(now_ts: float) -> None:
    expired_keys = [
        key for key, entry in _model_data_cache.items()
        if float(entry.get("expires_at", 0.0)) <= now_ts
    ]
    for key in expired_keys:
        _model_data_cache.pop(key, None)

    if len(_model_data_cache) <= _MODEL_DATA_CACHE_MAX_ENTRIES:
        return

    by_oldest = sorted(_model_data_cache.items(), key=lambda item: float(item[1].get("created_at", 0.0)))
    to_remove = len(_model_data_cache) - _MODEL_DATA_CACHE_MAX_ENTRIES
    for key, _ in by_oldest[:to_remove]:
        _model_data_cache.pop(key, None)


def _get_cached_model_data(species_abbrev: Optional[str], forest_unit: Optional[str]):
    now_ts = time.time()
    key = _model_data_cache_key(species_abbrev, forest_unit)
    with _model_data_cache_lock:
        entry = _model_data_cache.get(key)
        if not entry:
            return None
        if float(entry.get("expires_at", 0.0)) <= now_ts:
            _model_data_cache.pop(key, None)
            return None
        df = entry.get("value")
        return df.copy(deep=True) if df is not None else None


def _set_cached_model_data(species_abbrev: Optional[str], forest_unit: Optional[str], df, ttl_seconds: int = _MODEL_DATA_CACHE_TTL_SECONDS) -> None:
    now_ts = time.time()
    key = _model_data_cache_key(species_abbrev, forest_unit)
    with _model_data_cache_lock:
        _model_data_cache[key] = {
            "created_at": now_ts,
            "expires_at": now_ts + max(1, int(ttl_seconds)),
            "value": df.copy(deep=True),
        }
        _prune_model_data_cache_locked(now_ts)


def _clear_runtime_caches() -> None:
    with _response_cache_lock:
        _response_cache.clear()
    with _model_data_cache_lock:
        _model_data_cache.clear()


def _make_job(req: TrainRequest) -> Dict[str, Any]:
    """Create a new job record and add to registry."""
    job_id = str(uuid.uuid4())
    job: Dict[str, Any] = {
        "job_id":      job_id,
        "kind":        "training",
        "status":      "running",
        "started_at":  datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "request":     req.model_dump(),
        "progress":    {"total": 0, "done": 0, "failed": 0, "current": None},
        "results":     {},
        "errors":      {},
    }
    with _job_lock:
        _job_registry[job_id] = job
    return job


def _run_training_job(job: Dict, req: TrainRequest) -> None:
    """Target function executed in a background thread."""
    job_id = job["job_id"]
    try:
        pipeline = RGRTrainingPipeline(
            populate_tcr=req.populate_tcr,
            model_version=req.model_version,
        )

        # Determine which combos to train
        if req.train_all:
            from data.extractors import VALID_COMBINATIONS as combos
        elif req.species_abbrev:
            combos = [(req.species_abbrev, None)]
        else:
            combos = [(None, req.forest_unit)]

        with _job_lock:
            job["progress"]["total"] = len(combos)

        for species_abbrev, forest_unit in combos:
            label = species_abbrev or f"FU_{forest_unit}"
            with _job_lock:
                job["progress"]["current"] = label

            try:
                result = pipeline.train_single_model(
                    species_abbrev, forest_unit, save_to_db=req.save_to_db
                )
                with _job_lock:
                    if result:
                        job["results"][label] = result
                        job["progress"]["done"] += 1
                    else:
                        job["errors"][label] = "Insufficient data or query error"
                        job["progress"]["failed"] += 1
            except Exception as exc:
                logger.error("Training failed for %s: %s", label, exc)
                with _job_lock:
                    job["errors"][label] = str(exc)
                    job["progress"]["failed"] += 1

        with _job_lock:
            job["status"]      = "completed"
            job["progress"]["current"] = None
            job["finished_at"] = datetime.now(timezone.utc).isoformat()

        _clear_runtime_caches()

    except Exception as exc:
        logger.error("Training job %s crashed: %s", job_id, exc)
        with _job_lock:
            job["status"]      = "failed"
            job["errors"]["_job"] = str(exc)
            job["finished_at"] = datetime.now(timezone.utc).isoformat()


class PredictRequest(BaseModel):
    spec_code:   int   = Field(...,  description="Species code, e.g. 30 = Hard Maple")
    forest_unit: str   = Field(...,  description="Forest unit, e.g. 'TH1'")
    BA:          float = Field(...,  description="Basal area (m²/ha)")
    AGSfrac:     float = Field(...,  description="AGS fraction (0–1), provide measured value")
    TCR:         int   = Field(5,    description="Timber capability rating (1–7)")
    DQ:          float = Field(...,  description="Quadratic mean diameter (cm)")
    q:           float = Field(...,  description="q-factor")
    Cut:         int   = Field(0,    description="Cutting indicator (0 or 1)")
    easting:     Optional[float] = Field(None, description="Easting (NAD83 EPSG:26917)")
    northing:    Optional[float] = Field(None, description="Northing (NAD83 EPSG:26917)")
    projection_years: int        = Field(0, description="Years to project (0 = single step)")
    model_version:    str        = Field("1.0")


class BatchPredictRequest(BaseModel):
    rows: List[Dict]
    model_version: str = "1.0"


class AutoPredictRequest(BaseModel):
    species_abbrev: str = Field(..., description="Species abbreviation, e.g. Mh, Or, Cedar")
    forest_unit: str = Field(..., description="Forest unit, e.g. TH1, HDSL2")
    current_year: int = Field(..., description="Projection start year")
    projection_years: int = Field(10, description="Number of years to project")
    model_version: str = Field("1.0")


# ---------------------------------------------------------------------------
# Existing endpoints (from DatabaseLinker/main.py logic kept here)
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "BA Growth Tool API is running!"}


@app.get("/test-connection")
def test_connection(db: DbSession):
    """Verify the database connection (SQLite / PostgreSQL / SQL Server)."""
    try:
        version = get_server_version(db)
        return {"status": "success", "dialect": get_dialect(), "sql_version": version}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DB connection failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tables")
def get_tables():
    """List all tables in the database."""
    try:
        return {"tables": list_tables()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Pipeline – metadata
# ---------------------------------------------------------------------------

@app.get("/models/species")
def list_species():
    """Return all supported species configurations."""
    return {"species": SPECIES_CONFIG}


@app.get("/models/forest-units")
def list_forest_units():
    """Return all supported forest units."""
    return {"forest_units": FOREST_UNITS}


@app.get("/models/combinations")
def list_valid_combinations():
    """Return configured model definitions (species models + forest-unit models)."""
    return {"combinations": [{"species": s, "forest_unit": f} for s, f in VALID_COMBINATIONS]}


@app.get("/api/species-list")
def api_species_list():
    species_groups = [_species_display_name(abbrev) for abbrev in SPECIES_CONFIG.keys()]
    forest_unit_groups = [_forest_unit_display_name(forest_unit) for forest_unit in FOREST_UNITS]
    model_groups = species_groups + forest_unit_groups
    mapping = {
        **{_species_display_name(abbrev): abbrev for abbrev in SPECIES_CONFIG.keys()},
        **{_forest_unit_display_name(forest_unit): forest_unit for forest_unit in FOREST_UNITS},
    }
    return {
        "species_groups": model_groups,
        "model_groups": model_groups,
        "forest_units": forest_unit_groups,
        "mapping": mapping,
    }


@app.get("/api/bagrowth/summary-stats")
def api_bagrowth_summary_stats(species_group: str, db: DbSession):
    import pandas as pd

    group = _resolve_model_group(species_group)
    species_abbrev = group["species_abbrev"]
    forest_unit = group["forest_unit"]
    cache_key = _make_cache_key(
        "api_bagrowth_summary_stats",
        species_abbrev=species_abbrev,
        forest_unit=forest_unit,
    )
    cached_response = _get_cached_response(cache_key)
    if cached_response is not None:
        return cached_response

    df = _extract_model_group_data(species_abbrev, forest_unit, db)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for selected species group")

    total_measurements = int(len(df))
    filtered = df.copy()
    if "DQ" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["DQ"], errors="coerce") >= 9]
    filtered = filtered.dropna(subset=["BA", "AGSfrac", "TCR", "DQ", "q"])
    filtered["BA2"] = pd.to_numeric(filtered["BA"], errors="coerce") ** 2
    excluded = total_measurements - int(len(filtered))

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No usable rows found after filtering")

    response_payload = {
        "species_group": group["display"],
        "plots_analyzed": int(filtered["PlotKey"].nunique()) if "PlotKey" in filtered.columns else 0,
        "measurements": int(len(filtered)),
        "excluded_measurements_below_dq_threshold": int(max(0, excluded)),
        "variables": {
            "BA": _summary_stats_for_column(filtered, "BA"),
            "BA2": _summary_stats_for_column(filtered, "BA2"),
            "AGSfrac": _summary_stats_for_column(filtered, "AGSfrac"),
            "TCR": _summary_stats_for_column(filtered, "TCR"),
            "DQ": _summary_stats_for_column(filtered, "DQ"),
            "q": _summary_stats_for_column(filtered, "q"),
            "BAgrowth": _summary_stats_for_column(filtered, "BAgrowth"),
        },
    }

    _set_cached_response(cache_key, response_payload)
    return response_payload


@app.get("/api/bagrowth/formula-calculation")
def api_bagrowth_formula_calculation(species_group: str, db: DbSession, model_version: str = "1.0"):
    import pandas as pd

    group = _resolve_model_group(species_group)
    species_abbrev = group["species_abbrev"]
    forest_unit = group["forest_unit"]
    spec_code = int(SPECIES_CONFIG[species_abbrev]["code"]) if species_abbrev else None
    cache_key = _make_cache_key(
        "api_bagrowth_formula_calculation",
        species_abbrev=species_abbrev,
        forest_unit=forest_unit,
        model_version=model_version,
    )
    cached_response = _get_cached_response(cache_key)
    if cached_response is not None:
        return cached_response

    df = _extract_model_group_data(species_abbrev, forest_unit, db)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for selected species group")

    filtered = df.copy()
    if "DQ" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["DQ"], errors="coerce") >= 9]
    filtered = filtered.dropna(subset=["BA", "AGSfrac", "TCR", "DQ", "q"])

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No usable rows found after filtering")

    cut_growth_floor = _cut_positive_growth_floor(filtered)

    means = {
        "BA": float(pd.to_numeric(filtered["BA"], errors="coerce").mean()),
        "AGSfrac": float(pd.to_numeric(filtered["AGSfrac"], errors="coerce").mean()),
        "TCR": float(pd.to_numeric(filtered["TCR"], errors="coerce").mean()),
        "DQ": float(pd.to_numeric(filtered["DQ"], errors="coerce").mean()),
        "q": float(pd.to_numeric(filtered["q"], errors="coerce").mean()),
    }
    means["BA2"] = means["BA"] ** 2

    use_db_model = True
    try:
        predictor = RGRPredictor(db, model_version=model_version)
        model_info = _model_info_for_group(spec_code, forest_unit, model_version=model_version, db=db)
    except ValueError:
        if forest_unit is not None:
            raise HTTPException(status_code=404, detail=f"No trained coefficients found for forest unit '{forest_unit}'. Train FU model first.")
        predictor = None
        model_info = _load_fallback_coefficients()
        use_db_model = False

    tcr_rounded = int(max(1, min(7, round(means["TCR"]))))

    if use_db_model and predictor is not None:
        without_cut_value = float(
            predictor.predict_ba_growth(
                spec_code,
                    forest_unit,
                means["BA"],
                means["AGSfrac"],
                tcr_rounded,
                means["DQ"],
                means["q"],
                0,
            )
        )
        with_cut_value = float(
            predictor.predict_ba_growth(
                spec_code,
                    forest_unit,
                means["BA"],
                means["AGSfrac"],
                tcr_rounded,
                means["DQ"],
                means["q"],
                1,
            )
        )
    else:
        without_cut_rgr = _predict_rgr_from_coefficients(
            model_info,
            means["BA"],
            means["AGSfrac"],
            tcr_rounded,
            means["DQ"],
            means["q"],
            0,
        )
        with_cut_rgr = _predict_rgr_from_coefficients(
            model_info,
            means["BA"],
            means["AGSfrac"],
            tcr_rounded,
            means["DQ"],
            means["q"],
            1,
        )
        without_cut_value = float(without_cut_rgr * means["BA"])
        with_cut_value = float(with_cut_rgr * means["BA"])

    without_cut_value, _ = _non_negative_growth(without_cut_value, 0.0)
    with_cut_value, _ = _apply_cut_growth_floor(with_cut_value, 0.0, 1, means["BA"], cut_growth_floor)

    response_payload = {
        "species_group": group["display"],
        "formula": "BAgrowth = β₀ + β₁·BA + β₂·BA² + β₃·AGSfrac + β₄·TCR + β₅·DQ + β₆·q + β₇·Cut",
        "formula_explanation": "Formula shown with BA² term for parity with legacy UI; current trained model uses linear BA and computes BAgrowth via RGR × BA.",
        "inputs": {
            "mean_BA": means["BA"],
            "mean_BA2": means["BA2"],
            "mean_AGSfrac": means["AGSfrac"],
            "mean_TCR": means["TCR"],
            "mean_DQ": means["DQ"],
            "mean_q": means["q"],
        },
        "coefficients": {
            "intercept": float(model_info["intercept"]),
            "BA": float(model_info["coefficients"]["BA"]),
            "BA2": float(model_info["coefficients"]["BA2"]),
            "AGSfrac": float(model_info["coefficients"]["AGSfrac"]),
            "TCR": float(model_info["coefficients"]["TCR"]),
            "DQ": float(model_info["coefficients"]["DQ"]),
            "q": float(model_info["coefficients"]["q"]),
            "Cut": float(model_info["coefficients"]["Cut"]),
        },
        "scenarios": {
            "without_cut": {
                "cut_value": 0,
                "label": "Without Cut",
                "description": "No harvesting/treatment",
                "calculated_bagrowth": without_cut_value,
            },
            "with_cut": {
                "cut_value": 1,
                "label": "With Cut",
                "description": "Harvest/treatment",
                "calculated_bagrowth": with_cut_value,
            },
        },
        "measurements_used": int(len(filtered)),
        "plots_used": int(filtered["PlotKey"].nunique()) if "PlotKey" in filtered.columns else 0,
        "excluded_measurements_below_dq_threshold": int(max(0, len(df) - len(filtered))),
    }

    _set_cached_response(cache_key, response_payload)
    return response_payload


@app.get("/api/model-coefficients")
def api_model_coefficients(db: DbSession, species_group: Optional[str] = None, model_version: str = "1.0"):
    """Return coefficients strictly from permanent tblRGRCoefficients."""

    try:
        store = CoefficientStore(db)
        store.ensure_coefficients_table_exists()

        if species_group:
            group = _resolve_model_group(species_group)
            species_abbrev = group["species_abbrev"]
            forest_unit = group["forest_unit"]
            spec_code = int(SPECIES_CONFIG[species_abbrev]["code"]) if species_abbrev else None
        else:
            latest = store.get_latest_model_selector(model_version=model_version)
            if latest is None:
                raise HTTPException(
                    status_code=404,
                    detail="No coefficients found in tblRGRCoefficients. Train a model first.",
                )

            spec_code = latest["spec_code"]
            forest_unit = latest["forest_unit"]
            species_abbrev = _SPEC_CODE_TO_ABBREV.get(int(spec_code)) if spec_code is not None else None
            if forest_unit:
                group = {
                    "display": _forest_unit_display_name(forest_unit),
                    "species_abbrev": None,
                    "forest_unit": forest_unit,
                }
            elif species_abbrev:
                group = {
                    "display": _species_display_name(species_abbrev),
                    "species_abbrev": species_abbrev,
                    "forest_unit": None,
                }
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Latest coefficients row does not map to a known species/forest unit.",
                )

        coefs = store.load_coefficients(spec_code, forest_unit, model_version=model_version)
        meta = store.get_model_info(spec_code, forest_unit, model_version=model_version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "species_group": group["display"],
        "source": "tblRGRCoefficients",
        "model_version": model_version,
        "intercept": float(coefs["intercept"]),
        "coefficients": {
            "BA": float(coefs["BA"]),
            "BA2": float(coefs.get("BA2", 0.0)),
            "AGSfrac": float(coefs["AGSfrac"]),
            "TCR": float(coefs["TCR"]),
            "DQ": float(coefs["DQ"]),
            "q": float(coefs["q"]),
            "Cut": float(coefs["Cut"]),
        },
        "model_info": {
            "n_samples": int(coefs["n_samples"]) if coefs.get("n_samples") is not None else None,
            "r2_score": float(coefs["r_squared"]) if coefs.get("r_squared") is not None else None,
            "rmse": float(coefs["rmse"]) if coefs.get("rmse") is not None else None,
            "model_type": "RGR linear model",
            "training_date": meta["trained_date"] if meta and meta.get("trained_date") else None,
            "status": "trained",
            "coefficient_id": int(meta["coefficient_id"]) if meta and meta.get("coefficient_id") is not None else None,
        },
    }


@app.get("/api/bagrowth/calculate")
def api_bagrowth_calculate(
    species_group: str,
    current_year: int,
    db: DbSession,
    projection_years: int = 10,
    model_version: str = "1.0",
    refresh: bool = False,
):
    import pandas as pd

    start_time = time.time()
    group = _resolve_model_group(species_group)
    species_abbrev = group["species_abbrev"]
    forest_unit = group["forest_unit"]
    spec_code = int(SPECIES_CONFIG[species_abbrev]["code"]) if species_abbrev else None
    cache_key = _make_cache_key(
        "api_bagrowth_calculate",
        species_abbrev=species_abbrev,
        forest_unit=forest_unit,
        current_year=int(current_year),
        projection_years=int(projection_years),
        model_version=model_version,
    )
    if not refresh:
        cached_response = _get_cached_response(cache_key)
        if cached_response is not None:
            if "pipeline_metrics" in cached_response:
                cached_response["pipeline_metrics"]["cache_reason"] = "API response cache hit"
                cached_response["pipeline_metrics"]["elapsed_seconds"] = round(time.time() - start_time, 3)
            return cached_response

    df = _extract_model_group_data(species_abbrev, forest_unit, db)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for selected species group")

    filtered = df.copy()
    if "DQ" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["DQ"], errors="coerce") >= 9]
    filtered = filtered.dropna(subset=["BA", "AGSfrac", "TCR", "DQ", "q"])

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No usable rows found after filtering")

    cut_growth_floor = _cut_positive_growth_floor(filtered)

    use_db_model = True
    try:
        predictor = RGRPredictor(db, model_version=model_version)
        model_info = _model_info_for_group(spec_code, forest_unit, model_version=model_version, db=db)
    except ValueError:
        if forest_unit is not None:
            raise HTTPException(status_code=404, detail=f"No trained coefficients found for forest unit '{forest_unit}'. Train FU model first.")
        predictor = None
        model_info = _load_fallback_coefficients()
        use_db_model = False

    per_plot_results: List[Dict[str, Any]] = []
    growth_values: List[float] = []
    rgr_values: List[float] = []

    latest_idx = filtered.groupby("PlotKey")["EndYear"].idxmax() if "PlotKey" in filtered.columns else filtered.index
    latest_rows = filtered.loc[latest_idx]

    per_plot_db_rows: List[Dict[str, Any]] = []
    for _, row in latest_rows.iterrows():
        ba_current = float(row["BA"])
        ags = float(row["AGSfrac"])
        tcr = int(max(1, min(7, round(float(row["TCR"])))) )
        dq = float(row["DQ"])
        q_val = float(row["q"])
        cut = int(row["Cut"]) if pd.notna(row.get("Cut")) else 0

        if use_db_model and predictor is not None:
            rgr_pred = float(predictor.predict_rgr(spec_code, forest_unit, ba_current, ags, tcr, dq, q_val, cut))
        else:
            rgr_pred = _predict_rgr_from_coefficients(model_info, ba_current, ags, tcr, dq, q_val, cut)
        ba_growth_pred = float(rgr_pred * ba_current)
        ba_growth_pred, rgr_pred = _apply_cut_growth_floor(ba_growth_pred, rgr_pred, cut, ba_current, cut_growth_floor)

        per_plot_results.append(
            {
                "PlotKey": str(row["PlotKey"]),
                "BA_current": ba_current,
                "BAgrowth_predicted": ba_growth_pred,
                "RGR_predicted": rgr_pred,
            }
        )
        per_plot_db_rows.append(
            {
                "PlotKey": str(row.get("PlotKey")) if pd.notna(row.get("PlotKey")) else None,
                "StartYear": int(row["StartYear"]) if pd.notna(row.get("StartYear")) else None,
                "EndYear": int(row["EndYear"]) if pd.notna(row.get("EndYear")) else None,
                "GrowthInterval_Years": float(row["GrowthInterval_Years"]) if pd.notna(row.get("GrowthInterval_Years")) else None,
                "BA": ba_current,
                "AGSfrac": ags,
                "TCR": tcr,
                "DQ": dq,
                "q": q_val,
                "Cut": cut,
                "BAgrowth_Input": float(row["BAgrowth"]) if pd.notna(row.get("BAgrowth")) else None,
                "RGR_Input": float(row["RGR"]) if pd.notna(row.get("RGR")) else None,
                "RGR_Predicted": rgr_pred,
                "BAgrowth_Predicted": ba_growth_pred,
            }
        )
        growth_values.append(ba_growth_pred)
        rgr_values.append(rgr_pred)

    # Persist fetched calculation inputs and per-plot calculation rows for frontend auditability
    try:
        payload_cols = [
            col for col in [
                "PlotKey", "StartYear", "EndYear", "GrowthInterval_Years",
                "BA", "AGSfrac", "TCR", "DQ", "q", "Cut", "BAgrowth", "RGR"
            ] if col in filtered.columns
        ]
        payload = {
            "filtered_rows_data": filtered[payload_cols].to_dict(orient="records"),
            "latest_rows_data": latest_rows[payload_cols].to_dict(orient="records"),
        }
        store = CoefficientStore(db)
        frontend_calc_id = store.save_frontend_calc_data(
            spec_code=spec_code,
            coefficient_id=(int(model_info["coefficient_id"]) if use_db_model and model_info.get("coefficient_id") is not None else None),
            species_abbrev=species_abbrev,
            forest_unit=forest_unit,
            model_version=model_version,
            current_year=int(current_year),
            projection_years=int(projection_years),
            extracted_rows=int(len(df)),
            filtered_rows=int(len(filtered)),
            latest_rows=int(len(latest_rows)),
            used_db_model=use_db_model,
            data_payload=payload,
        )
        store.save_frontend_calc_plot_rows(frontend_calc_id, per_plot_db_rows)
    except Exception as exc:
        db.rollback()
        logger.warning("Failed to persist frontend calculation input snapshot: %s", exc)

    n_plots = len(per_plot_results)
    mean_growth = float(sum(growth_values) / n_plots) if n_plots else 0.0
    mean_rgr = float(sum(rgr_values) / n_plots) if n_plots else 0.0
    mean_ba_current = float(pd.to_numeric(latest_rows["BA"], errors="coerce").mean())

    std_growth = float(pd.Series(growth_values).std(ddof=1)) if n_plots > 1 else 0.0
    stderr = (std_growth / math.sqrt(n_plots)) if n_plots > 0 else 0.0
    ci_lower = mean_growth - 1.96 * stderr
    ci_upper = mean_growth + 1.96 * stderr

    mean_inputs = {
        "BA": mean_ba_current,
        "AGSfrac": float(pd.to_numeric(latest_rows["AGSfrac"], errors="coerce").mean()),
        "TCR": int(max(1, min(7, round(float(pd.to_numeric(latest_rows["TCR"], errors="coerce").mean()))))),
        "DQ": float(pd.to_numeric(latest_rows["DQ"], errors="coerce").mean()),
        "q": float(pd.to_numeric(latest_rows["q"], errors="coerce").mean()),
        "Cut": int(round(float(pd.to_numeric(latest_rows["Cut"], errors="coerce").fillna(0).mean()))),
    }
    mean_inputs["Cut"] = 1 if mean_inputs["Cut"] >= 1 else 0

    yearly_projection: List[Dict[str, Any]] = []
    ba_current = float(mean_inputs["BA"])
    yearly_projection.append(
        {
            "year": int(current_year),
            "BA_predicted": ba_current,
            "BAgrowth_predicted": 0.0,
            "RGR": 0.0,
        }
    )

    for year_offset in range(1, max(1, int(projection_years)) + 1):
        if use_db_model and predictor is not None:
            rgr_projection = float(
                predictor.predict_rgr(
                    spec_code,
                    forest_unit,
                    ba_current,
                    float(mean_inputs["AGSfrac"]),
                    int(mean_inputs["TCR"]),
                    float(mean_inputs["DQ"]),
                    float(mean_inputs["q"]),
                    int(mean_inputs["Cut"]),
                )
            )
        else:
            rgr_projection = _predict_rgr_from_coefficients(
                model_info,
                ba_current,
                float(mean_inputs["AGSfrac"]),
                int(mean_inputs["TCR"]),
                float(mean_inputs["DQ"]),
                float(mean_inputs["q"]),
                int(mean_inputs["Cut"]),
            )

        ba_growth_projection, rgr_projection = _apply_cut_growth_floor(
            rgr_projection * ba_current,
            rgr_projection,
            int(mean_inputs["Cut"]),
            ba_current,
            cut_growth_floor,
        )
        ba_current = ba_current + ba_growth_projection

        yearly_projection.append(
            {
                "year": int(current_year) + year_offset,
                "BA_predicted": float(ba_current),
                "BAgrowth_predicted": float(ba_growth_projection),
                "RGR": float(rgr_projection),
            }
        )

    elapsed_seconds = round(time.time() - start_time, 3)

    response_payload = {
        "species_group": group["display"],
        "mean_BAgrowth": mean_growth,
        "mean_RGR": mean_rgr,
        "std_BAgrowth": std_growth,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_plots": n_plots,
        "mean_BA_current": mean_ba_current,
        "per_plot_results": per_plot_results,
        "filtered_rows_data": filtered[payload_cols].to_dict(orient="records") if "payload_cols" in locals() else [],
        "latest_rows_data": latest_rows[payload_cols].to_dict(orient="records") if "payload_cols" in locals() else [],
        "outlier_analysis": {
            "method": "outside_mean_plus_minus_2sd",
            "BA":       _two_sigma_outlier_stats(filtered, "BA"),
            "AGSfrac":  _two_sigma_outlier_stats(filtered, "AGSfrac"),
            "TCR":      _two_sigma_outlier_stats(filtered, "TCR"),
            "DQ":       _two_sigma_outlier_stats(filtered, "DQ"),
            "q":        _two_sigma_outlier_stats(filtered, "q"),
            "BAgrowth": _two_sigma_outlier_stats(filtered, "BAgrowth"),
            "RGR":      _two_sigma_outlier_stats(filtered, "RGR"),
        },
        "yearly_projection": yearly_projection,
        "model_info": model_info,
        "pipeline_metrics": {
            "training_skipped": True,
            "cache_reason": "Using stored coefficients" if use_db_model else "Using fallback artifact coefficients",
            "elapsed_seconds": elapsed_seconds,
        },
    }

    if not refresh:
        _set_cached_response(cache_key, response_payload)
    return response_payload


@app.get("/models/stored")
def list_stored_models(db: DbSession):
    """List all coefficient records stored in the database."""
    try:
        import numpy as np
        import pandas as pd
        predictor = RGRPredictor(db)
        df = predictor.list_available_models()
        safe_df = df.replace([np.inf, -np.inf], np.nan)
        safe_df = safe_df.astype(object).where(pd.notna(safe_df), None)
        return {"models": safe_df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/models/training-results")
def get_training_results():
    """Return the latest training results JSON from artifacts/reports."""
    reports_dir = os.path.join(_BACKEND_DIR, "artifacts", "reports")
    if not os.path.isdir(reports_dir):
        raise HTTPException(status_code=404, detail="Training reports directory not found")

    matches = glob.glob(os.path.join(reports_dir, "training_results_*.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="No training report files found")

    latest_report = max(matches, key=os.path.getmtime)
    try:
        with open(latest_report, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return {
            "report_file": os.path.basename(latest_report),
            "training_results": payload,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Pipeline – training
# ---------------------------------------------------------------------------

@app.post("/models/train", status_code=202)
def train_models(req: TrainRequest):
    """
    Start a training job asynchronously.

    Returns immediately with a job_id.  Poll GET /models/training-status/{job_id}
    to track progress.  Training 13 models with full historical data (1977-2024)
    takes 1-2 hours; this endpoint never blocks.
    """
    # Validate request before spawning the thread
    if not req.train_all:
        if bool(req.species_abbrev) == bool(req.forest_unit):
            raise HTTPException(
                status_code=400,
                detail="Provide exactly one of species_abbrev or forest_unit, or set train_all=true.",
            )

    job = _make_job(req)
    thread = threading.Thread(
        target=_run_training_job,
        args=(job, req),
        daemon=True,
        name=f"training-{job['job_id'][:8]}",
    )
    thread.start()

    return {
        "job_id":    job["job_id"],
        "status":    "running",
        "message":   "Training started in background. Poll /models/training-status/{job_id} for progress.",
        "poll_url":  f"/models/training-status/{job['job_id']}",
    }


@app.get("/models/training-status")
def list_training_jobs():
    """Return summary of all training jobs (running and completed)."""
    with _job_lock:
        return {
            "jobs": [
                {
                    "job_id":      j["job_id"],
                    "kind":        j.get("kind", "training"),
                    "status":      j["status"],
                    "started_at":  j["started_at"],
                    "finished_at": j["finished_at"],
                    "progress":    j["progress"],
                }
                for j in _job_registry.values()
            ]
        }


@app.get("/models/training-status/{job_id}")
def get_training_status(job_id: str):
    """Poll detailed status for a specific training job."""
    with _job_lock:
        job = _job_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    with _job_lock:
        return dict(job)



# ---------------------------------------------------------------------------
# Pipeline – prediction
# ---------------------------------------------------------------------------

@app.post("/models/predict")
def predict(req: PredictRequest, db: DbSession):
    """
    Predict RGR and BA growth for a single observation.

    If easting/northing provided, TCR is resolved from the spatial shapefile.
    If projection_years > 0, a multi-year BA projection is returned.
    """
    try:
        predictor = RGRPredictor(db, model_version=req.model_version)

        # Resolve TCR from coordinates when supplied
        tcr = req.TCR
        if req.easting is not None and req.northing is not None:
            from data.extractors import _get_tcr
            tcr = _get_tcr(req.easting, req.northing)

        rgr       = predictor.predict_rgr(
            req.spec_code, req.forest_unit,
            req.BA, req.AGSfrac, tcr, req.DQ, req.q, req.Cut,
        )
        ba_growth, rgr = _non_negative_growth(rgr * req.BA, rgr)

        response: Dict = {
            "spec_code":   req.spec_code,
            "forest_unit": req.forest_unit,
            "tcr_used":    tcr,
            "RGR":         rgr,
            "BAgrowth":    ba_growth,
        }

        if req.projection_years > 0:
            projection_rows: List[Dict[str, Any]] = [{"Year": 0, "BA": float(req.BA), "BAgrowth": 0.0, "RGR": 0.0}]
            ba_current = float(req.BA)
            for year_offset in range(1, max(1, int(req.projection_years)) + 1):
                proj_rgr = float(
                    predictor.predict_rgr(
                        req.spec_code,
                        req.forest_unit,
                        ba_current,
                        req.AGSfrac,
                        tcr,
                        req.DQ,
                        req.q,
                        req.Cut,
                    )
                )
                proj_growth, proj_rgr = _non_negative_growth(proj_rgr * ba_current, proj_rgr)
                ba_current += proj_growth
                projection_rows.append({"Year": year_offset, "BA": ba_current, "BAgrowth": proj_growth, "RGR": proj_rgr})
            response["projection"] = projection_rows

        return response

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/models/predict/batch")
def predict_batch(req: BatchPredictRequest, db: DbSession):
    """
    Batch prediction endpoint.  Expects a list of dicts each containing:
    SpecCode, ForestUnit, BA, AGSfrac, TCR, DQ, q, Cut.
    """
    import pandas as pd

    try:
        df        = pd.DataFrame(req.rows)
        predictor = RGRPredictor(db, model_version=req.model_version)
        result_df = predictor.batch_predict(df)
        return {"predictions": result_df.to_dict(orient="records")}
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/models/auto-predict")
def auto_predict(req: AutoPredictRequest, db: DbSession):
    """
    Automatically derive representative model inputs from extracted species data,
    run predictions, and return summary + yearly projection + per-plot estimates.
    """
    import math
    import pandas as pd

    try:
        if req.species_abbrev not in SPECIES_CONFIG:
            raise HTTPException(status_code=400, detail=f"Unknown species_abbrev '{req.species_abbrev}'")
        if req.forest_unit not in FOREST_UNITS:
            raise HTTPException(status_code=400, detail=f"Unknown forest_unit '{req.forest_unit}'")

        extractor = BAGrowthDataExtractor(db, populate_tcr=True)
        df = extractor.extract_training_data(req.species_abbrev, None, since_year=0)

        if df.empty:
            raise HTTPException(status_code=404, detail="No training data available for selected species")

        latest_idx = df.groupby("PlotKey")["EndYear"].idxmax()
        latest = df.loc[latest_idx].copy()
        latest = latest.dropna(subset=["BA", "AGSfrac", "TCR", "DQ", "q"]) 

        if latest.empty:
            raise HTTPException(status_code=404, detail="No usable rows found after filtering missing values")

        cut_growth_floor = _cut_positive_growth_floor(latest)

        predictor = RGRPredictor(db, model_version=req.model_version)
        spec_code = int(SPECIES_CONFIG[req.species_abbrev]["code"])

        per_plot_records: List[Dict[str, Any]] = []
        growth_values: List[float] = []
        rgr_values: List[float] = []

        for _, row in latest.iterrows():
            ba_current = float(row["BA"])
            ags = float(row["AGSfrac"])
            tcr = int(row["TCR"])
            dq = float(row["DQ"])
            q_val = float(row["q"])
            cut = int(row["Cut"]) if pd.notna(row.get("Cut")) else 0

            rgr_pred = predictor.predict_rgr(
                spec_code,
                req.forest_unit,
                ba_current,
                ags,
                tcr,
                dq,
                q_val,
                cut,
            )
            ba_growth_pred, rgr_pred = _apply_cut_growth_floor(
                rgr_pred * ba_current,
                rgr_pred,
                cut,
                ba_current,
                cut_growth_floor,
            )

            per_plot_records.append(
                {
                    "PlotKey": str(row["PlotKey"]),
                    "BA_current": ba_current,
                    "BAgrowth_predicted": ba_growth_pred,
                    "RGR_predicted": rgr_pred,
                }
            )
            growth_values.append(ba_growth_pred)
            rgr_values.append(rgr_pred)

        n_plots = len(per_plot_records)
        mean_growth = float(sum(growth_values) / n_plots)
        mean_rgr = float(sum(rgr_values) / n_plots)
        mean_ba_current = float(latest["BA"].mean())

        if n_plots > 1:
            std_growth = float(pd.Series(growth_values).std(ddof=1))
        else:
            std_growth = 0.0

        stderr = (std_growth / math.sqrt(n_plots)) if n_plots > 0 else 0.0
        ci_lower = mean_growth - 1.96 * stderr
        ci_upper = mean_growth + 1.96 * stderr

        mean_inputs = {
            "BA": mean_ba_current,
            "AGSfrac": float(latest["AGSfrac"].mean()),
            "TCR": int(round(float(latest["TCR"].mean()))),
            "DQ": float(latest["DQ"].mean()),
            "q": float(latest["q"].mean()),
            "Cut": int(round(float(latest["Cut"].fillna(0).mean()))),
        }
        mean_inputs["TCR"] = max(1, min(7, mean_inputs["TCR"]))
        mean_inputs["Cut"] = 1 if mean_inputs["Cut"] >= 1 else 0

        yearly_projection: List[Dict[str, Any]] = []
        ba_projection = float(mean_inputs["BA"])
        yearly_projection.append(
            {
                "year": int(req.current_year),
                "BA_predicted": ba_projection,
                "BAgrowth_predicted": 0.0,
                "RGR": 0.0,
            }
        )
        for year_offset in range(1, max(1, int(req.projection_years)) + 1):
            proj_rgr = float(
                predictor.predict_rgr(
                    spec_code,
                    req.forest_unit,
                    ba_projection,
                    mean_inputs["AGSfrac"],
                    mean_inputs["TCR"],
                    mean_inputs["DQ"],
                    mean_inputs["q"],
                    mean_inputs["Cut"],
                )
            )
            proj_growth, proj_rgr = _apply_cut_growth_floor(
                proj_rgr * ba_projection,
                proj_rgr,
                int(mean_inputs["Cut"]),
                ba_projection,
                cut_growth_floor,
            )
            ba_projection += proj_growth
            yearly_projection.append(
                {
                    "year": int(req.current_year) + year_offset,
                    "BA_predicted": float(ba_projection),
                    "BAgrowth_predicted": float(proj_growth),
                    "RGR": float(proj_rgr),
                }
            )

        return {
            "species_abbrev": req.species_abbrev,
            "species_name": SPECIES_CONFIG[req.species_abbrev]["name"],
            "forest_unit": req.forest_unit,
            "current_year": req.current_year,
            "projection_years": req.projection_years,
            "auto_inputs": mean_inputs,
            "n_plots": n_plots,
            "n_measurements": int(len(df)),
            "mean_BA_current": mean_ba_current,
            "mean_BAgrowth": mean_growth,
            "mean_RGR": mean_rgr,
            "std_BAgrowth": std_growth,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "yearly_projection": yearly_projection,
            "per_plot_results": per_plot_records,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Auto prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Model benchmarking (backend/ml)
# ---------------------------------------------------------------------------

from ml.benchmark import BenchmarkConfig, benchmark_all  # noqa: E402
from ml.evaluation import CV_STRATEGIES  # noqa: E402
from ml.features import feature_set_names  # noqa: E402
from ml.registry import DEFAULT_MODELS, MODEL_SPECS, list_models  # noqa: E402
from ml.reports import BENCHMARK_DIR as _BENCHMARK_DIR_DEFAULT  # noqa: E402
from ml.reports import latest_json, load_json, write_json, write_markdown  # noqa: E402

_BENCHMARK_DIR = _BENCHMARK_DIR_DEFAULT


class BenchmarkRequest(BaseModel):
    groups:         Optional[List[str]] = Field(None, description="Model groups (e.g. ['Mh','FU_TH1']); default = all")
    models:         Optional[List[str]] = Field(None, description="Algorithm names from /api/benchmarks/models; default = all available")
    cv_strategy:    str   = Field("group", description="group | temporal | kfold")
    n_splits:       int   = Field(5, ge=2, le=20)
    seed:           int   = Field(42)
    feature_set:    str   = Field("base", description="base | extended")
    alpha:          float = Field(0.10, gt=0.0, lt=0.5, description="Prediction-interval miscoverage")
    leakage_check:  bool  = Field(True)
    temporal_check: bool  = Field(True)
    intervals:      bool  = Field(True)
    importance:     bool  = Field(True)
    populate_tcr:   bool  = Field(False, description="Override TCR from the shapefile during extraction")


def _benchmark_group_keys() -> List[str]:
    return [sp for sp in SPECIES_CONFIG] + [f"FU_{fu}" for fu in FOREST_UNITS]


def _resolve_benchmark_groups(keys: Optional[List[str]]) -> List[Dict[str, Optional[str]]]:
    if not keys:
        keys = _benchmark_group_keys()
    resolved = []
    for key in keys:
        group = _resolve_model_group(key)
        group["key"] = group["species_abbrev"] or f"FU_{group['forest_unit']}"
        resolved.append(group)
    return resolved


def _run_benchmark_job(job: Dict, req: BenchmarkRequest, groups: List[Dict[str, Optional[str]]]) -> None:
    from db.engine import get_session

    job_id = job["job_id"]
    try:
        session = get_session()
        try:
            extractor = BAGrowthDataExtractor(session, populate_tcr=req.populate_tcr)
            data = {}
            for group in groups:
                try:
                    df = extractor.extract_training_data(group["species_abbrev"], group["forest_unit"])
                except Exception as exc:  # noqa: BLE001
                    with _job_lock:
                        job["errors"][group["key"]] = f"extraction failed: {exc}"
                    continue
                if not df.empty:
                    data[group["key"]] = df
        finally:
            session.close()

        with _job_lock:
            job["progress"]["total"] = len(data)

        def _progress(group_key: str, stage: str) -> None:
            with _job_lock:
                job["progress"]["current"] = f"{group_key}: {stage}"

        config = BenchmarkConfig(
            models=req.models or list(DEFAULT_MODELS),
            cv_strategy=req.cv_strategy,
            n_splits=req.n_splits,
            seed=req.seed,
            feature_set=req.feature_set,
            alpha=req.alpha,
            leakage_check=req.leakage_check,
            temporal_check=req.temporal_check,
            intervals=req.intervals,
            importance=req.importance,
        )
        report = benchmark_all(data, config, progress=_progress)
        json_path = write_json(report, out_dir=_BENCHMARK_DIR)
        md_path = write_markdown(report, out_dir=_BENCHMARK_DIR)

        with _job_lock:
            job["status"] = "completed"
            job["progress"]["done"] = len(report["groups"])
            job["progress"]["failed"] = sum(1 for g in report["groups"].values() if g.get("error"))
            job["progress"]["current"] = None
            job["results"] = {
                "report_file": os.path.basename(json_path),
                "markdown_file": os.path.basename(md_path),
                "summary": report["summary"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:  # noqa: BLE001
        logger.error("Benchmark job %s crashed: %s", job_id, exc)
        with _job_lock:
            job["status"] = "failed"
            job["errors"]["_job"] = str(exc)
            job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/api/benchmarks/models")
def benchmark_registry():
    """Candidate algorithms, CV strategies and feature sets the benchmark understands."""
    return {
        "models": list_models(),
        "default_models": list(DEFAULT_MODELS),
        "cv_strategies": list(CV_STRATEGIES),
        "feature_sets": feature_set_names(),
        "groups": _benchmark_group_keys(),
    }


@app.get("/api/benchmarks")
def benchmark_latest():
    """Latest benchmark report (JSON written by scripts/benchmark_models.py or POST /api/benchmarks/run)."""
    path = latest_json(_BENCHMARK_DIR)
    if path is None:
        raise HTTPException(status_code=404, detail="No benchmark report found. Run POST /api/benchmarks/run "
                                                    "or python scripts/benchmark_models.py --all")
    return {"report_file": os.path.basename(path), "report": load_json(path)}


@app.get("/api/benchmarks/{model_group}")
def benchmark_for_group(model_group: str):
    """One model group's section of the latest benchmark report."""
    path = latest_json(_BENCHMARK_DIR)
    if path is None:
        raise HTTPException(status_code=404, detail="No benchmark report found.")
    report = load_json(path)
    try:
        group = _resolve_model_group(model_group)
    except HTTPException:
        raise HTTPException(status_code=404, detail=f"Unknown model group '{model_group}'")
    key = group["species_abbrev"] or f"FU_{group['forest_unit']}"
    section = report.get("groups", {}).get(key)
    if section is None:
        raise HTTPException(status_code=404, detail=f"Group '{key}' is not in the latest report ({os.path.basename(path)})")
    return {"report_file": os.path.basename(path), "generated_at": report.get("generated_at"), **section}


@app.post("/api/benchmarks/run", status_code=202)
def benchmark_run(req: BenchmarkRequest):
    """
    Start a benchmark job in the background.  Poll /models/training-status/{job_id};
    when it completes, GET /api/benchmarks returns the new leaderboard.
    """
    if req.cv_strategy not in CV_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"cv_strategy must be one of {list(CV_STRATEGIES)}")
    if req.feature_set not in feature_set_names():
        raise HTTPException(status_code=400, detail=f"feature_set must be one of {feature_set_names()}")
    unknown = [m for m in (req.models or []) if m not in MODEL_SPECS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown model(s): {unknown}. See /api/benchmarks/models")
    groups = _resolve_benchmark_groups(req.groups)

    job_id = str(uuid.uuid4())
    job: Dict[str, Any] = {
        "job_id": job_id,
        "kind": "benchmark",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "request": req.model_dump(),
        "progress": {"total": 0, "done": 0, "failed": 0, "current": "extracting data"},
        "results": {},
        "errors": {},
    }
    with _job_lock:
        _job_registry[job_id] = job

    threading.Thread(target=_run_benchmark_job, args=(job, req, groups), daemon=True,
                     name=f"benchmark-{job_id[:8]}").start()
    return {
        "job_id": job_id,
        "status": "running",
        "message": "Benchmark started in background.",
        "poll_url": f"/models/training-status/{job_id}",
        "result_url": "/api/benchmarks",
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
