from __future__ import annotations

import json
import mimetypes
import os
import traceback
from http import HTTPStatus
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parents[2]
CACHE_DIR = REPO_ROOT / "models" / "cache"
FRONTEND_DIST_DIR = BACKEND_DIR.parent / "frontend" / "dist"
DATA_PATH_CANDIDATES = (
    REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_engineered.csv",
    REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_cleaned.csv",
    REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_train.csv",
)
MODEL_ARTIFACT_BUNDLES = (
    {
        "label": "cascade_xgboost",
        "sieve": REPO_ROOT / "models" / "sieve_model.pkl",
        "verifier": REPO_ROOT / "models" / "verifier_model.pkl",
        "features": REPO_ROOT / "models" / "features_list.pkl",
        "config": REPO_ROOT / "models" / "cascade_config.json",
    },
)

_CACHED_DF: pd.DataFrame | None = None
_CACHED_SIEVE: Any | None = None
_CACHED_VERIFIER: Any | None = None
_CACHED_FEATURES: list[str] | None = None
_CACHED_THRESHOLD: float | None = None
_SUMMARY_CACHE: dict[str, Any] | None = None
_SCHEMA_CACHE: dict[str, Any] | None = None
_DATASET_COLUMNS_CACHE: dict[Path, set[str]] = {}
_RESOLVED_DATA_PATH: Path | None = None
_RESOLVED_ARTIFACT_BUNDLE: dict[str, Any] | None = None

PATIENT_CONTEXT_FIELDS = {"Age", "Gender", "Unit1"}
VITAL_SIGN_FIELDS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"}

FIELD_LABELS = {
    "HR": "Heart Rate", "O2Sat": "Oxygen Saturation", "Temp": "Temperature",
    "SBP": "Systolic Blood Pressure", "MAP": "Mean Arterial Pressure",
    "DBP": "Diastolic Blood Pressure", "Resp": "Respiratory Rate",
    "HCO3": "Bicarbonate", "FiO2": "Fraction of Inspired Oxygen",
    "pH": "Blood pH", "PaCO2": "Arterial CO2", "BUN": "Blood Urea Nitrogen",
    "Calcium": "Calcium", "Chloride": "Chloride", "Creatinine": "Creatinine",
    "Glucose": "Glucose", "Lactate": "Lactate", "Magnesium": "Magnesium",
    "Phosphate": "Phosphate", "Potassium": "Potassium", "Hct": "Hematocrit",
    "Hgb": "Hemoglobin", "WBC": "White Blood Cell Count", "Platelets": "Platelets",
    "Age": "Age", "Gender": "Gender", "Unit1": "Unit1 Indicator",
    "Unspecified_ICU_Type": "ICU Admission Source Unspecified"
}
FIELD_UNITS = {
    "HR": "bpm", "O2Sat": "%", "Temp": "deg C", "SBP": "mmHg", "MAP": "mmHg",
    "DBP": "mmHg", "Resp": "breaths/min", "HCO3": "mmol/L", "FiO2": "fraction",
    "pH": "pH", "PaCO2": "mmHg", "BUN": "mg/dL", "Calcium": "mg/dL",
    "Chloride": "mmol/L", "Creatinine": "mg/dL", "Glucose": "mg/dL",
    "Lactate": "mmol/L", "Magnesium": "mg/dL", "Phosphate": "mg/dL",
    "Potassium": "mmol/L", "Hct": "%", "Hgb": "g/dL", "WBC": "10^9/L",
    "Platelets": "10^9/L", "Age": "years", "Unspecified_ICU_Type": "Binary"
}

INPUT_GROUPS = [
    {"id": "patient_context", "label": "Patient Context", "description": "Demographics and ICU admission context."},
    {"id": "vital_signs", "label": "Vital Signs", "description": "Bedside hemodynamic and respiratory measurements."},
    {"id": "lab_markers", "label": "Laboratory Markers", "description": "Chemistry and hematology values."}
]

class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status

def _log(message: str) -> None:
    print(f"[SEPSIS_BACKEND] {message}", flush=True)

def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ApiError(f"{label} not found at '{path}'.", status=HTTPStatus.INTERNAL_SERVER_ERROR)

def _find_artifact_bundle() -> dict[str, Any] | None:
    for bundle in MODEL_ARTIFACT_BUNDLES:
        if bundle["sieve"].exists() and bundle["verifier"].exists() and \
           bundle["features"].exists() and bundle["config"].exists():
            return bundle
    return None

def _resolve_artifact_bundle() -> dict[str, Any]:
    global _RESOLVED_ARTIFACT_BUNDLE
    if _RESOLVED_ARTIFACT_BUNDLE is not None: return _RESOLVED_ARTIFACT_BUNDLE
    bundle = _find_artifact_bundle()
    if bundle is None:
        raise ApiError("Cascade model artifacts were not found.", status=HTTPStatus.INTERNAL_SERVER_ERROR)
    _RESOLVED_ARTIFACT_BUNDLE = bundle
    return _RESOLVED_ARTIFACT_BUNDLE

class CascadeModel:
    """Wrapper for the Two-Stage Cascade architecture."""
    def __init__(self, sieve, verifier):
        self.sieve = sieve
        self.verifier = verifier
    
    def predict_proba(self, X):
        s_prob = self.sieve.predict_proba(X)[:, 1]
        v_prob = self.verifier.predict_proba(X)[:, 1]
        combined = s_prob * v_prob
        return np.vstack([1 - combined, combined]).T

    @property
    def feature_importances_(self):
        return self.verifier.feature_importances_

def _load_model_and_features() -> tuple[Any, list[str], float]:
    global _CACHED_SIEVE, _CACHED_VERIFIER, _CACHED_FEATURES, _CACHED_THRESHOLD
    if _CACHED_SIEVE is not None and _CACHED_VERIFIER is not None:
        return CascadeModel(_CACHED_SIEVE, _CACHED_VERIFIER), _CACHED_FEATURES, _CACHED_THRESHOLD

    bundle = _resolve_artifact_bundle()
    _log("Loading Two-Stage Cascade Models...")
    _CACHED_SIEVE = joblib.load(bundle["sieve"])
    _CACHED_VERIFIER = joblib.load(bundle["verifier"])
    _CACHED_FEATURES = list(joblib.load(bundle["features"]))
    with open(bundle["config"], 'r') as f:
        config = json.load(f)
        _CACHED_THRESHOLD = float(config.get("optimal_threshold", 0.5))
    return CascadeModel(_CACHED_SIEVE, _CACHED_VERIFIER), _CACHED_FEATURES, _CACHED_THRESHOLD

def _load_feature_names() -> list[str]:
    _, features, _ = _load_model_and_features()
    return features

def _resolve_data_path(features: list[str]) -> Path | None:
    global _RESOLVED_DATA_PATH
    if _RESOLVED_DATA_PATH is not None: return _RESOLVED_DATA_PATH
    for candidate in DATA_PATH_CANDIDATES:
        if candidate.exists():
            _RESOLVED_DATA_PATH = candidate
            return candidate
    return None

def _load_data() -> pd.DataFrame | None:
    global _CACHED_DF
    if _CACHED_DF is not None: return _CACHED_DF
    path = _resolve_data_path(_load_feature_names())
    if path is None: return None
    _CACHED_DF = pd.read_csv(path)
    return _CACHED_DF

DEFAULT_FIELD_STATS: dict[str, tuple[float, float, float]] = {
    "HR": (20.0, 280.0, 84.34), "O2Sat": (50.0, 100.0, 97.18),
    "Temp": (26.6, 42.22, 36.87), "SBP": (40.0, 250.0, 123.55),
    "MAP": (30.0, 200.0, 82.54), "DBP": (20.0, 160.0, 63.5),
    "Resp": (4.0, 60.0, 18.7), "HCO3": (5.0, 55.0, 24.15),
    "FiO2": (0.2, 1.0, 0.33), "pH": (6.62, 7.93, 7.39),
    "PaCO2": (10.0, 100.0, 40.24), "BUN": (1.0, 268.0, 21.91),
    "Calcium": (1.0, 20.0, 8.09), "Chloride": (26.0, 145.0, 105.85),
    "Creatinine": (0.1, 29.86, 1.33), "Glucose": (10.0, 988.0, 130.58),
    "Lactate": (0.2, 28.9, 1.69), "Magnesium": (0.2, 9.8, 2.03),
    "Phosphate": (0.2, 18.8, 3.43), "Potassium": (1.0, 10.0, 4.08),
    "Hct": (5.5, 71.7, 31.59), "Hgb": (2.2, 30.0, 10.56),
    "WBC": (0.1, 296.1, 11.05), "Platelets": (1.0, 2322.0, 201.92),
    "Age": (14.0, 100.0, 62.01), "Gender": (0.0, 1.0, 0.56),
    "Unit1": (-1.0, 1.0, -0.09),
}

def get_prediction_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None: return _SCHEMA_CACHE
    _, features, _ = _load_model_and_features()
    
    input_features = [f for f in features if f in FIELD_LABELS]
    fields = []
    for feature in input_features:
        stats = DEFAULT_FIELD_STATS.get(feature)
        if stats is not None:
            min_v, max_v, mean_v = stats
        else:
            min_v, max_v, mean_v = 0.0, 200.0, 0.0
            
        fields.append({
            "name": feature, "label": FIELD_LABELS[feature],
            "group": _group_for_feature(feature), "kind": "number",
            "unit": FIELD_UNITS.get(feature, ""), "default_value": round(mean_v, 2),
            "min_value": round(min_v, 2), "max_value": round(max_v, 2), "step": 0.1
        })

    _SCHEMA_CACHE = {
        "model_name": "Two-Stage Cascade", "feature_count": len(features),
        "primary_feature_count": len(fields),
        "derived_feature_count": len(features) - len(fields) - sum(1 for f in features if f.endswith("_Measured")),
        "measurement_flag_count": sum(1 for f in features if f.endswith("_Measured")),
        "fields": fields, "input_groups": INPUT_GROUPS,
    }
    return _SCHEMA_CACHE

def _group_for_feature(feature: str) -> str:
    if feature in PATIENT_CONTEXT_FIELDS: return "patient_context"
    if feature in VITAL_SIGN_FIELDS: return "vital_signs"
    return "lab_markers"

def _series_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns: return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")

def _derive_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    derived = frame.copy()
    hr = _series_or_nan(derived, "HR")
    o2 = _series_or_nan(derived, "O2Sat")
    temp = _series_or_nan(derived, "Temp")
    sbp = _series_or_nan(derived, "SBP")
    map_v = _series_or_nan(derived, "MAP")
    dbp = _series_or_nan(derived, "DBP")
    resp = _series_or_nan(derived, "Resp")
    lact = _series_or_nan(derived, "Lactate")
    wbc = _series_or_nan(derived, "WBC")
    bun = _series_or_nan(derived, "BUN")
    creat = _series_or_nan(derived, "Creatinine")
    ph = _series_or_nan(derived, "pH")
    plat = _series_or_nan(derived, "Platelets")
    gluc = _series_or_nan(derived, "Glucose")
    hco3 = _series_or_nan(derived, "HCO3")
    fio2 = _series_or_nan(derived, "FiO2")
    pco2 = _series_or_nan(derived, "PaCO2")
    calc = _series_or_nan(derived, "Calcium")
    chl = _series_or_nan(derived, "Chloride")
    mag = _series_or_nan(derived, "Magnesium")
    phos = _series_or_nan(derived, "Phosphate")
    pot = _series_or_nan(derived, "Potassium")
    hct = _series_or_nan(derived, "Hct")
    hgb = _series_or_nan(derived, "Hgb")

    derived["Unit1_Unknown"] = 0.0

    temporal_cols = ['HR', 'MAP', 'SBP', 'Resp', 'O2Sat', 'Temp', 'Lactate', 'Creatinine', 'WBC', 'Glucose', 'Platelets']
    for col in temporal_cols:
        s = _series_or_nan(derived, col)
        derived[f"{col}_delta_1h"] = 0.0
        derived[f"{col}_delta_6h"] = 0.0
        derived[f"{col}_roll6_mean"] = s
        derived[f"{col}_roll6_std"] = 0.0

    derived["qSOFA_score"] = (sbp <= 100.0).astype(float) + (resp >= 22.0).astype(float)
    derived["SIRS_Temp"] = ((temp < 36.0) | (temp > 38.0)).astype(float)
    derived["SIRS_HR"] = (hr > 90.0).astype(float)
    derived["SIRS_WBC"] = ((wbc < 4.0) | (wbc > 12.0)).astype(float)
    derived["SIRS_score"] = derived["SIRS_Temp"] + derived["SIRS_HR"] + derived["SIRS_WBC"]
    derived["renal_flag"] = (creat >= 1.2).astype(float)
    derived["coag_flag"] = (plat < 150.0).astype(float)
    derived["acidosis_flag"] = (ph < 7.35).astype(float)

    adv_cols = ['HR', 'MAP', 'SBP', 'Resp', 'O2Sat', 'Temp', 'Lactate', 'Glucose']
    for col in adv_cols:
        s = _series_or_nan(derived, col)
        derived[f"{col}_zscore"] = 0.0
        derived[f"{col}_adv_12h_mean"] = s
        derived[f"{col}_adv_12h_max"] = s
        derived[f"{col}_adv_12h_min"] = s
        derived[f"{col}_adv_6h_slope"] = 0.0
        derived[f"{col}_adv_delta"] = 0.0

    return derived

def predict_sepsis(patient_data: dict[str, Any]) -> dict[str, Any]:
    model, features, threshold = _load_model_and_features()
    normalized = {}
    for f in features:
        if f.endswith("_Measured"):
            source = f[:-9]
            normalized[f] = [0.0 if patient_data.get(source) is None else 1.0]
        else:
            normalized[f] = [float(patient_data.get(f, np.nan))]
    
    input_frame = pd.DataFrame(normalized)
    input_frame = _derive_engineered_features(input_frame)
    prob = float(model.predict_proba(input_frame[features])[0, 1])
    
    importances = model.feature_importances_
    contributor_values = input_frame[features].iloc[0].to_dict()
    paired = [(features[i], float(contributor_values.get(features[i], 0)), float(importances[i])) for i in range(len(features))]
    paired.sort(key=lambda x: x[2], reverse=True)
    top_contributors = [
        {"feature": f, "value": v if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else None, "contribution_score": s}
        for f, v, s in paired[:5]
    ]
    
    return {
        "sepsis_probability": prob,
        "risk_level": "High" if prob >= 0.5 else ("Medium" if prob >= 0.2 else "Low"),
        "threshold_used": threshold,
        "top_contributors": top_contributors
    }

def get_health_status() -> dict[str, Any]:
    return {"status": "ok"}

def _dispatch_rpc(func_name: str, args: dict[str, Any]) -> Any:
    if func_name == "get_prediction_schema":
        return get_prediction_schema()
    elif func_name == "get_summary_stats":
        return get_summary_stats()
    elif func_name == "get_feature_importance":
        return get_feature_importance(**args)
    elif func_name == "get_cohort_analysis":
        return get_cohort_analysis(**args)
    elif func_name == "predict_sepsis":
        return predict_sepsis(**args)
    else:
        raise ApiError(f"Unknown RPC function: {func_name}")

def get_summary_stats() -> dict[str, Any]:
    global _SUMMARY_CACHE
    if _SUMMARY_CACHE: return _SUMMARY_CACHE
    precomputed = CACHE_DIR / "summary_stats.json"
    if precomputed.exists():
        with open(precomputed, 'r') as f:
            _SUMMARY_CACHE = json.load(f)
        return _SUMMARY_CACHE
    bundle = _resolve_artifact_bundle()
    with open(bundle["config"], 'r') as f: config = json.load(f)
    df = _load_data()
    if df is not None:
        sepsis_rate = float(df["SepsisLabel"].mean()) if "SepsisLabel" in df.columns else 0.01798
        avg_age = float(df["Age"].mean()) if "Age" in df.columns else 62.0
        total_patients = int(df["Patient_ID"].nunique()) if "Patient_ID" in df.columns else len(df)
    else:
        sepsis_rate = 0.01798
        avg_age = 62.0
        total_patients = 40335
    _SUMMARY_CACHE = {
        "total_patients": total_patients,
        "sepsis_rate": sepsis_rate,
        "avg_age": avg_age,
        "auc": config.get("auc_score", 0.8180),
        "threshold": config.get("optimal_threshold", 0.448),
        "model_name": "Two-Stage Cascade (XGBoost)"
    }
    return _SUMMARY_CACHE

def get_cohort_analysis(cohort_feature: str = "AgeGroup") -> list[dict[str, Any]]:
    cache_file = CACHE_DIR / f"cohorts_{cohort_feature}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    df = _load_data()
    if df is None:
        return []
    if cohort_feature not in df.columns:
        return []
    cohorts = []
    for val in sorted(df[cohort_feature].dropna().unique()):
        mask = df[cohort_feature] == val
        subset = df[mask]
        total = len(subset)
        sepsis_cases = int(subset["SepsisLabel"].sum()) if "SepsisLabel" in subset.columns else 0
        recall = 0.0
        precision = 0.0
        if sepsis_cases > 0:
            preds = subset.get("Prediction", None)
            if preds is not None:
                tp = ((preds == 1) & (subset["SepsisLabel"] == 1)).sum()
                fn = ((preds == 0) & (subset["SepsisLabel"] == 1)).sum()
                fp = ((preds == 1) & (subset["SepsisLabel"] == 0)).sum()
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        cohorts.append({
            "cohort": str(val),
            "count": total,
            "sepsis_cases": sepsis_cases,
            "recall": recall,
            "precision": precision
        })
    return cohorts

def get_feature_importance(top_n: int = 10) -> list[dict[str, Any]]:
    model, features, _ = _load_model_and_features()
    ranked = [{"feature": f, "importance": float(i)} for f, i in zip(features, model.feature_importances_)]
    return sorted(ranked, key=lambda x: x["importance"], reverse=True)[:top_n]

def _json_safe(v):
    if isinstance(v, (np.floating, float)): return float(v)
    if isinstance(v, (np.integer, int)): return int(v)
    return v

class SepsisRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        super().end_headers()

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(_json_safe(data)).encode('utf-8'))

    def _serve_static(self, path: str) -> None:
        if not FRONTEND_DIST_DIR.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Frontend not built.")
            return

        clean = path.lstrip("/") or "index.html"
        file_path = FRONTEND_DIST_DIR / clean
        if not file_path.exists():
            file_path = FRONTEND_DIST_DIR / "index.html"

        mime_type, _ = mimetypes.guess_type(str(file_path))
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()
        with open(file_path, "rb") as f:
            self.wfile.write(f.read())

    def do_GET(self):
        if self.path == "/api/summary": self._send_json(get_summary_stats())
        elif self.path == "/api/health": self._send_json({"status": "ok"})
        elif self.path.startswith("/api/feature-importance"):
            params = parse_qs(urlparse(self.path).query)
            top_n = int(params.get("top_n", ["10"])[0])
            self._send_json(get_feature_importance(top_n))
        else: self._serve_static(self.path)

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_len).decode('utf-8'))
        if self.path == "/api/predict": self._send_json(predict_sepsis(body))
        elif self.path == "/api/rpc":
            func = body.get("func")
            args = body.get("args", {})
            if func == "get_prediction_schema": self._send_json(get_prediction_schema())
            elif func == "get_summary_stats": self._send_json(get_summary_stats())
            elif func == "get_feature_importance": self._send_json(get_feature_importance(**args))
            elif func == "get_cohort_analysis": self._send_json(get_cohort_analysis(**args))
            elif func == "predict_sepsis": self._send_json(predict_sepsis(**args))
            else: self._send_json({"error": f"Unknown RPC function: {func}"})
        else: self.send_response(404); self.end_headers()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    _log(f"Starting Cascade Backend on port {port}...")
    HTTPServer(("0.0.0.0", port), SepsisRequestHandler).serve_forever()
