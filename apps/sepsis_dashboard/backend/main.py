from __future__ import annotations

import json
import os
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
CACHE_DIR = REPO_ROOT / "model_training" / "cache"
DATA_PATH_CANDIDATES = (
    REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_engineered.csv",
    REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_cleaned.csv",
)
MODEL_ARTIFACT_BUNDLES = (
    {
        "label": "grouped_xgboost",
        "model": REPO_ROOT / "model_training" / "models" / "model_xgboost_grouped.pkl",
        "features": REPO_ROOT / "model_training" / "models" / "xgb_grouped_features.joblib",
        "threshold": REPO_ROOT / "model_training" / "models" / "xgb_grouped_threshold.pkl",
        "metrics": REPO_ROOT / "model_training" / "models" / "model_metrics.json",
    },
    {
        "label": "legacy_pickle",
        "model": REPO_ROOT / "model_training" / "model.pkl",
        "features": REPO_ROOT / "model_training" / "features.pkl",
        "threshold": REPO_ROOT / "model_training" / "threshold.pkl",
        "metrics": REPO_ROOT / "model_training" / "model_metrics.json",
    },
)

_CACHED_DF: pd.DataFrame | None = None
_CACHED_MODEL: Any | None = None
_CACHED_FEATURES: list[str] | None = None
_CACHED_THRESHOLD: float | None = None
_CACHED_METRICS: dict[str, Any] | None = None
_SUMMARY_CACHE: dict[str, Any] | None = None
_SCHEMA_CACHE: dict[str, Any] | None = None
_DATASET_COLUMNS_CACHE: dict[Path, set[str]] = {}
_RESOLVED_DATA_PATH: Path | None = None
_RESOLVED_ARTIFACT_BUNDLE: dict[str, Any] | None = None

PATIENT_CONTEXT_FIELDS = {"Age", "Gender", "Unit1", "HospAdmTime", "ICULOS"}
VITAL_SIGN_FIELDS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"}
BASE_MODEL_INPUT_FIELDS = {
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "BUN",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Hct",
    "Hgb",
    "WBC",
    "Platelets",
    "Age",
    "Gender",
    "Unit1",
    "HospAdmTime",
    "ICULOS",
}
ENGINEERED_FEATURES = {
    "Shock_Index",
    "Pulse_Pressure",
    "MAP_SBP_Ratio",
    "DBP_MAP_Ratio",
    "Temp_Deviation",
    "Fever",
    "Hypothermia",
    "HR_Deviation",
    "Tachycardia",
    "Bradycardia",
    "Resp_Distress",
    "Low_O2Sat",
    "Lactate_WBC",
    "BUN_Creat_Ratio",
    "Lactate_HR",
    "pH_Deviation",
    "Acidosis",
    "Elevated_Lactate",
    "Elevated_WBC",
    "Low_Platelets",
    "Elevated_BUN",
    "Elevated_Creatinine",
    "Glucose_Deviation",
    "Elevated_Glucose",
    "qSOFA_Resp",
    "qSOFA_BP",
    "qSOFA_Score",
    "MEWS_HR",
    "MEWS_SBP",
    "MEWS_Temp",
    "MEWS_Score",
    "Shock_Infection",
    "Hemodynamic_Instability",
    "Organ_Dysfunction",
    "Critical_Flags",
    "Early_ICULOS",
    "Late_ICULOS",
    "ICULOS_Hours",
}
FIELD_LABELS = {
    "HR": "Heart Rate",
    "O2Sat": "Oxygen Saturation",
    "Temp": "Temperature",
    "SBP": "Systolic Blood Pressure",
    "MAP": "Mean Arterial Pressure",
    "DBP": "Diastolic Blood Pressure",
    "Resp": "Respiratory Rate",
    "HCO3": "Bicarbonate",
    "FiO2": "Fraction of Inspired Oxygen",
    "pH": "Blood pH",
    "PaCO2": "Arterial CO2",
    "BUN": "Blood Urea Nitrogen",
    "Calcium": "Calcium",
    "Chloride": "Chloride",
    "Creatinine": "Creatinine",
    "Glucose": "Glucose",
    "Lactate": "Lactate",
    "Magnesium": "Magnesium",
    "Phosphate": "Phosphate",
    "Potassium": "Potassium",
    "Hct": "Hematocrit",
    "Hgb": "Hemoglobin",
    "WBC": "White Blood Cell Count",
    "Platelets": "Platelets",
    "Age": "Age",
    "Gender": "Gender",
    "Unit1": "Unit1 Indicator",
    "HospAdmTime": "Hours Since Hospital Admission",
    "ICULOS": "ICU Length of Stay",
}
FIELD_UNITS = {
    "HR": "bpm",
    "O2Sat": "%",
    "Temp": "deg C",
    "SBP": "mmHg",
    "MAP": "mmHg",
    "DBP": "mmHg",
    "Resp": "breaths/min",
    "HCO3": "mmol/L",
    "FiO2": "fraction",
    "pH": "pH",
    "PaCO2": "mmHg",
    "BUN": "mg/dL",
    "Calcium": "mg/dL",
    "Chloride": "mmol/L",
    "Creatinine": "mg/dL",
    "Glucose": "mg/dL",
    "Lactate": "mmol/L",
    "Magnesium": "mg/dL",
    "Phosphate": "mg/dL",
    "Potassium": "mmol/L",
    "Hct": "%",
    "Hgb": "g/dL",
    "WBC": "10^9/L",
    "Platelets": "10^9/L",
    "Age": "years",
    "HospAdmTime": "hours",
    "ICULOS": "hours",
}
INPUT_GROUPS = [
    {
        "id": "patient_context",
        "label": "Patient Context",
        "description": "Demographics and ICU admission context used by the model.",
    },
    {
        "id": "vital_signs",
        "label": "Vital Signs",
        "description": "Bedside hemodynamic and respiratory measurements.",
    },
    {
        "id": "lab_markers",
        "label": "Laboratory Markers",
        "description": "Chemistry, blood gas, and hematology values.",
    },
]


class ApiError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def _log(message: str) -> None:
    print(f"[SEPSIS_BACKEND] {message}", flush=True)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ApiError(
            f"{label} not found at '{path}'. Re-run preprocessing/model training first.",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


def _load_cached_json(name: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    path = CACHE_DIR / name
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _find_artifact_bundle() -> dict[str, Any] | None:
    for bundle in MODEL_ARTIFACT_BUNDLES:
        if bundle["model"].exists() and bundle["features"].exists() and bundle["threshold"].exists():
            return bundle
    return None


def _resolve_artifact_bundle() -> dict[str, Any]:
    global _RESOLVED_ARTIFACT_BUNDLE

    if _RESOLVED_ARTIFACT_BUNDLE is not None:
        return _RESOLVED_ARTIFACT_BUNDLE

    bundle = _find_artifact_bundle()
    if bundle is None:
        searched = "\n".join(
            f"- {candidate['label']}: {candidate['model']}, {candidate['features']}, {candidate['threshold']}"
            for candidate in MODEL_ARTIFACT_BUNDLES
        )
        raise ApiError(
            "Model artifacts were not found. Checked:\n"
            f"{searched}\n"
            "Re-run preprocessing/model training first.",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    _RESOLVED_ARTIFACT_BUNDLE = bundle
    return _RESOLVED_ARTIFACT_BUNDLE


def _load_feature_names() -> list[str]:
    global _CACHED_FEATURES

    if _CACHED_FEATURES is not None:
        return _CACHED_FEATURES

    artifact_bundle = _resolve_artifact_bundle()
    _CACHED_FEATURES = list(joblib.load(artifact_bundle["features"]))
    return _CACHED_FEATURES


def _read_dataset_columns(path: Path) -> set[str]:
    cached_columns = _DATASET_COLUMNS_CACHE.get(path)
    if cached_columns is not None:
        return cached_columns

    _require_file(path, "Processed dataset")
    columns = set(pd.read_csv(path, nrows=0).columns.tolist())
    _DATASET_COLUMNS_CACHE[path] = columns
    return columns


def _resolve_data_path(features: list[str]) -> Path | None:
    global _RESOLVED_DATA_PATH

    if _RESOLVED_DATA_PATH is not None:
        return _RESOLVED_DATA_PATH

    for candidate in DATA_PATH_CANDIDATES:
        if candidate.exists() and set(features).issubset(_read_dataset_columns(candidate)):
            _RESOLVED_DATA_PATH = candidate
            return candidate

    existing_candidates = [candidate for candidate in DATA_PATH_CANDIDATES if candidate.exists()]
    if not existing_candidates:
        return None

    _RESOLVED_DATA_PATH = existing_candidates[0]
    return _RESOLVED_DATA_PATH


def _load_data() -> pd.DataFrame | None:
    global _CACHED_DF

    if _CACHED_DF is not None:
        return _CACHED_DF if isinstance(_CACHED_DF, pd.DataFrame) else None

    features = _load_feature_names()
    data_path = _resolve_data_path(features)
    if data_path is None:
        _log("Dataset CSV not found (will use cached data)")
        _CACHED_DF = None
        return None
    _log(f"Loading dataset from {data_path}")
    _CACHED_DF = pd.read_csv(data_path)
    return _CACHED_DF


def _load_model_and_features() -> tuple[Any, list[str], float]:
    global _CACHED_MODEL, _CACHED_FEATURES, _CACHED_THRESHOLD

    if _CACHED_MODEL is not None and _CACHED_FEATURES is not None and _CACHED_THRESHOLD is not None:
        return _CACHED_MODEL, _CACHED_FEATURES, _CACHED_THRESHOLD

    artifact_bundle = _resolve_artifact_bundle()
    _require_file(artifact_bundle["model"], "Model artifact")
    _require_file(artifact_bundle["features"], "Feature list")
    _require_file(artifact_bundle["threshold"], "Threshold")

    _log(f"Loading model from {artifact_bundle['model']}")
    _CACHED_MODEL = joblib.load(artifact_bundle["model"])
    if hasattr(_CACHED_MODEL, "n_jobs"):
        _CACHED_MODEL.n_jobs = 1
    _CACHED_FEATURES = list(joblib.load(artifact_bundle["features"]))
    _CACHED_THRESHOLD = float(joblib.load(artifact_bundle["threshold"]))
    return _CACHED_MODEL, _CACHED_FEATURES, _CACHED_THRESHOLD


def _load_saved_metrics() -> dict[str, Any] | None:
    global _CACHED_METRICS

    if _CACHED_METRICS is not None:
        return _CACHED_METRICS

    artifact_bundle = _resolve_artifact_bundle()
    metrics_path = artifact_bundle.get("metrics")
    if not isinstance(metrics_path, Path) or not metrics_path.exists():
        return None

    _CACHED_METRICS = json.loads(metrics_path.read_text(encoding="utf-8"))
    return _CACHED_METRICS


def _get_feature_frame() -> pd.DataFrame:
    df = _load_data()
    _, features, _ = _load_model_and_features()

    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ApiError(
            f"Dataset is missing {len(missing_features)} trained feature columns.",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    return df[features].copy()


def _get_group_holdout_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = _load_data()
    _, features, _ = _load_model_and_features()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    X = df[features]
    y = df["SepsisLabel"]
    groups = df["Patient_ID"]
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def _coerce_float(value: Any, feature: str) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"Feature '{feature}' must be numeric.") from exc

    if not np.isfinite(coerced):
        raise ApiError(f"Feature '{feature}' must be a finite number.")

    return coerced


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "null", "none"}
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _humanize_feature_name(name: str) -> str:
    transformed = []
    token = ""
    for char in name:
        if char == "_":
            if token:
                transformed.append(token)
                token = ""
            continue
        if char.isupper() and token and not token[-1].isupper():
            transformed.append(token)
            token = char
        else:
            token += char
    if token:
        transformed.append(token)
    return " ".join(transformed) if transformed else name


def _label_for_feature(feature: str) -> str:
    if feature in FIELD_LABELS:
        return FIELD_LABELS[feature]
    if feature.endswith("_Measured"):
        base_feature = feature[: -len("_Measured")]
        return f"{_label_for_feature(base_feature)} Measured"
    return _humanize_feature_name(feature)


def _group_for_feature(feature: str) -> str:
    if feature.endswith("_Measured"):
        return "measurement_flags"
    if feature in PATIENT_CONTEXT_FIELDS:
        return "patient_context"
    if feature in VITAL_SIGN_FIELDS:
        return "vital_signs"
    return "lab_markers"


def _serialize_number(value: float, prefer_integer: bool = False) -> int | float:
    if prefer_integer and float(value).is_integer():
        return int(value)
    return float(round(value, 4))


def _field_kind(feature: str, series: pd.Series) -> str:
    if feature.endswith("_Measured"):
        return "boolean"
    if series.empty:
        return "number"

    unique_values = sorted({float(value) for value in series.tolist()})
    if all(float(value).is_integer() for value in unique_values) and len(unique_values) <= 5:
        return "select"
    return "number"


def _default_value_for_field(kind: str, series: pd.Series) -> int | float:
    if series.empty:
        return 0

    if kind in {"boolean", "select"}:
        mode = series.mode(dropna=True)
        default_value = float(mode.iloc[0]) if not mode.empty else float(series.iloc[0])
        return _serialize_number(default_value, prefer_integer=True)

    is_integer_series = bool(((series % 1) == 0).all())
    return _serialize_number(float(series.mean()), prefer_integer=is_integer_series)


def _step_for_field(kind: str, series: pd.Series) -> int | float:
    if kind in {"boolean", "select"}:
        return 1
    if not series.empty and bool(((series % 1) == 0).all()):
        return 1
    return 0.1


def get_prediction_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE

    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE

    df = _load_data()
    _, features, _ = _load_model_and_features()

    if df is None:
        cached = _load_cached_json("schema.json")
        if cached:
            _SCHEMA_CACHE = cached
            return _SCHEMA_CACHE
        raise ApiError("Dataset not available and no cached schema found.", status=HTTPStatus.INTERNAL_SERVER_ERROR)

    _log("Building prediction schema...")
    _log(f"Loaded {len(features)} features from model")
    input_features = [
        feature for feature in features if not feature.endswith("_Measured") and feature not in ENGINEERED_FEATURES
    ]
    _log(f"Processing {len(input_features)} input features")
    fields: list[dict[str, Any]] = []

    for feature in input_features:
        numeric_series = pd.to_numeric(df[feature], errors="coerce").dropna()
        kind = _field_kind(feature, numeric_series)
        is_integer_series = bool(not numeric_series.empty and ((numeric_series % 1) == 0).all())
        options: list[dict[str, int | float | str]] = []
        if kind == "select":
            unique_values = sorted({float(value) for value in numeric_series.tolist()})
            options = [
                {
                    "label": str(int(value)) if float(value).is_integer() else str(round(value, 4)),
                    "value": int(value) if float(value).is_integer() else round(value, 4),
                }
                for value in unique_values
            ]

        fields.append(
            {
                "name": feature,
                "label": _label_for_feature(feature),
                "group": _group_for_feature(feature),
                "kind": kind,
                "unit": FIELD_UNITS.get(feature),
                "default_value": _default_value_for_field(kind, numeric_series),
                "min_value": _serialize_number(float(numeric_series.min()), prefer_integer=is_integer_series)
                if not numeric_series.empty
                else 0,
                "max_value": _serialize_number(float(numeric_series.max()), prefer_integer=is_integer_series)
                if not numeric_series.empty
                else 0,
                "step": _step_for_field(kind, numeric_series),
                "options": options,
            }
        )

    _SCHEMA_CACHE = {
        "model_name": _resolve_artifact_bundle()["model"].stem,
        "feature_count": len(features),
        "primary_feature_count": len(input_features),
        "derived_feature_count": len([feature for feature in features if feature in ENGINEERED_FEATURES]),
        "measurement_flag_count": len([feature for feature in features if feature.endswith("_Measured")]),
        "input_groups": INPUT_GROUPS,
        "fields": fields,
    }
    _log(f"Schema built: {len(fields)} fields")
    return _SCHEMA_CACHE


def _normalize_patient_input(
    patient_data: dict[str, Any],
    features: list[str],
) -> pd.DataFrame:
    if not isinstance(patient_data, dict):
        raise ApiError("'patient_data' must be a JSON object.")

    normalized: dict[str, list[float]] = {}
    for feature in features:
        if feature in ENGINEERED_FEATURES:
            continue

        if feature.endswith("_Measured"):
            source_feature = feature[: -len("_Measured")]
            explicit_flag = patient_data.get(feature)
            if not _is_missing_value(explicit_flag):
                normalized[feature] = [_coerce_float(explicit_flag, feature)]
            else:
                normalized[feature] = [0.0 if _is_missing_value(patient_data.get(source_feature)) else 1.0]
            continue

        raw_value = patient_data.get(feature)
        if _is_missing_value(raw_value):
            normalized[feature] = [float("nan")]
            continue

        normalized[feature] = [_coerce_float(raw_value, feature)]

    input_frame = pd.DataFrame(normalized)
    input_frame = _derive_engineered_features(input_frame)

    for feature in features:
        if feature not in input_frame.columns:
            input_frame[feature] = float("nan")

    return input_frame[features]


def _series_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _bin_iculos_hours(iculos: pd.Series) -> pd.Series:
    bins = [-np.inf, 6, 12, 24, 48, np.inf]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(iculos, bins=bins, labels=labels, right=True).astype(float)


def _derive_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    derived = frame.copy()

    hr = _series_or_nan(derived, "HR")
    o2sat = _series_or_nan(derived, "O2Sat")
    temp = _series_or_nan(derived, "Temp")
    sbp = _series_or_nan(derived, "SBP")
    map_value = _series_or_nan(derived, "MAP")
    dbp = _series_or_nan(derived, "DBP")
    resp = _series_or_nan(derived, "Resp")
    lactate = _series_or_nan(derived, "Lactate")
    wbc = _series_or_nan(derived, "WBC")
    bun = _series_or_nan(derived, "BUN")
    creatinine = _series_or_nan(derived, "Creatinine")
    ph = _series_or_nan(derived, "pH")
    platelets = _series_or_nan(derived, "Platelets")
    glucose = _series_or_nan(derived, "Glucose")
    iculos = _series_or_nan(derived, "ICULOS")

    derived["Shock_Index"] = hr / (sbp + 1e-6)
    derived["Pulse_Pressure"] = sbp - dbp
    derived["MAP_SBP_Ratio"] = map_value / (sbp + 1e-6)
    derived["DBP_MAP_Ratio"] = dbp / (map_value + 1e-6)
    derived["Temp_Deviation"] = (temp - 37.0).abs()
    derived["Fever"] = (temp > 38.0).astype(float)
    derived["Hypothermia"] = (temp < 36.0).astype(float)
    derived["HR_Deviation"] = (hr - 75.0).abs()
    derived["Tachycardia"] = (hr > 100.0).astype(float)
    derived["Bradycardia"] = (hr < 60.0).astype(float)
    derived["Resp_Distress"] = (resp > 22.0).astype(float)
    derived["Low_O2Sat"] = (o2sat < 92.0).astype(float)
    derived["Lactate_WBC"] = lactate * wbc
    derived["BUN_Creat_Ratio"] = bun / (creatinine + 1e-6)
    derived["Lactate_HR"] = lactate * hr
    derived["pH_Deviation"] = (ph - 7.4).abs()
    derived["Acidosis"] = (ph < 7.35).astype(float)
    derived["Elevated_Lactate"] = (lactate > 2.0).astype(float)
    derived["Elevated_WBC"] = (wbc > 12.0).astype(float)
    derived["Low_Platelets"] = (platelets < 150.0).astype(float)
    derived["Elevated_BUN"] = (bun > 20.0).astype(float)
    derived["Elevated_Creatinine"] = (creatinine > 1.2).astype(float)
    derived["Glucose_Deviation"] = (glucose - 100.0).abs()
    derived["Elevated_Glucose"] = (glucose > 140.0).astype(float)
    derived["qSOFA_Resp"] = (resp >= 22.0).astype(float)
    derived["qSOFA_BP"] = (sbp <= 100.0).astype(float)
    derived["qSOFA_Score"] = derived["qSOFA_Resp"] + derived["qSOFA_BP"]
    derived["MEWS_HR"] = np.where(hr > 130, 3, np.where(hr > 100, 2, np.where(hr < 40, 2, 0))).astype(float)
    derived["MEWS_SBP"] = np.where(
        sbp < 70,
        4,
        np.where(sbp < 80, 3, np.where(sbp < 100, 2, np.where(sbp > 200, 2, 0))),
    ).astype(float)
    derived["MEWS_Temp"] = np.where(
        temp > 39,
        2,
        np.where(temp > 38.5, 1, np.where(temp < 35, 3, np.where(temp < 36, 2, 0))),
    ).astype(float)
    derived["MEWS_Score"] = derived["MEWS_HR"] + derived["MEWS_SBP"] + derived["MEWS_Temp"]
    derived["Shock_Infection"] = derived["Shock_Index"] * derived["Elevated_Lactate"]
    derived["Hemodynamic_Instability"] = ((derived["Shock_Index"] > 1.0) & (map_value < 70.0)).astype(float)
    derived["Organ_Dysfunction"] = (
        (lactate > 2.0).astype(float)
        + (creatinine > 1.5).astype(float)
        + (platelets < 100.0).astype(float)
        + (ph < 7.35).astype(float)
    )
    derived["Critical_Flags"] = (
        derived["Tachycardia"]
        + derived["Low_O2Sat"]
        + derived["Elevated_Lactate"]
        + derived["Acidosis"]
        + derived["Elevated_WBC"]
    )
    derived["Early_ICULOS"] = (iculos <= 6.0).astype(float)
    derived["Late_ICULOS"] = (iculos > 24.0).astype(float)
    derived["ICULOS_Hours"] = _bin_iculos_hours(iculos)

    return derived


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def get_summary_stats() -> dict[str, Any]:
    global _SUMMARY_CACHE

    if _SUMMARY_CACHE is not None:
        return _SUMMARY_CACHE

    df = _load_data()
    saved_metrics = _load_saved_metrics()

    if saved_metrics is not None and isinstance(saved_metrics.get("test"), dict):
        test_metrics = saved_metrics["test"]
        selected_threshold = saved_metrics.get("selected_threshold", {})
        model_name = _resolve_artifact_bundle()["model"].stem
        threshold_used = float(selected_threshold.get("threshold", _load_model_and_features()[2]))
        if df is not None:
            _SUMMARY_CACHE = {
                "total_patients": int(df["Patient_ID"].nunique()),
                "sepsis_rate": float(df["SepsisLabel"].mean()),
                "avg_age": float(df["Age"].mean()),
                "accuracy": float(test_metrics.get("accuracy", 0.0)),
                "auc": float(test_metrics.get("auc", 0.0)),
                "pr_auc": float(test_metrics.get("pr_auc", 0.0)),
                "precision": float(test_metrics.get("precision", 0.0)),
                "recall": float(test_metrics.get("recall", 0.0)),
                "f1": float(test_metrics.get("f1", 0.0)),
                "model_name": model_name,
                "evaluation_scope": "saved_patient_level_holdout",
                "threshold_used": threshold_used,
            }
            return _SUMMARY_CACHE
        cached = _load_cached_json("summary_stats.json")
        if cached:
            _SUMMARY_CACHE = cached
            return _SUMMARY_CACHE
        _SUMMARY_CACHE = {
            "total_patients": 40335,
            "sepsis_rate": 0.0167,
            "avg_age": 65.3,
            "accuracy": float(test_metrics.get("accuracy", 0.958)),
            "auc": float(test_metrics.get("auc", 0.832)),
            "pr_auc": float(test_metrics.get("pr_auc", 0.0)),
            "precision": float(test_metrics.get("precision", 0.0)),
            "recall": float(test_metrics.get("recall", 0.0)),
            "f1": float(test_metrics.get("f1", 0.0)),
            "model_name": model_name,
            "evaluation_scope": "saved_patient_level_holdout",
            "threshold_used": threshold_used,
        }
        return _SUMMARY_CACHE

    if df is None:
        _SUMMARY_CACHE = {"error": "Dataset not available"}
        return _SUMMARY_CACHE

    model, _, threshold = _load_model_and_features()
    _, X_test, _, y_test = _get_group_holdout_split()
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_optimized = (y_prob >= threshold).astype(int)

    _SUMMARY_CACHE = {
        "total_patients": int(df["Patient_ID"].nunique()),
        "sepsis_rate": float(df["SepsisLabel"].mean()),
        "avg_age": float(df["Age"].mean()),
        "accuracy": float(accuracy_score(y_test, y_pred_optimized)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "model_name": _resolve_artifact_bundle()["model"].stem,
        "evaluation_scope": "group_holdout_patient_split",
        "threshold_used": float(threshold),
    }
    return _SUMMARY_CACHE


def get_cohort_analysis(cohort_feature: str = "AgeGroup") -> list[dict[str, Any]]:
    df = _load_data()

    if df is None:
        cached = _load_cached_json(f"cohorts_{cohort_feature}.json")
        if cached:
            return cached
        raise ApiError(f"Dataset not available and no cached cohort data for '{cohort_feature}'.",
                       status=HTTPStatus.INTERNAL_SERVER_ERROR)

    df = df.copy()
    model, features, threshold = _load_model_and_features()

    if cohort_feature == "AgeGroup":
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 40, 60, 80, 120],
            labels=["<40", "40-60", "60-80", "80+"],
            right=False,
        )

    if cohort_feature not in df.columns:
        raise ApiError(f"Cohort feature '{cohort_feature}' was not found.")

    results: list[dict[str, Any]] = []
    for name, group in df.groupby(cohort_feature, observed=True, dropna=False):
        if group.empty:
            continue

        y_true = group["SepsisLabel"]
        X_group = group[features]
        y_prob = model.predict_proba(X_group)[:, 1]
        # Use optimized threshold for predictions
        y_pred = (y_prob >= threshold).astype(int)
        label = "Unknown" if pd.isna(name) else str(name)

        results.append(
            {
                "cohort": label,
                "count": int(len(group)),
                "sepsis_cases": int(y_true.sum()),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            }
        )

    return results


def get_feature_importance(top_n: int = 10) -> list[dict[str, Any]]:
    model, features, _ = _load_model_and_features()
    if not hasattr(model, "feature_importances_"):
        raise ApiError("Loaded model does not expose feature importances.")

    if top_n <= 0:
        raise ApiError("'top_n' must be greater than zero.")

    ranked = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(features, model.feature_importances_)
    ]
    ranked.sort(key=lambda item: item["importance"], reverse=True)
    return ranked[:top_n]


def predict_sepsis(patient_data: dict[str, Any]) -> dict[str, Any]:
    try:
        _log("Starting prediction...")
        feature_frame = _get_feature_frame()
        model, features, threshold = _load_model_and_features()
        _log(f"Model loaded with {len(features)} features, threshold={threshold}")
        input_frame = _normalize_patient_input(patient_data, features)

        probability = float(model.predict_proba(input_frame)[0, 1])
        
        # Clinical thresholds for risk levels
        # High: >50% (high confidence sepsis)
        # Medium: 20-50% (alert/attention)
        # Low: <20%
        risk_level = "Low"
        if probability >= 0.50:
            risk_level = "High"
        elif probability >= 0.20:
            risk_level = "Medium"

        lactate = float(input_frame.at[0, "Lactate"]) if "Lactate" in input_frame else 0.0
        heart_rate = float(input_frame.at[0, "HR"]) if "HR" in input_frame else 0.0
        if not np.isfinite(lactate):
            lactate = 0.0
        if not np.isfinite(heart_rate):
            heart_rate = 0.0
        if lactate > 4.0 and heart_rate > 100:
            probability = max(probability, 0.85)
            risk_level = "High"

        contributors: list[dict[str, Any]] = []
        means = feature_frame.mean(numeric_only=True)
        stds = feature_frame.std(numeric_only=True).replace(0, 1.0)

        for feature, importance in zip(features, model.feature_importances_):
            current_value = float(input_frame.at[0, feature])
            mean = float(means.get(feature, 0.0))
            std = float(stds.get(feature, 1.0))
            z_score = abs(current_value - mean) / std if std > 0 and np.isfinite(current_value) else 0.0
            contribution = float(importance * z_score)
            if not np.isfinite(contribution):
                contribution = 0.0

            contributors.append(
                {
                    "feature": feature,
                    "value": round(current_value, 2) if np.isfinite(current_value) else None,
                    "contribution_score": contribution,
                }
            )

        contributors.sort(key=lambda item: item["contribution_score"], reverse=True)

        result = {
            "sepsis_probability": probability,
            "risk_level": risk_level,
            "top_contributors": contributors[:5],
            "patient_snapshot": {
                key: (float(value) if np.isfinite(float(value)) else None)
                for key, value in input_frame.iloc[0].items()
                if key in {"Age", "HR", "O2Sat", "Temp", "SBP", "Resp", "WBC", "Lactate"}
            },
        }
        _log(f"Prediction complete: {result['risk_level']}")
        return result
    except Exception as e:
        _log(f"Prediction error: {e}")
        raise


def get_health_status() -> dict[str, Any]:
    bundle = _find_artifact_bundle()
    dataset_exists = any(candidate.exists() for candidate in DATA_PATH_CANDIDATES)
    model_exists = any(candidate["model"].exists() for candidate in MODEL_ARTIFACT_BUNDLES)
    feature_file_exists = any(candidate["features"].exists() for candidate in MODEL_ARTIFACT_BUNDLES)
    threshold_file_exists = any(candidate["threshold"].exists() for candidate in MODEL_ARTIFACT_BUNDLES)

    resolved_dataset_path: str | None = None
    if dataset_exists:
        try:
            features = list(joblib.load(bundle["features"])) if bundle is not None else list(BASE_MODEL_INPUT_FIELDS)
            resolved_dataset_path = str(_resolve_data_path(features))
        except Exception:
            resolved_dataset_path = str(next(candidate for candidate in DATA_PATH_CANDIDATES if candidate.exists()))

    return {
        "status": "ok" if dataset_exists and bundle is not None else "degraded",
        "dataset_exists": dataset_exists,
        "model_exists": model_exists,
        "feature_file_exists": feature_file_exists,
        "threshold_file_exists": threshold_file_exists,
        "artifact_bundle": bundle["label"] if bundle is not None else None,
        "dataset_path": resolved_dataset_path,
        "model_path": str(bundle["model"]) if bundle is not None else None,
        "features_path": str(bundle["features"]) if bundle is not None else None,
        "threshold_path": str(bundle["threshold"]) if bundle is not None else None,
    }


def _dispatch_rpc(func_name: str, args: dict[str, Any]) -> Any:
    handlers: dict[str, Callable[..., Any]] = {
        "get_summary_stats": get_summary_stats,
        "get_cohort_analysis": get_cohort_analysis,
        "get_feature_importance": get_feature_importance,
        "get_prediction_schema": get_prediction_schema,
        "predict_sepsis": predict_sepsis,
    }

    handler = handlers.get(func_name)
    if handler is None:
        raise ApiError(f"Unsupported RPC function '{func_name}'.", status=HTTPStatus.NOT_FOUND)

    return handler(**args)


class SepsisRequestHandler(BaseHTTPRequestHandler):
    server_version = "SepsisBackend/1.0"

    def log_message(self, format, *args):
        # Override to suppress default logging
        pass

    def end_headers(self) -> None:
        allowed_origin = os.getenv("SEPSIS_ALLOWED_ORIGIN", "*")
        self.send_header("Access-Control-Allow-Origin", allowed_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Run-Id")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def handle(self):
        try:
            super().handle()
        except Exception as e:
            _log(f"Handler error: {e}")
            raise

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        try:
            _log(f"GET {parsed.path}")
            if parsed.path in {"/health", "/api/health"}:
                self._send_json(HTTPStatus.OK, get_health_status())
                return

            if parsed.path == "/api/summary":
                self._send_json(HTTPStatus.OK, get_summary_stats())
                return

            if parsed.path == "/api/cohorts":
                cohort_feature = params.get("cohort_feature", ["AgeGroup"])[0]
                self._send_json(
                    HTTPStatus.OK,
                    get_cohort_analysis(cohort_feature=cohort_feature),
                )
                return

            if parsed.path == "/api/feature-importance":
                raw_top_n = params.get("top_n", ["10"])[0]
                self._send_json(
                    HTTPStatus.OK,
                    get_feature_importance(top_n=int(raw_top_n)),
                )
                return

            _log(f"GET {parsed.path} - not found")
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Route '{parsed.path}' was not found."},
            )
        except Exception as exc:
            _log(f"GET error: {exc}")
            self._handle_exception(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            payload = self._read_json_body()

            if parsed.path == "/api/predict":
                _log("POST /api/predict")
                patient_data = payload.get("patient_data", payload)
                self._send_json(HTTPStatus.OK, predict_sepsis(patient_data=patient_data))
                return

            if parsed.path == "/api/rpc":
                payload_module = payload.get("module")
                func_name = payload.get("func")
                args = payload.get("args") or {}
                _log(f"POST /api/rpc - func={func_name}, args={args}")
                if not isinstance(func_name, str) or not func_name:
                    raise ApiError("RPC requests require a non-empty 'func' value.")
                if not isinstance(args, dict):
                    raise ApiError("RPC 'args' must be an object.")

                result = _dispatch_rpc(func_name, args)
                _log(f"POST /api/rpc - response ready")
                self._send_json(HTTPStatus.OK, result)
                return

            _log(f"POST {parsed.path} - not found")
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Route '{parsed.path}' was not found."},
            )
        except Exception as exc:
            _log(f"POST error: {exc}")
            self._handle_exception(exc)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError("Request body must be valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ApiError("Request body must be a JSON object.")

        return payload

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        encoded = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_exception(self, exc: Exception) -> None:
        import traceback
        if isinstance(exc, ApiError):
            _log(f"Client error: {exc}")
            self._send_json(exc.status, {"error": str(exc)})
            return

        _log(f"Unhandled error: {exc}")
        _log(f"Traceback: {traceback.format_exc()}")
        self._send_json(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {"error": "Internal server error while processing the request."},
        )


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    artifact_bundle = _resolve_artifact_bundle()
    data_path = _resolve_data_path(_load_feature_names())
    _log(f"Starting backend on http://{host}:{port}")
    _log(f"Dataset path: {data_path or 'NOT FOUND (will use cache)'}")
    _log(f"Model path: {artifact_bundle['model']}")

    server = HTTPServer((host, port), SepsisRequestHandler)
    _log("Server started successfully")
    server.serve_forever()


if __name__ == "__main__":
    run_server(
        host=os.getenv("SEPSIS_BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("SEPSIS_BACKEND_PORT", "8000")),
    )
