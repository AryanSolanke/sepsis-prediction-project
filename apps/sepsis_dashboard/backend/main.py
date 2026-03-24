from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
DATA_PATH = REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_cleaned.csv"
MODEL_PATH = REPO_ROOT / "model_training" / "models" / "sepsis_xgb_model.joblib"
FEATURES_PATH = REPO_ROOT / "model_training" / "models" / "feature_names.joblib"

_CACHED_DF: pd.DataFrame | None = None
_CACHED_MODEL: Any | None = None
_CACHED_FEATURES: list[str] | None = None
_SUMMARY_CACHE: dict[str, Any] | None = None
_SCHEMA_CACHE: dict[str, Any] | None = None

PATIENT_CONTEXT_FIELDS = {"Age", "Gender", "Unit1", "HospAdmTime", "ICULOS"}
VITAL_SIGN_FIELDS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"}
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
    print(f"[SEPSIS_BACKEND] {message}")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ApiError(
            f"{label} not found at '{path}'. Re-run preprocessing/model training first.",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


def _load_data() -> pd.DataFrame:
    global _CACHED_DF

    if _CACHED_DF is not None:
        return _CACHED_DF

    _require_file(DATA_PATH, "Processed dataset")
    _log(f"Loading dataset from {DATA_PATH}")
    _CACHED_DF = pd.read_csv(DATA_PATH)
    return _CACHED_DF


def _load_model_and_features() -> tuple[Any, list[str]]:
    global _CACHED_MODEL, _CACHED_FEATURES

    if _CACHED_MODEL is not None and _CACHED_FEATURES is not None:
        return _CACHED_MODEL, _CACHED_FEATURES

    _require_file(MODEL_PATH, "Model artifact")
    _require_file(FEATURES_PATH, "Feature list")

    _log(f"Loading model from {MODEL_PATH}")
    _CACHED_MODEL = joblib.load(MODEL_PATH)
    if hasattr(_CACHED_MODEL, "n_jobs"):
        _CACHED_MODEL.n_jobs = 1
    _CACHED_FEATURES = list(joblib.load(FEATURES_PATH))
    return _CACHED_MODEL, _CACHED_FEATURES


def _get_feature_frame() -> pd.DataFrame:
    df = _load_data()
    _, features = _load_model_and_features()

    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ApiError(
            f"Dataset is missing {len(missing_features)} trained feature columns.",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    return df[features].copy()


def _get_group_holdout_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = _load_data()
    _, features = _load_model_and_features()

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
    _, features = _load_model_and_features()
    input_features = [feature for feature in features if not feature.endswith("_Measured")]
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
        "model_name": MODEL_PATH.stem,
        "feature_count": len(features),
        "primary_feature_count": len(input_features),
        "measurement_flag_count": len([feature for feature in features if feature.endswith("_Measured")]),
        "input_groups": INPUT_GROUPS,
        "fields": fields,
    }
    return _SCHEMA_CACHE


def _normalize_patient_input(
    patient_data: dict[str, Any],
    features: list[str],
) -> pd.DataFrame:
    if not isinstance(patient_data, dict):
        raise ApiError("'patient_data' must be a JSON object.")

    normalized: dict[str, list[float]] = {}
    for feature in features:
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

    return pd.DataFrame(normalized)


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
    model, _ = _load_model_and_features()
    _, X_test, _, y_test = _get_group_holdout_split()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    _SUMMARY_CACHE = {
        "total_patients": int(df["Patient_ID"].nunique()),
        "sepsis_rate": float(df["SepsisLabel"].mean()),
        "avg_age": float(df["Age"].mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "model_name": MODEL_PATH.stem,
        "evaluation_scope": "group_holdout_patient_split",
    }
    return _SUMMARY_CACHE


def get_cohort_analysis(cohort_feature: str = "AgeGroup") -> list[dict[str, Any]]:
    df = _load_data().copy()
    model, features = _load_model_and_features()

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
        y_pred = model.predict(X_group)
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
    model, features = _load_model_and_features()
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
    feature_frame = _get_feature_frame()
    model, features = _load_model_and_features()
    input_frame = _normalize_patient_input(patient_data, features)

    probability = float(model.predict_proba(input_frame)[0, 1])
    risk_level = "Low"
    if probability >= 0.4:
        risk_level = "High"
    elif probability >= 0.15:
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

    return {
        "sepsis_probability": probability,
        "risk_level": risk_level,
        "top_contributors": contributors[:5],
        "patient_snapshot": {
            key: (float(value) if np.isfinite(float(value)) else None)
            for key, value in input_frame.iloc[0].items()
            if key in {"Age", "HR", "O2Sat", "Temp", "SBP", "Resp", "WBC", "Lactate"}
        },
    }


def get_health_status() -> dict[str, Any]:
    return {
        "status": "ok",
        "dataset_exists": DATA_PATH.exists(),
        "model_exists": MODEL_PATH.exists(),
        "feature_file_exists": FEATURES_PATH.exists(),
        "dataset_path": str(DATA_PATH),
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
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

    def end_headers(self) -> None:
        allowed_origin = os.getenv("SEPSIS_ALLOWED_ORIGIN", "*")
        self.send_header("Access-Control-Allow-Origin", allowed_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Run-Id")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        try:
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

            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Route '{parsed.path}' was not found."},
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_exception(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            payload = self._read_json_body()

            if parsed.path == "/api/predict":
                patient_data = payload.get("patient_data", payload)
                self._send_json(HTTPStatus.OK, predict_sepsis(patient_data=patient_data))
                return

            if parsed.path == "/api/rpc":
                func_name = payload.get("func")
                args = payload.get("args") or {}
                if not isinstance(func_name, str) or not func_name:
                    raise ApiError("RPC requests require a non-empty 'func' value.")
                if not isinstance(args, dict):
                    raise ApiError("RPC 'args' must be an object.")

                self._send_json(HTTPStatus.OK, _dispatch_rpc(func_name, args))
                return

            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Route '{parsed.path}' was not found."},
            )
        except Exception as exc:  # noqa: BLE001
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
        if isinstance(exc, ApiError):
            _log(f"Client error: {exc}")
            self._send_json(exc.status, {"error": str(exc)})
            return

        _log(f"Unhandled error: {exc}")
        self._send_json(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {"error": "Internal server error while processing the request."},
        )


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    _log(f"Starting backend on http://{host}:{port}")
    _log(f"Dataset path: {DATA_PATH}")
    _log(f"Model path: {MODEL_PATH}")

    with ThreadingHTTPServer((host, port), SepsisRequestHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    run_server(
        host=os.getenv("SEPSIS_BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("SEPSIS_BACKEND_PORT", "8000")),
    )
