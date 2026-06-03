from __future__ import annotations

from flask import Flask, jsonify, request

try:
    from .main import (
        ApiError,
        _dispatch_rpc,
        _json_safe,
        get_cohort_analysis,
        get_feature_importance,
        get_health_status,
        get_summary_stats,
        predict_sepsis,
    )
except ImportError:
    from main import (  # type: ignore
        ApiError,
        _dispatch_rpc,
        _json_safe,
        get_cohort_analysis,
        get_feature_importance,
        get_health_status,
        get_summary_stats,
        predict_sepsis,
    )


app = Flask(__name__)


@app.errorhandler(ApiError)
def handle_api_error(exc: ApiError):
    return jsonify({"error": str(exc)}), int(exc.status)


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    return jsonify({"error": "Internal server error while processing the request."}), 500


@app.route("/health")
@app.route("/api/health")
def health():
    return jsonify(_json_safe(get_health_status()))


@app.route("/api/summary")
def summary():
    return jsonify(_json_safe(get_summary_stats()))


@app.route("/api/cohorts")
def cohorts():
    cohort_feature = request.args.get("cohort_feature", "AgeGroup")
    return jsonify(_json_safe(get_cohort_analysis(cohort_feature=cohort_feature)))


@app.route("/api/feature-importance")
def feature_importance():
    top_n = int(request.args.get("top_n", "10"))
    return jsonify(_json_safe(get_feature_importance(top_n=top_n)))


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    patient_data = payload.get("patient_data", payload)
    return jsonify(_json_safe(predict_sepsis(patient_data=patient_data)))


@app.route("/api/rpc", methods=["POST"])
def rpc():
    payload = request.get_json(silent=True) or {}
    func_name = payload.get("func")
    args = payload.get("args") or {}
    if not isinstance(func_name, str) or not func_name:
        raise ApiError("RPC requests require a non-empty 'func' value.")
    if not isinstance(args, dict):
        raise ApiError("RPC 'args' must be an object.")

    return jsonify(_json_safe(_dispatch_rpc(func_name, args)))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, threaded=True)
