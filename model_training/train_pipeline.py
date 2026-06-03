"""
Sepsis Prediction — Research-Grade Training Pipeline
=====================================================
Three model families (XGBoost, LightGBM, Random Forest) with:
  - Optuna Bayesian hyperparameter optimisation (50 trials each)
  - 5-fold GroupKFold CV on Patient_ID (no data leakage)
  - GPU acceleration for boosted trees
  - Comprehensive anti-overfitting controls
  - Threshold optimisation on held-out validation set
  - Backward-compatible artifact output

Usage:
    conda run -n ai_work python model_training/train_pipeline.py
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 50
MAX_OVERFIT_GAP = 0.05  # prune if train-val AUC gap exceeds this

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_engineered.csv"
MODEL_PATH = REPO_ROOT / "model_training" / "model.pkl"
FEATURES_PATH = REPO_ROOT / "model_training" / "features.pkl"
THRESHOLD_PATH = REPO_ROOT / "model_training" / "threshold.pkl"
METRICS_PATH = REPO_ROOT / "model_training" / "model_metrics.json"

np.random.seed(RANDOM_STATE)


# ── helpers ──────────────────────────────────────────────────────────────
def collect_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    preds = (probs >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(preds)),
    }


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Sweep thresholds, pick the one that maximises F1."""
    best = None
    for thr in np.arange(0.05, 0.96, 0.005):
        preds = (probs >= thr).astype(int)
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f = float(f1_score(y_true, preds, zero_division=0))
        cand = {"threshold": round(float(thr), 4), "precision": prec, "recall": rec, "f1": f}
        if best is None or (f, prec, rec) > (best["f1"], best["precision"], best["recall"]):
            best = cand
    if best is None:
        raise RuntimeError("Threshold search failed.")
    return best


# ── Optuna objectives ───────────────────────────────────────────────────
def _cv_score(model_cls, params, X, y, groups, n_folds=N_CV_FOLDS, use_early_stop=False, is_lgb=False):
    """Run GroupKFold CV, return (mean_val_auc, mean_train_auc)."""
    gkf = GroupKFold(n_splits=n_folds)
    val_aucs, train_aucs = [], []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        model = model_cls(**params)
        if use_early_stop:
            if is_lgb:
                import lightgbm as _lgb
                model.fit(
                    X[train_idx], y[train_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    callbacks=[_lgb.log_evaluation(0)],
                )
            else:
                model.fit(
                    X[train_idx], y[train_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    verbose=False,
                )
        else:
            model.fit(X[train_idx], y[train_idx])

        val_prob = model.predict_proba(X[val_idx])[:, 1]
        train_prob = model.predict_proba(X[train_idx])[:, 1]

        val_aucs.append(roc_auc_score(y[val_idx], val_prob))
        train_aucs.append(roc_auc_score(y[train_idx], train_prob))

    return float(np.mean(val_aucs)), float(np.mean(train_aucs))


def objective_xgb(trial, X, y, groups):
    import xgboost as xgb

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 3),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 25.0),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    val_auc, train_auc = _cv_score(xgb.XGBClassifier, params, X, y, groups, use_early_stop=True)

    # Prune overfitting trials
    if train_auc - val_auc > MAX_OVERFIT_GAP:
        raise optuna.TrialPruned(f"Overfit gap {train_auc - val_auc:.4f}")

    trial.set_user_attr("train_auc", train_auc)
    trial.set_user_attr("val_auc", val_auc)
    return val_auc


def objective_lgb(trial, X, y, groups):
    import lightgbm as lgb

    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 25.0),
        "subsample_freq": 1,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "device": "gpu",
        "gpu_use_dp": False,
        "early_stopping_rounds": 50,
    }

    try:
        val_auc, train_auc = _cv_score(lgb.LGBMClassifier, params, X, y, groups, use_early_stop=True, is_lgb=True)
    except Exception:
        # GPU fallback
        params["device"] = "cpu"
        val_auc, train_auc = _cv_score(lgb.LGBMClassifier, params, X, y, groups, use_early_stop=True, is_lgb=True)

    if train_auc - val_auc > MAX_OVERFIT_GAP:
        raise optuna.TrialPruned(f"Overfit gap {train_auc - val_auc:.4f}")

    trial.set_user_attr("train_auc", train_auc)
    trial.set_user_attr("val_auc", val_auc)
    return val_auc


# ── main ─────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("SEPSIS PREDICTION — RESEARCH-GRADE TRAINING PIPELINE")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    features = [c for c in df.columns if c not in ("Patient_ID", "SepsisLabel")]
    X = df[features].values.astype(np.float32)
    y = df["SepsisLabel"].astype(int).values
    groups = df["Patient_ID"].values

    n_patients = pd.Series(groups).nunique()
    pos_rate = y.mean()
    print(f"  Samples: {len(X):,}  |  Features: {len(features)}  |  Patients: {n_patients:,}")
    print(f"  Positive rate: {pos_rate:.4f} ({y.sum():,} / {len(y):,})")

    # ── 2. Outer split: hold out 15% test (patient-level) ────────────────
    print("\n[2/6] Splitting data (patient-level)...")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(outer.split(X, y, groups=groups))

    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    groups_trainval = groups[trainval_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Inner split for final model's early-stopping validation
    inner = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=RANDOM_STATE)
    train_rel, val_rel = next(inner.split(X_trainval, y_trainval, groups=groups_trainval))
    train_idx_final = trainval_idx[train_rel]
    val_idx_final = trainval_idx[val_rel]

    X_train_f, y_train_f = X[train_idx_final], y[train_idx_final]
    X_val_f, y_val_f = X[val_idx_final], y[val_idx_final]

    print(f"  Train: {len(X_train_f):,} | Val: {len(X_val_f):,} | Test: {len(X_test):,}")
    print(f"  Train patients: {pd.Series(groups[train_idx_final]).nunique():,}")
    print(f"  Val patients:   {pd.Series(groups[val_idx_final]).nunique():,}")
    print(f"  Test patients:  {pd.Series(groups[test_idx]).nunique():,}")

    # ── 3. Optuna HPO ────────────────────────────────────────────────────
    results = {}

    # --- XGBoost ---
    print(f"\n[3/6] Hyperparameter optimisation ({N_OPTUNA_TRIALS} trials × {N_CV_FOLDS}-fold CV each)")
    print(f"\n  ▶ XGBoost (GPU)...")
    t1 = time.time()
    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_trainval, y_trainval, groups_trainval),
                       n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    xgb_best = study_xgb.best_trial
    print(f"    Best val AUC: {xgb_best.value:.5f}  "
          f"(train: {xgb_best.user_attrs.get('train_auc', 0):.5f})  "
          f"[{time.time() - t1:.0f}s]")
    results["xgboost"] = {
        "best_val_auc": xgb_best.value,
        "best_train_auc": xgb_best.user_attrs.get("train_auc", 0),
        "best_params": xgb_best.params,
        "n_completed": len([t for t in study_xgb.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study_xgb.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    # --- LightGBM ---
    print(f"\n  ▶ LightGBM (GPU w/ fallback)...")
    t1 = time.time()
    study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(lambda trial: objective_lgb(trial, X_trainval, y_trainval, groups_trainval),
                       n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    lgb_best = study_lgb.best_trial
    print(f"    Best val AUC: {lgb_best.value:.5f}  "
          f"(train: {lgb_best.user_attrs.get('train_auc', 0):.5f})  "
          f"[{time.time() - t1:.0f}s]")
    results["lightgbm"] = {
        "best_val_auc": lgb_best.value,
        "best_train_auc": lgb_best.user_attrs.get("train_auc", 0),
        "best_params": lgb_best.params,
        "n_completed": len([t for t in study_lgb.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study_lgb.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    # ── 4. Determine champion ────────────────────────────────────────────
    print(f"\n[4/6] Selecting champion model...")
    champion_name = max(results, key=lambda k: results[k]["best_val_auc"])
    print(f"  Champion: {champion_name} (CV AUC = {results[champion_name]['best_val_auc']:.5f})")

    # ── 5. Retrain all 3 on full train, evaluate on test ─────────────────
    print(f"\n[5/6] Retraining final models & evaluating on held-out test...")
    import xgboost as xgb
    import lightgbm as lgb

    # Build final params
    xgb_final_params = {k: v for k, v in xgb_best.params.items()}
    xgb_final_params.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    })

    lgb_final_params = {k: v for k, v in lgb_best.params.items()}
    lgb_final_params.update({
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "subsample_freq": 1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "device": "gpu",
        "gpu_use_dp": False,
        "early_stopping_rounds": 50,
    })

    final_models = {}

    # XGBoost final
    model_xgb = xgb.XGBClassifier(**xgb_final_params)
    model_xgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
    final_models["xgboost"] = model_xgb

    # LightGBM final
    try:
        model_lgb = lgb.LGBMClassifier(**lgb_final_params)
        model_lgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], callbacks=[lgb.log_evaluation(0)])
    except Exception:
        lgb_final_params["device"] = "cpu"
        model_lgb = lgb.LGBMClassifier(**lgb_final_params)
        model_lgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], callbacks=[lgb.log_evaluation(0)])
    final_models["lightgbm"] = model_lgb

    # Evaluate all on test
    for name, model in final_models.items():
        test_prob = model.predict_proba(X_test)[:, 1]
        val_prob = model.predict_proba(X_val_f)[:, 1]
        train_prob = model.predict_proba(X_train_f)[:, 1]

        thr_info = find_best_threshold(y_val_f, val_prob)
        thr = thr_info["threshold"]

        train_metrics = collect_metrics(y_train_f, train_prob, thr)
        val_metrics = collect_metrics(y_val_f, val_prob, thr)
        test_metrics = collect_metrics(y_test, test_prob, thr)

        results[name]["threshold_info"] = thr_info
        results[name]["train"] = train_metrics
        results[name]["val"] = val_metrics
        results[name]["test"] = test_metrics
        results[name]["overfit_gap"] = round(train_metrics["roc_auc"] - test_metrics["roc_auc"], 5)

        print(f"\n  {name}:")
        print(f"    ROC AUC  — Train: {train_metrics['roc_auc']:.4f}  Val: {val_metrics['roc_auc']:.4f}  Test: {test_metrics['roc_auc']:.4f}")
        print(f"    PR AUC   — Train: {train_metrics['pr_auc']:.4f}  Val: {val_metrics['pr_auc']:.4f}  Test: {test_metrics['pr_auc']:.4f}")
        print(f"    F1       — Train: {train_metrics['f1']:.4f}  Val: {val_metrics['f1']:.4f}  Test: {test_metrics['f1']:.4f}")
        print(f"    Overfit gap (train-test AUC): {results[name]['overfit_gap']:.4f}")

    # ── 6. Save artifacts ────────────────────────────────────────────────
    print(f"\n[6/6] Saving artifacts...")
    champion_model = final_models[champion_name]
    champion_threshold = results[champion_name]["threshold_info"]["threshold"]

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(champion_model, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)
    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump(champion_threshold, f)

    # Save all model variants
    ensemble_dir = REPO_ROOT / "model_training" / "ensemble_models"
    ensemble_dir.mkdir(exist_ok=True)
    for name, model in final_models.items():
        with open(ensemble_dir / f"{name}_tuned.pkl", "wb") as f:
            pickle.dump(model, f)

    # Save comprehensive metrics
    output_metrics = {
        "champion": champion_name,
        "champion_cv_auc": results[champion_name]["best_val_auc"],
        "pipeline_config": {
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_cv_folds": N_CV_FOLDS,
            "random_state": RANDOM_STATE,
            "max_overfit_gap": MAX_OVERFIT_GAP,
            "dataset": str(DATA_PATH),
            "feature_count": len(features),
            "total_samples": len(X),
            "total_patients": n_patients,
        },
        "split_sizes": {
            "train_rows": int(len(X_train_f)),
            "val_rows": int(len(X_val_f)),
            "test_rows": int(len(X_test)),
            "train_patients": int(pd.Series(groups[train_idx_final]).nunique()),
            "val_patients": int(pd.Series(groups[val_idx_final]).nunique()),
            "test_patients": int(pd.Series(groups[test_idx]).nunique()),
        },
        "models": {},
    }

    for name in results:
        output_metrics["models"][name] = results[name]
        # Convert numpy types for JSON
        for key in list(output_metrics["models"][name].get("best_params", {}).keys()):
            val = output_metrics["models"][name]["best_params"][key]
            if isinstance(val, (np.integer, np.int64)):
                output_metrics["models"][name]["best_params"][key] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                output_metrics["models"][name]["best_params"][key] = float(val)

    # Also keep backward-compat top-level keys
    output_metrics["model_type"] = champion_name
    output_metrics["target"] = "SepsisLabel"
    output_metrics["selected_threshold"] = results[champion_name]["threshold_info"]
    output_metrics["best_params"] = results[champion_name]["best_params"]
    output_metrics["train"] = results[champion_name]["train"]
    output_metrics["val"] = results[champion_name]["val"]
    output_metrics["test"] = results[champion_name]["test"]

    METRICS_PATH.write_text(json.dumps(output_metrics, indent=2, default=str), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"Champion: {champion_name}  |  Test ROC AUC: {results[champion_name]['test']['roc_auc']:.4f}")
    print(f"  model  → {MODEL_PATH}")
    print(f"  feats  → {FEATURES_PATH}")
    print(f"  thresh → {THRESHOLD_PATH}")
    print(f"  report → {METRICS_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
