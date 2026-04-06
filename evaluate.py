from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.tree import plot_tree


RANDOM_STATE = 42
TARGET_COL = "SepsisLabel"
GROUP_COL = "Patient_ID"
CLASS_NAMES = ["No Sepsis", "Sepsis"]

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = REPO_ROOT / "model_training" / "models" / "sepsis_rf_physionet_best.joblib"
DEFAULT_DATA_PATH = REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_cleaned.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "random_forest_visualizations"


sns.set_theme(style="whitegrid", context="talk")


@dataclass
class EvaluationArtifacts:
    model: Any
    feature_names: list[str]
    X_test: pd.DataFrame
    y_test: pd.Series
    X_sample: pd.DataFrame
    y_sample: pd.Series
    y_pred: np.ndarray
    y_prob: np.ndarray


def resolve_feature_names(model: Any) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise ValueError(
        "The saved random forest model does not expose feature names. "
        "Please retrain it with a pandas DataFrame input or save the feature list separately."
    )


def stratified_sample(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    _, sample_idx = train_test_split(
        X.index,
        test_size=max_rows,
        stratify=y,
        random_state=random_state,
    )
    sample_idx = pd.Index(sample_idx).sort_values()
    return X.loc[sample_idx].reset_index(drop=True), y.loc[sample_idx].reset_index(drop=True)


def load_evaluation_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    data_path: Path = DEFAULT_DATA_PATH,
    sample_size: int = 5000,
) -> EvaluationArtifacts:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}")

    print(f"Loading random forest model from {model_path}...")
    model = joblib.load(model_path)
    if hasattr(model, "set_params") and "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    feature_names = resolve_feature_names(model)

    print(f"Loading evaluation data from {data_path}...")
    required_columns = feature_names + [TARGET_COL, GROUP_COL]
    df = pd.read_csv(data_path, usecols=required_columns)

    X = df[feature_names]
    y = df[TARGET_COL].astype(int)
    groups = df[GROUP_COL]

    print("Recreating the held-out GroupKFold split...")
    gkf = GroupKFold(n_splits=5)
    _, test_idx = next(gkf.split(X, y, groups=groups))
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print(f"Generating predictions for {len(X_test):,} held-out rows...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    X_sample, y_sample = stratified_sample(X_test, y_test, max_rows=sample_size)
    print(f"Using a {len(X_sample):,}-row stratified sample for the heavier plots.")

    return EvaluationArtifacts(
        model=model,
        feature_names=feature_names,
        X_test=X_test,
        y_test=y_test,
        X_sample=X_sample,
        y_sample=y_sample,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(figure_path)


def build_summary_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "test_rows": int(len(y_true)),
        "positive_cases": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision_at_0_50": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_0_50": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_0_50": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity_at_0_50": float(specificity),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_metric_outputs(
    output_dir: Path,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[str, str, dict[str, float | int]]:
    metrics = build_summary_metrics(y_true, y_pred, y_prob)
    metrics_path = output_dir / "metrics_summary.json"
    report_path = output_dir / "classification_report.csv"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).transpose()
    report_df.to_csv(report_path, index=True)

    return str(metrics_path), str(report_path), metrics


def plot_class_balance(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path) -> str:
    actual_counts = y_true.value_counts().reindex([0, 1], fill_value=0)
    predicted_counts = pd.Series(y_pred).value_counts().reindex([0, 1], fill_value=0)

    comparison_counts = pd.DataFrame(
        {
            "Actual": actual_counts.values,
            "Predicted": predicted_counts.values,
        },
        index=CLASS_NAMES,
    )
    comparison_rates = comparison_counts.div(comparison_counts.sum(axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    comparison_counts.plot(kind="bar", ax=axes[0], color=["#0f766e", "#b45309"])
    axes[0].set_title("Held-Out Fold Class Counts")
    axes[0].set_ylabel("Rows")
    axes[0].tick_params(axis="x", rotation=0)

    comparison_rates.plot(kind="bar", ax=axes[1], color=["#0f766e", "#b45309"])
    axes[1].set_title("Held-Out Fold Class Rates")
    axes[1].set_ylabel("Share of rows")
    axes[1].tick_params(axis="x", rotation=0)

    return save_figure(fig, output_dir, "01_class_balance_and_prediction_rates.png")


def plot_confusion_matrices(y_true: pd.Series, y_pred: np.ndarray, output_dir: Path) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_normalized = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title("Normalized Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    return save_figure(fig, output_dir, "02_confusion_matrices.png")


def plot_roc_and_pr(y_true: pd.Series, y_prob: np.ndarray, output_dir: Path) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(fpr, tpr, color="#ea580c", lw=2.5, label=f"ROC AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], color="#6b7280", linestyle="--", lw=1.5, label="Random baseline")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color="#047857", lw=2.5, label=f"Average Precision = {pr_auc:.4f}")
    axes[1].axhline(baseline, color="#6b7280", linestyle="--", lw=1.5, label=f"Positive rate = {baseline:.4f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    return save_figure(fig, output_dir, "03_roc_and_precision_recall_curves.png")


def plot_probability_and_calibration(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_dir: Path,
) -> str:
    calibration_x, calibration_y = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    score_df = pd.DataFrame({"True label": y_true.map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}), "Score": y_prob})
    brier = brier_score_loss(y_true, y_prob)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    sns.histplot(
        data=score_df,
        x="Score",
        hue="True label",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        palette=["#0f766e", "#dc2626"],
        ax=axes[0],
    )
    axes[0].set_title("Predicted Probability Distribution")
    axes[0].set_xlim(0, 1)

    sns.boxplot(
        data=score_df,
        x="True label",
        y="Score",
        hue="True label",
        palette=["#0f766e", "#dc2626"],
        dodge=False,
        ax=axes[1],
    )
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    axes[1].set_title("Probability by True Class")
    axes[1].set_ylim(0, 1)

    axes[2].plot([0, 1], [0, 1], linestyle="--", color="#6b7280", lw=1.5, label="Perfect calibration")
    axes[2].plot(calibration_x, calibration_y, marker="o", color="#7c3aed", lw=2.5, label=f"Brier score = {brier:.4f}")
    axes[2].set_title("Calibration Curve")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed positive rate")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper left")

    return save_figure(fig, output_dir, "04_probability_distribution_and_calibration.png")


def threshold_metrics_frame(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    thresholds = np.linspace(0, 1, 101)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "specificity": specificity,
            }
        )

    return pd.DataFrame(rows)


def gains_and_lift_frame(y_true: pd.Series, y_prob: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    ordered_true = np.asarray(y_true)[np.argsort(y_prob)[::-1]]
    cumulative_positives = np.cumsum(ordered_true)
    total_rows = len(ordered_true)
    total_positives = cumulative_positives[-1]
    total_negatives = total_rows - total_positives

    population_share = np.arange(1, total_rows + 1) / total_rows
    gains = cumulative_positives / total_positives
    negative_cumulative = np.cumsum(1 - ordered_true)
    false_positive_share = negative_cumulative / total_negatives
    true_positive_share = cumulative_positives / total_positives
    ks_values = true_positive_share - false_positive_share
    ks_index = int(np.argmax(ks_values))
    ks_max = float(ks_values[ks_index])
    ks_population_share = float(population_share[ks_index])

    return (
        pd.DataFrame(
            {
                "population_share": population_share,
                "gains": gains,
                "lift": gains / population_share,
                "true_positive_share": true_positive_share,
                "false_positive_share": false_positive_share,
                "ks": ks_values,
            }
        ),
        ks_max,
        ks_population_share,
    )


def plot_threshold_gain_lift_and_ks(y_true: pd.Series, y_prob: np.ndarray, output_dir: Path) -> str:
    threshold_df = threshold_metrics_frame(y_true, y_prob)
    gains_df, ks_max, ks_population_share = gains_and_lift_frame(y_true, y_prob)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    threshold_df.plot(
        x="threshold",
        y=["precision", "recall", "f1", "specificity"],
        ax=axes[0, 0],
        linewidth=2,
        color=["#1d4ed8", "#dc2626", "#7c3aed", "#059669"],
    )
    axes[0, 0].set_title("Metric Sensitivity to Threshold")
    axes[0, 0].set_xlabel("Classification threshold")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].plot(gains_df["population_share"], gains_df["gains"], color="#ea580c", lw=2.5, label="Model")
    axes[0, 1].plot([0, 1], [0, 1], color="#6b7280", linestyle="--", lw=1.5, label="Baseline")
    axes[0, 1].set_title("Cumulative Gains Curve")
    axes[0, 1].set_xlabel("Share of population targeted")
    axes[0, 1].set_ylabel("Share of sepsis cases captured")
    axes[0, 1].legend(loc="lower right")

    axes[1, 0].plot(gains_df["population_share"], gains_df["lift"], color="#0f766e", lw=2.5)
    axes[1, 0].axhline(1.0, color="#6b7280", linestyle="--", lw=1.5)
    axes[1, 0].set_title("Lift Curve")
    axes[1, 0].set_xlabel("Share of population targeted")
    axes[1, 0].set_ylabel("Lift over random")

    axes[1, 1].plot(
        gains_df["population_share"],
        gains_df["true_positive_share"],
        color="#dc2626",
        lw=2.5,
        label="Cumulative positive rate",
    )
    axes[1, 1].plot(
        gains_df["population_share"],
        gains_df["false_positive_share"],
        color="#2563eb",
        lw=2.5,
        label="Cumulative negative rate",
    )
    axes[1, 1].axvline(
        ks_population_share,
        color="#111827",
        linestyle="--",
        lw=1.5,
        label=f"KS max = {ks_max:.4f}",
    )
    axes[1, 1].set_title("KS Curve")
    axes[1, 1].set_xlabel("Share of population targeted")
    axes[1, 1].set_ylabel("Cumulative share")
    axes[1, 1].legend(loc="lower right")

    return save_figure(fig, output_dir, "05_threshold_gain_lift_and_ks_curves.png")


def plot_feature_importance(model: Any, feature_names: list[str], output_dir: Path, top_n: int) -> str:
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    top_features = importances.head(top_n).sort_values()
    cumulative_importance = importances.cumsum()
    features_for_80 = int((cumulative_importance <= 0.8).sum() + 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].barh(top_features.index, top_features.values, color="#7c3aed")
    axes[0].set_title(f"Top {top_n} Feature Importances")
    axes[0].set_xlabel("Mean decrease in impurity")

    axes[1].plot(
        np.arange(1, len(cumulative_importance) + 1),
        cumulative_importance.values,
        color="#0f766e",
        lw=2.5,
    )
    axes[1].axhline(0.8, color="#6b7280", linestyle="--", lw=1.5, label="80% importance")
    axes[1].axvline(features_for_80, color="#ea580c", linestyle="--", lw=1.5, label=f"{features_for_80} features")
    axes[1].set_title("Cumulative Feature Importance")
    axes[1].set_xlabel("Number of top-ranked features")
    axes[1].set_ylabel("Cumulative importance")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower right")

    return save_figure(fig, output_dir, "06_feature_importance_and_cumulative_importance.png")


def plot_permutation_importance(
    model: Any,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    output_dir: Path,
    top_n: int,
) -> str:
    importance = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean": importance.importances_mean,
                "std": importance.importances_std,
            }
        )
        .sort_values("mean", ascending=False)
        .head(top_n)
        .sort_values("mean")
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        importance_df["feature"],
        importance_df["mean"],
        xerr=importance_df["std"],
        color="#2563eb",
        ecolor="#1f2937",
        capsize=4,
    )
    ax.set_title(f"Top {top_n} Permutation Importances")
    ax.set_xlabel("Mean drop in average precision after shuffling")

    return save_figure(fig, output_dir, "07_permutation_importance.png")


def plot_partial_dependence(
    model: Any,
    X_sample: pd.DataFrame,
    output_dir: Path,
) -> str:
    X_plot = X_sample.astype(float)
    top_features = (
        pd.Series(model.feature_importances_, index=X_plot.columns)
        .sort_values(ascending=False)
        .head(4)
        .index
        .tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    PartialDependenceDisplay.from_estimator(
        model,
        X_plot,
        features=top_features,
        target=1,
        ax=axes.ravel(),
    )
    fig.suptitle("Partial Dependence for the 4 Most Important Features", y=1.02, fontsize=18)

    return save_figure(fig, output_dir, "08_partial_dependence_top_features.png")


def plot_sample_tree(model: Any, feature_names: list[str], output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(24, 14))
    plot_tree(
        model.estimators_[0],
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("One Tree from the Random Forest (depth limited to 3 for readability)")

    return save_figure(fig, output_dir, "09_sample_tree_from_random_forest.png")


def create_all_random_forest_visualizations(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    data_path: Path = DEFAULT_DATA_PATH,
    sample_size: int = 5000,
    top_n_features: int = 15,
    include_heavy_plots: bool = True,
) -> dict[str, Any]:
    artifacts = load_evaluation_artifacts(
        model_path=model_path,
        data_path=data_path,
        sample_size=sample_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path, report_path, metrics = save_metric_outputs(
        output_dir=output_dir,
        y_true=artifacts.y_test,
        y_pred=artifacts.y_pred,
        y_prob=artifacts.y_prob,
    )

    figure_paths = [
        plot_class_balance(artifacts.y_test, artifacts.y_pred, output_dir),
        plot_confusion_matrices(artifacts.y_test, artifacts.y_pred, output_dir),
        plot_roc_and_pr(artifacts.y_test, artifacts.y_prob, output_dir),
        plot_probability_and_calibration(artifacts.y_test, artifacts.y_prob, output_dir),
        plot_threshold_gain_lift_and_ks(artifacts.y_test, artifacts.y_prob, output_dir),
        plot_feature_importance(artifacts.model, artifacts.feature_names, output_dir, top_n_features),
        plot_sample_tree(artifacts.model, artifacts.feature_names, output_dir),
    ]

    if include_heavy_plots:
        figure_paths.append(
            plot_permutation_importance(
                artifacts.model,
                artifacts.X_sample,
                artifacts.y_sample,
                output_dir,
                top_n_features,
            )
        )
        figure_paths.append(
            plot_partial_dependence(
                artifacts.model,
                artifacts.X_sample,
                output_dir,
            )
        )

    print(f"Saved {len(figure_paths)} figures to {output_dir}")
    print(f"Metrics summary: {metrics_path}")
    print(f"Classification report: {report_path}")

    return {
        "output_dir": str(output_dir),
        "metrics_path": metrics_path,
        "classification_report_path": report_path,
        "figure_paths": figure_paths,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive evaluation report for the project's random forest model."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Stratified sample size for permutation importance and partial dependence plots.",
    )
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=15,
        help="How many features to show in the importance plots.",
    )
    parser.add_argument(
        "--skip-heavy-plots",
        action="store_true",
        help="Skip permutation importance and partial dependence if you only want the faster plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_all_random_forest_visualizations(
        output_dir=args.output_dir,
        model_path=args.model_path,
        data_path=args.data_path,
        sample_size=args.sample_size,
        top_n_features=args.top_n_features,
        include_heavy_plots=not args.skip_heavy_plots,
    )


if __name__ == "__main__":
    main()
