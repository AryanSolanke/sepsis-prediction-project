from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for the deep learning training scripts. "
        "Install PyTorch first, then rerun the model script."
    ) from exc


RANDOM_STATE = 42
TARGET_COL = "SepsisLabel"
GROUP_COL = "Patient_ID"
TIME_COL = "ICULOS"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "Datasets" / "processed" / "sepsis_icu_cleaned.csv"
MODEL_DIR = SCRIPT_DIR / "models"


@dataclass
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_names: list[str]


class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets_np = targets.astype(np.float32)
        self.targets = torch.from_numpy(self.targets_np)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class PatientSequenceDataset(Dataset):
    def __init__(
        self,
        features_by_patient: list[np.ndarray],
        targets_by_patient: list[np.ndarray],
        window_size: int,
    ) -> None:
        self.features_by_patient = features_by_patient
        self.targets_by_patient = targets_by_patient
        self.window_size = window_size
        self.lengths = np.asarray([len(patient) for patient in targets_by_patient], dtype=np.int64)
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.feature_dim = int(features_by_patient[0].shape[1]) if features_by_patient else 0
        self.flat_targets = (
            np.concatenate(targets_by_patient).astype(np.float32)
            if targets_by_patient
            else np.empty(0, dtype=np.float32)
        )

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patient_idx = int(np.searchsorted(self.cumulative_lengths, index, side="right"))
        patient_start = int(self.cumulative_lengths[patient_idx - 1]) if patient_idx > 0 else 0
        time_idx = index - patient_start

        patient_features = self.features_by_patient[patient_idx]
        patient_targets = self.targets_by_patient[patient_idx]

        start_idx = max(0, time_idx - self.window_size + 1)
        window = patient_features[start_idx : time_idx + 1]

        padded_window = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
        padded_window[-window.shape[0] :] = window

        return (
            torch.from_numpy(padded_window),
            torch.tensor(patient_targets[time_idx], dtype=torch.float32),
        )


class BinaryFocalLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        return (focal_factor * bce).mean()


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataframe(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    sort_columns = [GROUP_COL]
    if TIME_COL in df.columns:
        sort_columns.append(TIME_COL)
    return df.sort_values(sort_columns).reset_index(drop=True)


def infer_feature_names(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in [GROUP_COL, TARGET_COL]]


def group_train_val_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> SplitFrames:
    feature_names = infer_feature_names(df)

    initial_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(initial_split.split(df[feature_names], df[TARGET_COL], groups=df[GROUP_COL]))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    val_split = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(
        val_split.split(
            train_val_df[feature_names],
            train_val_df[TARGET_COL],
            groups=train_val_df[GROUP_COL],
        )
    )

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return SplitFrames(train=train_df, val=val_df, test=test_df, feature_names=feature_names)


def build_scaler(train_df: pd.DataFrame, feature_names: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_names])
    return scaler


def transform_tabular_frame(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    features = scaler.transform(frame[feature_names]).astype(np.float32)
    targets = frame[TARGET_COL].to_numpy(dtype=np.float32)
    return features, targets


def frame_to_patient_sequences(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_names: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    features_by_patient: list[np.ndarray] = []
    targets_by_patient: list[np.ndarray] = []

    for _, patient_frame in frame.groupby(GROUP_COL, sort=False):
        patient_features = scaler.transform(patient_frame[feature_names]).astype(np.float32)
        patient_targets = patient_frame[TARGET_COL].to_numpy(dtype=np.float32)
        features_by_patient.append(patient_features)
        targets_by_patient.append(patient_targets)

    return features_by_patient, targets_by_patient


def dataset_targets_array(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets_np"):
        return np.asarray(dataset.targets_np, dtype=np.float32)
    if hasattr(dataset, "flat_targets"):
        return np.asarray(dataset.flat_targets, dtype=np.float32)
    raise TypeError("Dataset does not expose target labels for balanced sampling.")


def balanced_sample_weights(targets: pd.Series | np.ndarray) -> np.ndarray:
    target_array = np.asarray(targets).astype(int)
    positive_count = max(int((target_array == 1).sum()), 1)
    negative_count = max(int((target_array == 0).sum()), 1)

    negative_weight = 0.5 / negative_count
    positive_weight = 0.5 / positive_count
    return np.where(target_array == 1, positive_weight, negative_weight).astype(np.float64)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def raw_positive_ratio(targets: pd.Series | np.ndarray) -> float:
    target_array = np.asarray(targets)
    positive_count = float((target_array == 1).sum())
    negative_count = float((target_array == 0).sum())
    if positive_count == 0:
        return 1.0
    return negative_count / positive_count


def derive_positive_class_weight(
    targets: pd.Series | np.ndarray,
    strategy: str = "sqrt",
    max_weight: float = 20.0,
) -> float:
    ratio = raw_positive_ratio(targets)

    if strategy == "none":
        weight = 1.0
    elif strategy == "full":
        weight = ratio
    elif strategy == "sqrt":
        weight = np.sqrt(ratio)
    elif strategy == "log":
        weight = np.log1p(ratio)
    else:
        raise ValueError(f"Unsupported class-weight strategy: {strategy}")

    return float(min(weight, max_weight))


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_criterion(loss_name: str, pos_weight: float, focal_gamma: float) -> nn.Module:
    if loss_name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    if loss_name == "focal":
        return BinaryFocalLoss(pos_weight=pos_weight, gamma=focal_gamma)
    raise ValueError(f"Unsupported loss: {loss_name}")


def _safe_metric(metric_fn: Any, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, y_prob))
    except ValueError:
        return float("nan")


def metrics_from_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    loss: float | None = None,
) -> dict[str, Any]:
    threshold = float(np.clip(threshold, 0.0, 1.0))
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "loss": float(loss) if loss is not None else float("nan"),
        "threshold": threshold,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_prob),
        "average_precision": _safe_metric(average_precision_score, y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else 0.0,
    }


def select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "precision_at_recall",
    target_recall: float = 0.70,
    beta: float = 1.0,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if len(np.unique(y_true)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    if objective == "precision_at_recall":
        feasible = recall >= target_recall
        if feasible.any():
            candidate_precision = np.where(feasible, precision, -np.inf)
            best_index = int(np.nanargmax(candidate_precision))
        else:
            best_index = int(np.nanargmax(recall))
    else:
        if objective == "f1":
            beta = 1.0
        elif objective == "f2":
            beta = 2.0

        beta_sq = beta**2
        denominator = beta_sq * precision + recall
        f_beta = np.where(denominator > 0, (1.0 + beta_sq) * precision * recall / denominator, 0.0)
        best_index = int(np.nanargmax(f_beta))

    chosen_threshold = float(np.clip(thresholds[best_index], 0.01, 0.99))
    return chosen_threshold


@torch.no_grad()
def predict_proba(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for batch_features, batch_targets in data_loader:
        batch_features = batch_features.to(device)
        logits = model(batch_features)
        probabilities = torch.sigmoid(logits).cpu().numpy()

        all_probabilities.append(probabilities)
        all_targets.append(batch_targets.numpy())

    return np.concatenate(all_targets), np.concatenate(all_probabilities)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for batch_features, batch_targets in data_loader:
        batch_features = batch_features.to(device)
        batch_targets = batch_targets.to(device)

        logits = model(batch_features)
        probabilities = torch.sigmoid(logits)

        if criterion is not None:
            loss = criterion(logits, batch_targets)
            total_loss += float(loss.item()) * int(batch_targets.size(0))
            total_examples += int(batch_targets.size(0))

        all_targets.append(batch_targets.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    loss_value = total_loss / total_examples if total_examples else float("nan")
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probabilities)
    return metrics_from_probabilities(y_true, y_prob, threshold=threshold, loss=loss_value)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    pos_weight: float,
    patience: int,
    loss_name: str = "focal",
    focal_gamma: float = 2.0,
    threshold_objective: str = "precision_at_recall",
    target_recall: float = 0.70,
    threshold_beta: float = 1.0,
    grad_clip: float | None = 1.0,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any]]:
    model.to(device)
    criterion = build_criterion(loss_name=loss_name, pos_weight=pos_weight, focal_gamma=focal_gamma).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, patience // 2),
    )

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_metrics: dict[str, Any] = {}
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            running_loss += float(loss.item()) * int(batch_targets.size(0))
            total_examples += int(batch_targets.size(0))

        train_loss = running_loss / total_examples if total_examples else float("nan")
        val_base_metrics = evaluate_model(model, val_loader, device, criterion=criterion, threshold=0.5)
        tuned_threshold = select_decision_threshold(
            val_base_metrics["y_true"],
            val_base_metrics["y_prob"],
            objective=threshold_objective,
            target_recall=target_recall,
            beta=threshold_beta,
        )
        val_metrics = metrics_from_probabilities(
            val_base_metrics["y_true"],
            val_base_metrics["y_prob"],
            threshold=tuned_threshold,
            loss=val_base_metrics["loss"],
        )

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_threshold": float(val_metrics["threshold"]),
            "val_roc_auc": float(val_metrics["roc_auc"]),
            "val_average_precision": float(val_metrics["average_precision"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_predicted_positive_rate": float(val_metrics["predicted_positive_rate"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.4f} | "
            f"val_pr_auc={val_metrics['average_precision']:.4f} | "
            f"val_threshold={val_metrics['threshold']:.3f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        current_score = float(val_metrics["average_precision"])
        scheduler.step(current_score)

        if current_score > best_score:
            best_score = current_score
            best_metrics = {
                **val_metrics,
                "threshold_objective": threshold_objective,
                "target_recall": float(target_recall),
                "loss_name": loss_name,
                "pos_weight": float(pos_weight),
            }
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    model.load_state_dict(best_state)
    return model, history, best_metrics


def print_final_report(
    model_name: str,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
) -> None:
    print(f"\n--- {model_name} Performance ---")
    print(classification_report(metrics["y_true"], metrics["y_pred"], zero_division=0))
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC (Avg Precision): {metrics['average_precision']:.4f}")

    if baseline_metrics is not None:
        print(
            f"Precision @ 0.50: {baseline_metrics['precision']:.4f} | "
            f"Recall @ 0.50: {baseline_metrics['recall']:.4f} | "
            f"F1 @ 0.50: {baseline_metrics['f1']:.4f}"
        )

    print(
        f"Precision @ {metrics['threshold']:.3f}: {metrics['precision']:.4f} | "
        f"Recall @ {metrics['threshold']:.3f}: {metrics['recall']:.4f} | "
        f"F1 @ {metrics['threshold']:.3f}: {metrics['f1']:.4f}"
    )


def save_training_artifacts(
    model_name: str,
    model: nn.Module,
    scaler: StandardScaler,
    feature_names: list[str],
    history: list[dict[str, float]],
    metrics: dict[str, Any],
    model_kwargs: dict[str, Any],
    output_dir: Path = MODEL_DIR,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"sepsis_{model_name}.pt"
    feature_names_path = output_dir / f"{model_name}_feature_names.joblib"
    scaler_path = output_dir / f"{model_name}_scaler.joblib"
    metrics_path = output_dir / f"{model_name}_metrics.json"
    history_path = output_dir / f"{model_name}_history.json"

    serializable_metrics = {
        "roc_auc": float(metrics["roc_auc"]),
        "average_precision": float(metrics["average_precision"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "threshold": float(metrics.get("threshold", 0.5)),
        "predicted_positive_rate": float(metrics.get("predicted_positive_rate", 0.0)),
        "threshold_objective": metrics.get("threshold_objective", ""),
        "target_recall": float(metrics.get("target_recall", 0.0)),
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_kwargs": model_kwargs,
        "feature_names": feature_names,
        "metrics": serializable_metrics,
        "history": history,
    }
    if extra_metadata:
        checkpoint.update(extra_metadata)

    torch.save(checkpoint, checkpoint_path)
    joblib.dump(feature_names, feature_names_path)
    joblib.dump(scaler, scaler_path)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable_metrics, handle, indent=2)

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return {
        "checkpoint_path": str(checkpoint_path),
        "feature_names_path": str(feature_names_path),
        "scaler_path": str(scaler_path),
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
    }
