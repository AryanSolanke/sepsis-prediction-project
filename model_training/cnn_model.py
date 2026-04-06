from __future__ import annotations

import argparse

from deep_learning_common import (
    MODEL_DIR,
    PatientSequenceDataset,
    SplitFrames,
    balanced_sample_weights,
    build_scaler,
    dataset_targets_array,
    derive_positive_class_weight,
    evaluate_model,
    frame_to_patient_sequences,
    group_train_val_test_split,
    load_dataframe,
    make_dataloader,
    nn,
    print_final_report,
    raw_positive_ratio,
    resolve_device,
    save_training_artifacts,
    set_seed,
    torch,
    train_model,
)


class SepsisTemporalCNN(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        avg_pool = encoded.mean(dim=2)
        max_pool = encoded.amax(dim=2)
        features = torch.cat([avg_pool, max_pool], dim=1)
        features = self.dropout(features)
        return self.classifier(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal CNN for sepsis prediction.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--loss", choices=["focal", "weighted_bce"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weight-strategy", choices=["none", "log", "sqrt", "full"], default="sqrt")
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--threshold-objective", choices=["precision_at_recall", "f1", "f2"], default="precision_at_recall")
    parser.add_argument("--target-recall", type=float, default=0.70)
    parser.add_argument("--threshold-beta", type=float, default=1.0)
    parser.add_argument("--balanced-sampling", action="store_true")
    return parser.parse_args()


def build_dataloaders(split_frames: SplitFrames, batch_size: int, window_size: int, balanced_sampling: bool):
    scaler = build_scaler(split_frames.train, split_frames.feature_names)

    train_features, train_targets = frame_to_patient_sequences(split_frames.train, scaler, split_frames.feature_names)
    val_features, val_targets = frame_to_patient_sequences(split_frames.val, scaler, split_frames.feature_names)
    test_features, test_targets = frame_to_patient_sequences(split_frames.test, scaler, split_frames.feature_names)

    train_dataset = PatientSequenceDataset(train_features, train_targets, window_size=window_size)
    val_dataset = PatientSequenceDataset(val_features, val_targets, window_size=window_size)
    test_dataset = PatientSequenceDataset(test_features, test_targets, window_size=window_size)

    train_weights = None
    if balanced_sampling:
        train_weights = balanced_sample_weights(dataset_targets_array(train_dataset))

    train_loader = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not balanced_sampling,
        sample_weights=train_weights,
    )
    val_loader = make_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = make_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    return scaler, train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    set_seed()
    device = resolve_device(args.device)

    print("Loading processed data...")
    df = load_dataframe()
    split_frames = group_train_val_test_split(df)

    print(
        f"Train rows: {len(split_frames.train):,} | "
        f"Val rows: {len(split_frames.val):,} | "
        f"Test rows: {len(split_frames.test):,}"
    )
    print(
        f"Window size: {args.window_size} | "
        f"Train patients: {split_frames.train['Patient_ID'].nunique():,} | "
        f"Val patients: {split_frames.val['Patient_ID'].nunique():,} | "
        f"Test patients: {split_frames.test['Patient_ID'].nunique():,}"
    )

    scaler, train_loader, val_loader, test_loader = build_dataloaders(
        split_frames,
        batch_size=args.batch_size,
        window_size=args.window_size,
        balanced_sampling=args.balanced_sampling,
    )

    imbalance_ratio = raw_positive_ratio(split_frames.train["SepsisLabel"])
    pos_weight = derive_positive_class_weight(
        split_frames.train["SepsisLabel"],
        strategy=args.class_weight_strategy,
        max_weight=args.pos_weight_cap,
    )
    print(
        f"Raw class imbalance (negative:positive) = {imbalance_ratio:.2f}:1 | "
        f"Effective pos_weight = {pos_weight:.2f} | "
        f"Loss = {args.loss} | "
        f"Balanced sampling = {args.balanced_sampling}"
    )

    model = SepsisTemporalCNN(input_dim=len(split_frames.feature_names), dropout=args.dropout)

    print(f"Training CNN on device={device}...")
    model, history, best_val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        patience=args.patience,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        threshold_objective=args.threshold_objective,
        target_recall=args.target_recall,
        threshold_beta=args.threshold_beta,
    )

    best_threshold = float(best_val_metrics.get("threshold", 0.5))
    print(
        f"Best validation PR AUC: {best_val_metrics.get('average_precision', float('nan')):.4f} | "
        f"Best validation ROC AUC: {best_val_metrics.get('roc_auc', float('nan')):.4f} | "
        f"Tuned threshold: {best_threshold:.3f}"
    )

    baseline_test_metrics = evaluate_model(model, test_loader, device, threshold=0.5)
    test_metrics = evaluate_model(model, test_loader, device, threshold=best_threshold)
    test_metrics["threshold_objective"] = args.threshold_objective
    test_metrics["target_recall"] = args.target_recall if args.threshold_objective == "precision_at_recall" else 0.0

    print_final_report("Temporal CNN", test_metrics, baseline_metrics=baseline_test_metrics)

    artifact_paths = save_training_artifacts(
        model_name="cnn",
        model=model,
        scaler=scaler,
        feature_names=split_frames.feature_names,
        history=history,
        metrics=test_metrics,
        model_kwargs={
            "input_dim": len(split_frames.feature_names),
            "dropout": args.dropout,
        },
        output_dir=MODEL_DIR,
        extra_metadata={
            "model_family": "temporal_cnn",
            "window_size": args.window_size,
            "loss_name": args.loss,
            "class_weight_strategy": args.class_weight_strategy,
            "balanced_sampling": args.balanced_sampling,
        },
    )

    print("\nSaved artifacts:")
    for label, path in artifact_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
