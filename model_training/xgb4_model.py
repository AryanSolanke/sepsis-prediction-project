import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

# 1. Setup
os.makedirs("models", exist_ok=True)
DATA_PATH = "..\\Datasets\\processed\\sepsis_icu_cleaned.csv"
df = pd.read_csv(DATA_PATH)

target_col = 'SepsisLabel'
group_col = 'Patient_ID'
features = [col for col in df.columns if col not in [group_col, target_col]]

X = df[features]
y = df[target_col]
groups = df[group_col]

# 2. Initialize GroupKFold
gkf = GroupKFold(n_splits=5)
fold_roc_aucs = []
fold_pr_aucs = []

print(f"Starting 5-Fold Group Cross-Validation...")

# 3. Cross-Validation Loop
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Dynamic Scale Weight for this specific fold
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_weight,
        tree_method='hist',
        eval_metric='aucpr',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    pr = average_precision_score(y_test, probs)
    
    fold_roc_aucs.append(roc)
    fold_pr_aucs.append(pr)
    
    print(f"Fold {fold+1} | ROC AUC: {roc:.4f} | PR AUC: {pr:.4f}")

# 4. Final Aggregated Results
print("\n--- Final Cross-Validation Summary ---")
print(f"Mean ROC AUC: {np.mean(fold_roc_aucs):.4f} (+/- {np.std(fold_roc_aucs):.4f})")
print(f"Mean PR AUC:  {np.mean(fold_pr_aucs):.4f} (+/- {np.std(fold_pr_aucs):.4f})")

# 5. Final Train on Full Dataset & Save
print("\nTraining final model on full dataset for deployment...")
final_scale_weight = (y == 0).sum() / (y == 1).sum()
final_model = XGBClassifier(
    n_estimators=model.best_iteration, # Use best iteration from last fold
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=final_scale_weight,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
final_model.fit(X, y)

# 6. DUAL DUMP
joblib.dump(final_model, "models/sepsis_xgb4.joblib")
joblib.dump(features, "models/xgb4_feature_names.joblib")

print("\nCross-validated model saved. Ready for evaluation.")