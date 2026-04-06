import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 1. Setup & Data Loading
os.makedirs("models", exist_ok=True)
DATA_PATH = "..\\Datasets\\processed\\sepsis_icu_cleaned.csv"
df = pd.read_csv(DATA_PATH)

target_col, group_col = 'SepsisLabel', 'Patient_ID'
features = [col for col in df.columns if col not in [group_col, target_col]]
X, y, groups = df[features], df[target_col], df[group_col]

# 2. Optimized Parameters (Unlocked to prevent Early Stopping at Iteration 1)
LGBM_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 63,           # Increased complexity
    'max_depth': 8,             # Defined depth to prevent runaway overfitting
    'min_child_samples': 50,    # Balanced for 1.5M rows
    'feature_fraction': 0.8,
    'lambda_l1': 0.5,           # Low regularization to force learning
    'lambda_l2': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# 3. Cross-Validation Execution
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
all_probs = np.zeros(len(X))
roc_aucs, pr_aucs = [], []

print("Starting 5-Fold Group Cross-Validation...")

for i, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Calculate balance weight to handle 1:55 imbalance
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_weight)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    
    probs = model.predict_proba(X_val)[:, 1]
    all_probs[val_idx] = probs
    
    fold_roc = roc_auc_score(y_val, probs)
    fold_pr = average_precision_score(y_val, probs)
    
    roc_aucs.append(fold_roc)
    pr_aucs.append(fold_pr)
    
    print(f"Fold {i} | ROC AUC: {fold_roc:.4f} | PR AUC: {fold_pr:.4f}")

# 4. FINAL OUTPUT BLOCK (Standardized Formatting)
y_pred = (all_probs >= 0.5).astype(int)

print("\n--- LightGBM Model Performance ---")
print(classification_report(y, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y, all_probs):.4f}")
print(f"PR AUC Score:  {average_precision_score(y, all_probs):.4f}")

print("\n--- Final Cross-Validation Summary ---")
print(f"Mean ROC AUC: {np.mean(roc_aucs):.4f} (+/- {np.std(roc_aucs):.4f})")
print(f"Mean PR AUC:  {np.mean(pr_aucs):.4f} (+/- {np.std(pr_aucs):.5f})")

# 5. Final Production Deployment
print("\nTraining final model on full dataset for deployment...")
final_scale = (y == 0).sum() / (y == 1).sum()
final_model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=final_scale)
final_model.fit(X, y)

# DUAL DUMP - Exactly 2 files
joblib.dump(final_model, "models/sepsis_lgbm.joblib")
joblib.dump(features, "models/lgbm_feature_names.joblib")

print("\nCross-validated model saved. Ready for evaluation.")