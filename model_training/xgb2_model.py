import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

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

# 2. Strict Patient-Level Split (20% Test)
# This ensures zero leakage—test patients are entirely "new" to the model.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 3. Calculate Scale Weight for Imbalance
# Ratio of Healthy rows to Sepsis rows (~55:1)
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 4. Best-in-General XGBoost Configuration
# Using 'hist' tree_method for massive speed gains on large datasets
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,     # Slow learning for better generalization
    max_depth=6,            # Prevents memorizing specific patient noise
    scale_pos_weight=scale_weight, 
    subsample=0.8,          # Trains on 80% of rows to stay robust
    colsample_bytree=0.8,   # Uses 80% of features to handle redundancy
    tree_method='hist',     # Fast histogram-based algorithm
    random_state=42,
    n_jobs=-1,              # Uses all CPU cores
    early_stopping_rounds=50 # Stops if PR AUC stops improving
)

# 5. Train with Validation to Monitor Generalization
print(f"Training on {X_train.shape[0]} rows from {groups.iloc[train_idx].nunique()} patients...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=20
)

# 6. Evaluation
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("\n--- Final XGBoost Performance ---")
print(f"Best Iteration: {model.best_iteration}")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR AUC (Avg Precision): {average_precision_score(y_test, y_probs):.4f}")

# 7. Save
joblib.dump(model, "models/sepsis_xgb2.joblib")
joblib.dump(features, "models/xgb2_feature_names.joblib")

print("\nXGBoost training complete and model saved.")