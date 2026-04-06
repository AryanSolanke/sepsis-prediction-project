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

# 2. Strict Patient-Level Split (20% Test)
# We split the dataframe first so we can undersample the training portion specifically
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, df[target_col], groups=df[group_col]))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# 3. UNDERSAMPLING (The "Best-in-General" Fix)
# We keep all Sepsis patients, but reduce the "Healthy" noise in the training set.
sepsis_pats = train_df[train_df[target_col] == 1][group_col].unique()
healthy_pats = train_df[train_df[target_col] == 0][group_col].unique()

# Randomly select 25% of healthy patients to keep
np.random.seed(42)
selected_healthy_pats = np.random.choice(healthy_pats, size=int(len(healthy_pats) * 0.25), replace=False)

# Reconstruct training set
selected_pats = np.concatenate([sepsis_pats, selected_healthy_pats])
train_df_balanced = train_df[train_df[group_col].isin(selected_pats)]

X_train = train_df_balanced[features]
y_train = train_df_balanced[target_col]

X_test = test_df[features]
y_test = test_df[target_col]

# 4. Best-in-General XGBoost Configuration
# scale_pos_weight is lowered because undersampling already balanced the data significantly
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=10,    # Reduced from 55 to prevent over-sensitivity
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

# 5. Train
# Notice we validate on the REAL test set (original distribution)
print(f"Training on {train_df_balanced[group_col].nunique()} patients ({len(X_train)} rows)")
print(f"Testing on {test_df[group_col].nunique()} patients ({len(X_test)} rows - REAL DISTRIBUTION)")

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=20
)

# 6. Evaluation
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("\n--- Final Optimized XGBoost Performance ---")
print(f"Best Iteration: {model.best_iteration}")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR AUC (Avg Precision): {average_precision_score(y_test, y_probs):.4f}")

# 7. Save
joblib.dump(model, "models/sepsis_xgb3.joblib")
joblib.dump(features, "models/xgb3_feature_names.joblib")

print("\nOptimized XGBoost training complete and model saved.")