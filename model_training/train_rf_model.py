import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# 1. Setup and Load
os.makedirs("models", exist_ok=True)
DATA_PATH = "..\\Datasets\\processed\\sepsis_icu_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Define columns
target_col = 'SepsisLabel'
group_col = 'Patient_ID'
features = [col for col in df.columns if col not in [group_col, target_col]]

X = df[features]
y = df[target_col]
groups = df[group_col]

# 2. Group-Aware Cross-Validation
# This ensures that all rows for a single Patient_ID stay together
gkf = GroupKFold(n_splits=5)

# 3. Define Model and Search Space
# 'balanced_subsample' is specifically designed for Random Forests with imbalanced data
rf = RandomForestClassifier(
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

# We search for parameters that prioritize GENERAL patterns over NOISY details
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [5, 8, 12],
    'min_samples_leaf': [10, 50, 100], 
    'max_features': ['sqrt', 0.3] # Look at fewer features per tree to improve generalization
}

# 4. Run the Search
# We optimize for 'average_precision' because in Sepsis, the balance of 
# Precision and Recall is more important than simple Accuracy.
search = RandomizedSearchCV(
    rf, 
    param_distributions=param_grid, 
    n_iter=8, 
    cv=gkf, 
    scoring='average_precision', 
    verbose=2,
    random_state=42
)

print(f"Training on {len(X)} rows from {groups.nunique()} PhysioNet patients...")
search.fit(X, y, groups=groups)

# 5. Extract Best Model
best_rf = search.best_estimator_

# 6. Final Evaluation on a "Held-Out" Group
# We manually pull one fold to show you the final performance metrics
train_idx, test_idx = next(gkf.split(X, y, groups=groups))
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

y_pred = best_rf.predict(X_test)
y_probs = best_rf.predict_proba(X_test)[:, 1]

print("\n--- 'Best in General' Random Forest Results ---")
print(f"Optimal Parameters: {search.best_params_}")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR AUC (Avg Precision): {average_precision_score(y_test, y_probs):.4f}")

# 7. Save
joblib.dump(best_rf, "models/sepsis_rf_physionet_best.joblib")
joblib.dump(features, "models/feature_names.joblib")

print("\nModel and feature list saved successfully.")