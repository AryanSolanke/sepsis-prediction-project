import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score

# Create models directory
os.makedirs("models", exist_ok=True)

# Load data
DATA_PATH = "..\\Datasets\\processed\\sepsis_icu_cleaned.csv"
df = pd.read_csv(DATA_PATH)

# Define features and target
features = [col for col in df.columns if col not in ['Patient_ID', 'SepsisLabel']]
X = df[features]
y = df['SepsisLabel']
groups = df['Patient_ID']

# 1. Split by Patient_ID to prevent data leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Training on {len(X_train)} rows ({groups.iloc[train_idx].nunique()} unique patients)...")

# 2. Initialize and Train XGBoost
# scale_pos_weight handles imbalance (ratio of negative to positive cases)
ratio = (y == 0).sum() / (y == 1).sum()

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=ratio, 
    random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
)
model.fit(X_train, y_train)

# 3. Evaluation
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("\n--- XGBoost Model Performance ---")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

# 4. Save model and feature list
joblib.dump(model, "models/sepsis_xgb.joblib")
joblib.dump(features, "models/xgb_feature_names.joblib")

print("\nXGBoost Model and feature names saved to 'models/'")
