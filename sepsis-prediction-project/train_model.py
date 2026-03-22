import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("Datasets/processed/sepsis_icu_cleaned.csv")

# Define features and target
# Exclude identifiers and the target
features = [col for col in df.columns if col not in ['Patient_ID', 'SepsisLabel']]
X = df[features]
y = df['SepsisLabel']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} rows...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

# Save model and feature list
joblib.dump(model, "models/sepsis_rf_model.joblib")
joblib.dump(features, "models/feature_names.joblib")
print("Model and feature names saved to models/")
