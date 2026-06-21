import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score

def main():
    print("Loading dataset...")
    # Load the processed training data
    df = pd.read_csv('../Datasets/processed/sepsis_icu_train.csv')

    print("Splitting data patient-wise...")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['Patient_ID']))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Define features and target
    target = 'SepsisLabel'
    
    features = [col for col in df.columns if col not in ['Patient_ID', target]]

    X_train = train_df[features]
    y_train = train_df[target]
    
    X_val = val_df[features]
    y_val = val_df[target]

    print("Training Balanced Random Forest...")

    rf = BalancedRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:, 1]

    rf_report = classification_report(y_val, y_pred)
    rf_auc = roc_auc_score(y_val, y_proba)

    print("\nClassification Report:")
    print(rf_report)
    print(f"ROC AUC Score: {rf_auc:.4f}")

    with open('model_reports.txt', 'w') as f:
        f.write("=== Balanced Random Forest ===\n")
        f.write(rf_report + "\n")
        f.write(f"ROC AUC Score: {rf_auc:.4f}\n\n")

## TRAINING SIMPLE XGBOOST CLASSIFIER
    print("Training simple XG Boost...")

    class_counts = y_train.value_counts()
    num_neg = class_counts[0]
    num_pos = class_counts[1]
    estimated_weight = num_neg / num_pos

    xgb = XGBClassifier(
        scale_pos_weight=estimated_weight,
        eval_metric='aucpr',
        random_state=42,
        n_estimators=300,
        device='cuda')
    
    xgb.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred_xgb = xgb.predict(X_val)
    y_probs_xgb = xgb.predict_proba(X_val)[:, 1]

    xgb_report = classification_report(y_val, y_pred_xgb)
    xgb_auc = roc_auc_score(y_val, y_probs_xgb)

    print("\nClassification Report:")
    print(xgb_report)
    print(f"ROC AUC Score: {xgb_auc:.4f}")

    with open('model_reports.txt', 'a') as f:
        f.write("=== XGBoost Classifier ===\n")
        f.write(xgb_report + "\n")
        f.write(f"ROC AUC Score: {xgb_auc:.4f}\n\n")

if __name__ == "__main__":
    main()
