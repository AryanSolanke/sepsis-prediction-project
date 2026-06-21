import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, matthews_corrcoef, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_lead_time(df_val, y_proba, threshold):
    """
    Calculates the clinical lead time: hours predicted before actual sepsis onset.
    """
    temp_df = df_val[['Patient_ID', 'ICULOS', 'SepsisLabel']].copy()
    temp_df['Probability'] = y_proba
    temp_df['Prediction'] = (y_proba >= threshold).astype(int)
    
    # Only look at patients who actually developed sepsis
    sepsis_patients = temp_df[temp_df['SepsisLabel'] == 1]['Patient_ID'].unique()
    lead_times = []
    
    for pid in sepsis_patients:
        p_data = temp_df[temp_df['Patient_ID'] == pid]
        
        # Actual onset time
        actual_onset = p_data[p_data['SepsisLabel'] == 1]['ICULOS'].min()
        
        # Predicted onset time
        preds = p_data[p_data['Prediction'] == 1]
        if not preds.empty:
            predicted_onset = preds['ICULOS'].min()
            # Positive lead time = predicted before actual onset
            lead_times.append(actual_onset - predicted_onset)
            
    return lead_times

def engineer_advanced_features(df):
    print("Engineering advanced temporal features...")
    df = df.sort_values(['Patient_ID', 'ICULOS'])
    
    # Apply Symmetric Log Transformation to HospAdmTime
    if 'HospAdmTime' in df.columns:
        df['HospAdmTime'] = np.sign(df['HospAdmTime']) * np.log1p(np.abs(df['HospAdmTime']))
    
    # We focus on key vitals and labs that change frequently
    key_features = ['HR', 'MAP', 'SBP', 'Resp', 'O2Sat', 'Temp', 'Lactate', 'Glucose']
    
    new_cols = {}
    for col in key_features:
        if col not in df.columns:
            continue
        
        # Optimized rolling operations using built-in methods instead of slow lambda transforms
        grp = df.groupby('Patient_ID')[col]
        
        new_cols[f'{col}_adv_12h_mean'] = grp.rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
        new_cols[f'{col}_adv_12h_max'] = grp.rolling(12, min_periods=1).max().reset_index(level=0, drop=True)
        new_cols[f'{col}_adv_12h_min'] = grp.rolling(12, min_periods=1).min().reset_index(level=0, drop=True)
        
        # Trend slope (difference over 6 hours divided by 6)
        new_cols[f'{col}_adv_6h_slope'] = (grp.diff(periods=6) / 6.0).fillna(0)
        
        # Delta changes
        new_cols[f'{col}_adv_delta'] = grp.diff(periods=1).fillna(0)
        
    # Concatenate all new columns at once to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    return df

def main():
    print("Loading dataset...")
    df = pd.read_csv('../Datasets/processed/sepsis_icu_train.csv')

    df = engineer_advanced_features(df)

    print("Splitting data patient-wise...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['Patient_ID']))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    target = 'SepsisLabel'
    features = [col for col in df.columns if col not in ['Patient_ID', target]]

    # Sklearn's Random Forest cannot handle NaNs. We use fillna(0) for engineered features 
    # (since base features were already imputed in preprocessing)
    X_train = train_df[features].fillna(0).astype(np.float32)
    y_train = train_df[target]
    groups_train = train_df['Patient_ID']
    
    X_val = val_df[features].fillna(0).astype(np.float32)
    y_val = val_df[target]

    print("Setting up Hyperparameter Tuning for Random Forest...")
    # Initialize Random Forest with balanced class weight
    rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=None # Each fit uses 1 core; we parallelize at the search level
    )

    # Grid optimized for Random Forest
    param_distributions = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [5, 10, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }

    # 1. StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=3)
    cv_splits = list(cv.split(X_train, y_train, groups=groups_train))

    # 2. RandomizedSearchCV
    # NOTE: Set n_jobs=-1 to parallelize different hyperparameter trials
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50, 
        scoring='f1',
        cv=cv_splits,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("Starting Advanced Model Training (with Tuning)...")
    start_time = time.time()
    
    search.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    best_rf = search.best_estimator_
    print(f"Best Parameters: {search.best_params_}")

    print("Evaluating Best Model with Threshold Optimization...")
    y_proba = best_rf.predict_proba(X_val)[:, 1]

    # Find the threshold that maximizes F1-score
    print("\nSearching for Optimal Threshold to maximize F1...")
    thresholds = np.linspace(0.05, 0.8, 16)
    best_f1 = 0
    best_t = 0.5

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        report_t = classification_report(y_val, y_pred_t, output_dict=True, zero_division=0)
        f1 = report_t['1']['f1-score']
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"\nOptimal Threshold Found: {best_t:.2f} (F1: {best_f1:.4f})")
    y_pred_final = (y_proba >= best_t).astype(int)
    
    # Calculate Core Metrics
    report = classification_report(y_val, y_pred_final)
    auc = roc_auc_score(y_val, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_final).ravel()
    
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0 
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    mcc = matthews_corrcoef(y_val, y_pred_final)
    brier = brier_score_loss(y_val, y_proba)
    
    # Likelihood Ratios
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    plr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else np.nan
    nlr = (1 - sensitivity) / specificity if specificity > 0 else np.nan

    # Clinical Analysis: Lead Time
    lead_times = calculate_lead_time(val_df, y_proba, best_t)
    avg_lead_time = np.mean(lead_times) if lead_times else 0
    
    print("\nClassification Report (Optimized Threshold):")
    print(report)
    print(f"ROC AUC Score: {auc:.4f}")

    print("\n--- Advanced Performance Metrics ---")
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"NPV: {npv:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Positive Likelihood Ratio (PLR): {plr:.4f}")
    print(f"Negative Likelihood Ratio (NLR): {nlr:.4f}")
    print(f"Average Lead Time: {avg_lead_time:.2f} hours")

    # Generate and Save Performance Graphs
    # ... rest of plotting code ...
    print("Generating performance graphs...")
    eval_dir = f"Performance_eval/RandomForest"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_val, y_proba, ax=ax)
    plt.title(f"RandomForest - ROC Curve")
    plt.savefig(f"{eval_dir}/roc_curve.png")
    plt.close()

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_val, y_proba, ax=ax)
    plt.title(f"RandomForest - Precision-Recall Curve")
    plt.savefig(f"{eval_dir}/pr_curve.png")
    plt.close()

    # 3. Calibration Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_predictions(y_val, y_proba, n_bins=10, ax=ax)
    plt.title(f"RandomForest - Calibration Plot")
    plt.savefig(f"{eval_dir}/calibration_plot.png")
    plt.close()

    # 4. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_final, cmap='Blues', ax=ax)
    plt.title(f"RandomForest - Confusion Matrix (Threshold: {best_t:.2f})")
    plt.savefig(f"{eval_dir}/confusion_matrix.png")
    plt.close()

    # 5. Lead Time Distribution
    if lead_times:
        plt.figure(figsize=(8, 6))
        sns.histplot(lead_times, bins=20, kde=True, color='salmon')
        plt.axvline(avg_lead_time, color='red', linestyle='--', label=f'Avg: {avg_lead_time:.1f}h')
        plt.title("RandomForest Lead Time Distribution")
        plt.xlabel("Hours before Actual Onset")
        plt.ylabel("Patient Count")
        plt.legend()
        plt.savefig(f"{eval_dir}/lead_time_distribution.png")
        plt.close()

    # 6. Feature Importance (Top 20)
    if hasattr(best_rf, 'feature_importances_'):
        importances = best_rf.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        top_features = [features[i] for i in indices]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances[indices], y=top_features, hue=top_features, palette='viridis', legend=False)
        plt.title(f"RandomForest - Top 20 Feature Importances")
        plt.tight_layout()
        plt.savefig(f"{eval_dir}/feature_importance.png")
        plt.close()


    # Store comprehensive results
    with open('model_reports.txt', 'a') as f:
        f.write("\n=== ADVANCED Random Forest (Comprehensive Evaluation) ===\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Best Params: {search.best_params_}\n")
        f.write(f"Optimal Threshold: {best_t:.2f}\n")
        f.write(f"PPV: {ppv:.4f}, NPV: {npv:.4f}\n")
        f.write(f"MCC: {mcc:.4f}, Brier Score: {brier:.4f}\n")
        f.write(f"PLR: {plr:.4f}, NLR: {nlr:.4f}\n")
        f.write(f"Average Lead Time: {avg_lead_time:.2f} hours\n")
        f.write(f"ROC AUC Score: {auc:.4f}\n")
        f.write(report + "\n")

if __name__ == "__main__":
    main()
