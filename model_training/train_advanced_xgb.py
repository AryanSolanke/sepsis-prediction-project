import pandas as pd
import numpy as np
import time
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import shap
import joblib
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay, matthews_corrcoef, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fix for XGBoost device mismatch warning
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

def calculate_lead_time(df_val, y_proba, threshold):
    temp_df = df_val[['Patient_ID', 'ICULOS', 'SepsisLabel']].copy()
    temp_df['Probability'] = y_proba
    temp_df['Prediction'] = (y_proba >= threshold).astype(int)
    sepsis_patients = temp_df[temp_df['SepsisLabel'] == 1]['Patient_ID'].unique()
    lead_times = []
    for pid in sepsis_patients:
        p_data = temp_df[temp_df['Patient_ID'] == pid]
        actual_onset = p_data[p_data['SepsisLabel'] == 1]['ICULOS'].min()
        preds = p_data[p_data['Prediction'] == 1]
        if not preds.empty:
            predicted_onset = preds['ICULOS'].min()
            lead_times.append(actual_onset - predicted_onset)
    return lead_times

def clip_lead_time_labels(df, hours=12):
    print(f"Clipping Sepsis labels to a {hours}h lead-time window...")
    sepsis_onsets = df[df['SepsisLabel'] == 1].groupby('Patient_ID')['ICULOS'].min().reset_index()
    sepsis_onsets.columns = ['Patient_ID', 'Onset_ICULOS']
    df = df.merge(sepsis_onsets, on='Patient_ID', how='left')
    mask_sepsis = df['Onset_ICULOS'].notnull()
    df.loc[mask_sepsis & (df['ICULOS'] < (df['Onset_ICULOS'] - hours)), 'SepsisLabel'] = 0
    df.loc[mask_sepsis & (df['ICULOS'] >= (df['Onset_ICULOS'] - hours)), 'SepsisLabel'] = 1
    return df.drop(columns=['Onset_ICULOS'])

def engineer_advanced_features(df):
    print("Engineering advanced temporal features...")
    df = df.sort_values(['Patient_ID', 'ICULOS'])
    if 'HospAdmTime' in df.columns:
        df['HospAdmTime'] = np.sign(df['HospAdmTime']) * np.log1p(np.abs(df['HospAdmTime']))
    key_features = ['HR', 'MAP', 'SBP', 'Resp', 'O2Sat', 'Temp', 'Lactate', 'Glucose']
    new_cols = {}
    print("Adding Patient-Baseline Normalization (Z-Scores)...")
    for col in key_features:
        if col not in df.columns: continue
        p_mean = df.groupby('Patient_ID')[col].transform('mean')
        p_std = df.groupby('Patient_ID')[col].transform('std')
        new_cols[f'{col}_zscore'] = (df[col] - p_mean) / (p_std + 1e-6)
    for col in key_features:
        if col not in df.columns: continue
        grp = df.groupby('Patient_ID')[col]
        new_cols[f'{col}_adv_12h_mean'] = grp.rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
        new_cols[f'{col}_adv_12h_max'] = grp.rolling(12, min_periods=1).max().reset_index(level=0, drop=True)
        new_cols[f'{col}_adv_12h_min'] = grp.rolling(12, min_periods=1).min().reset_index(level=0, drop=True)
        new_cols[f'{col}_adv_6h_slope'] = (grp.diff(periods=6) / 6.0).fillna(0)
        new_cols[f'{col}_adv_delta'] = grp.diff(periods=1).fillna(0)
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    return df

def main():
    print("Loading dataset...")
    df = pd.read_csv('../Datasets/processed/sepsis_icu_train.csv')
    df = clip_lead_time_labels(df, hours=12)
    df = engineer_advanced_features(df)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['Patient_ID']))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    target = 'SepsisLabel'
    features_to_drop = [
        'Patient_ID', target, 'ICULOS', 'HospAdmTime',
        'Platelets_Measured', 'SIRS_Resp', 'hypoxia_flag', 'Hgb_Measured', 
        'Creatinine_Measured', 'Phosphate_Measured', 'Lactate_adv_delta', 
        'Chloride_Measured', 'shock_flag', 'BUN_Measured', 'qSOFA_SBP', 
        'Magnesium_Measured', 'HCO3_Measured', 'qSOFA_Resp', 'WBC_Measured', 
        'Fibrinogen_Measured', 'Glucose_adv_delta', 'MAP_adv_delta', 
        'PTT_Measured', 'Bilirubin_direct_Measured'
    ]
    features = [col for col in df.columns if col not in features_to_drop]
    
    X_train = train_df[features].fillna(0).astype(np.float32)
    y_train = train_df[target]
    X_val = val_df[features].fillna(0).astype(np.float32)
    y_val = val_df[target]

    print(f"Training Final Model: {len(features)} features...")
    sieve_params = {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.05, 'max_delta_step': 2}
    verifier_params = {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'max_delta_step': 5, 'gamma': 0.1}

    print("--- STAGE 1: Sieve ---")
    sieve_model = XGBClassifier(**sieve_params, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), random_state=42, tree_method='hist', device='cuda')
    sieve_model.fit(X_train, y_train)

    print("--- STAGE 2: Verifier ---")
    sieve_proba_train = sieve_model.predict_proba(X_train)[:, 1]
    mask = sieve_proba_train >= 0.05
    X_train_2, y_train_2 = X_train[mask], y_train[mask]
    verifier_model = XGBClassifier(**verifier_params, scale_pos_weight=(y_train_2 == 0).sum() / (y_train_2 == 1).sum(), random_state=42, tree_method='hist', device='cuda')
    verifier_model.fit(X_train_2, y_train_2)

    print("Finalizing results...")
    s_val, v_val = sieve_model.predict_proba(X_val)[:, 1], verifier_model.predict_proba(X_val)[:, 1]
    y_proba = s_val * v_val

    # Find optimal threshold (Fast loop)
    best_f1, best_t = 0, 0.1
    for t in np.linspace(0.01, 0.6, 30):
        y_pred_t = (y_proba >= t).astype(int)
        f1 = classification_report(y_val, y_pred_t, output_dict=True, zero_division=0)['1']['f1-score']
        if f1 > best_f1: best_f1, best_t = f1, t

    y_pred_final = (y_proba >= best_t).astype(int)
    report = classification_report(y_val, y_pred_final)
    auc = roc_auc_score(y_val, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_final).ravel()
    brier = brier_score_loss(y_val, y_proba)
    lead_times = calculate_lead_time(val_df, y_proba, best_t)
    
    print(f"\nOptimal Threshold: {best_t:.3f} | F1: {best_f1:.4f}\n" + report)
    print(f"ROC AUC: {auc:.4f} | Lead Time: {np.mean(lead_times):.2f}h")

    eval_dir = "Performance_eval/XGBoost_Final"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 1. Plots
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); RocCurveDisplay.from_predictions(y_val, y_proba, ax=plt.gca()); plt.title("ROC")
    plt.subplot(1, 3, 2); PrecisionRecallDisplay.from_predictions(y_val, y_proba, ax=plt.gca()); plt.title("PR")
    plt.subplot(1, 3, 3); prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10); plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1],'--'); plt.title(f"Calib (Brier: {brier:.4f})")
    plt.savefig(f"{eval_dir}/performance_summary.png"); plt.close()

    plt.figure(figsize=(8, 6)); ConfusionMatrixDisplay.from_predictions(y_val, y_pred_final, cmap='Blues', ax=plt.gca()); plt.savefig(f"{eval_dir}/confusion_matrix.png"); plt.close()

    # 2. Importance
    imp = verifier_model.feature_importances_; idx = np.argsort(imp)[::-1]
    impact_df = pd.DataFrame({'Feature': features, 'SHAP_Importance': imp}).sort_values(by='SHAP_Importance', ascending=False)
    impact_df.to_csv(f"{eval_dir}/comprehensive_feature_importance.csv", index=False)
    
    # 3. Save Models
    os.makedirs('../models', exist_ok=True)
    joblib.dump(sieve_model, '../models/sieve_model.pkl')
    joblib.dump(verifier_model, '../models/verifier_model.pkl')
    joblib.dump(features, '../models/features.pkl')
    joblib.dump(best_t, '../models/threshold.pkl')

    # 4. SHAP (The only part that takes time)
    print("Generating SHAP detailed pages (Final Step)...")
    shap_v = shap.TreeExplainer(verifier_model).shap_values(X_val.values)
    shap_pos = shap_v[:, :, 1] if len(shap_v.shape) == 3 else shap_v
    shap_dir = f"{eval_dir}/SHAP_Detailed"; os.makedirs(shap_dir, exist_ok=True)
    for start in range(0, len(features), 40):
        end = min(start + 40, len(features)); p = (start // 40) + 1; curr = idx[start:end]
        plt.figure(figsize=(14, 12)); shap.summary_plot(shap_pos[:, curr], X_val.iloc[:, curr].values, feature_names=[features[i] for i in curr], plot_type="dot", max_display=40, show=False)
        plt.savefig(f"{shap_dir}/shap_beeswarm_p{p}.png"); plt.close()

    print("Training finished successfully. Models saved to ../models/")

if __name__ == "__main__": main()
