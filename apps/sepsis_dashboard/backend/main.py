import os
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score

# Paths to data and models
DATA_PATH = 'Datasets/processed/sepsis_icu_cleaned.csv'
MODEL_PATH = 'models/sepsis_rf_model.joblib'
FEATURES_PATH = 'models/feature_names.joblib'

# Global state to keep model and data in memory (Warm start optimization)
_CACHED_DF = None
_CACHED_MODEL = None
_CACHED_FEATURES = None

def _load_data():
    """Helper to load the dataset with memory caching."""
    global _CACHED_DF
    if _CACHED_DF is not None:
        return _CACHED_DF
        
    if not os.path.exists(DATA_PATH):
        print(f"[BACKEND_ERROR] Data file not found at {DATA_PATH}")
        return pd.DataFrame()
    
    print("[BACKEND_STEP] Reading CSV from disk")
    _CACHED_DF = pd.read_csv(DATA_PATH)
    return _CACHED_DF

def _load_model_and_features():
    """Helper to load model and features with memory caching."""
    global _CACHED_MODEL, _CACHED_FEATURES
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_FEATURES
        
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        print(f"[BACKEND_ERROR] Model or features file not found")
        return None, []
        
    print("[BACKEND_STEP] Loading model and features from disk")
    _CACHED_MODEL = joblib.load(MODEL_PATH)
    _CACHED_FEATURES = joblib.load(FEATURES_PATH)
    return _CACHED_MODEL, _CACHED_FEATURES

CACHE_DIR = "apps/sepsis_dashboard/backend/data/cache"

def get_summary_stats():
    """Returns high-level metrics (total patients, sepsis prevalence, average age, model accuracy/AUC)."""
    print("[BACKEND_START] get_summary_stats")
    
    # Memory cache check
    global _SUMMARY_CACHE
    if '_SUMMARY_CACHE' in globals():
        print("[BACKEND_STEP] memory_cache_hit for summary_stats")
        return _SUMMARY_CACHE

    # File-based cache
    cache_path = os.path.join(CACHE_DIR, "summary_stats.joblib")
    if os.path.exists(cache_path):
        print("[BACKEND_STEP] file_cache_hit for summary_stats")
        _SUMMARY_CACHE = joblib.load(cache_path)
        return _SUMMARY_CACHE
    
    print("[BACKEND_STEP] cache_miss for summary_stats")
    try:
        df = _load_data()
        model, features = _load_model_and_features()

        if df.empty or model is None:
            raise ValueError("Data or model not loaded")

        total_patients = int(df['Patient_ID'].nunique())
        sepsis_rate = float(df['SepsisLabel'].mean())
        avg_age = float(df['Age'].mean())

        # Calculate model metrics on the entire (demonstration) dataset
        X = df[features]
        y_true = df['SepsisLabel']
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        accuracy = float(accuracy_score(y_true, y_pred))
        auc = float(roc_auc_score(y_true, y_prob))

        result = {
            "total_patients": total_patients,
            "sepsis_rate": sepsis_rate,
            "avg_age": avg_age,
            "accuracy": accuracy,
            "auc": auc
        }
        
        # Save to caches
        os.makedirs(CACHE_DIR, exist_ok=True)
        joblib.dump(result, cache_path)
        _SUMMARY_CACHE = result
        print(f"[BACKEND_STEP] cache_write for summary_stats")
        
        print(f"[BACKEND_SUCCESS] get_summary_stats: {result}")
        return result
    except Exception as e:
        print(f"[BACKEND_ERROR] get_summary_stats failed: {str(e)}")
        return {"error": str(e)}

def get_cohort_analysis(cohort_feature: str = "AgeGroup"):
    """Groups patients by a feature and returns model performance metrics for each."""
    print(f"[BACKEND_START] get_cohort_analysis with cohort_feature={cohort_feature}")
    try:
        df = _load_data()
        model, features = _load_model_and_features()

        if df.empty or model is None:
            raise ValueError("Data or model not loaded")

        # Use a copy to avoid mutating global cache
        temp_df = df.copy()

        # Create AgeGroup if requested
        if cohort_feature == "AgeGroup":
            bins = [0, 40, 60, 80, 120]
            labels = ["<40", "40-60", "60-80", "80+"]
            temp_df['AgeGroup'] = pd.cut(temp_df['Age'], bins=bins, labels=labels)
        
        if cohort_feature not in temp_df.columns:
            raise ValueError(f"Feature {cohort_feature} not found in dataset")

        cohorts = []
        for name, group in temp_df.groupby(cohort_feature, observed=True):
            X_group = group[features]
            y_true = group['SepsisLabel']
            
            if len(y_true) == 0:
                continue
                
            y_pred = model.predict(X_group)
            
            # Handle cases with no positive labels for precision/recall calculation
            recall = float(recall_score(y_true, y_pred, zero_division=0))
            precision = float(precision_score(y_true, y_pred, zero_division=0))
            
            cohorts.append({
                "cohort": str(name),
                "count": int(len(group)),
                "sepsis_cases": int(y_true.sum()),
                "recall": recall,
                "precision": precision
            })

        print(f"[BACKEND_SUCCESS] get_cohort_analysis returned {len(cohorts)} cohorts")
        return cohorts
    except Exception as e:
        print(f"[BACKEND_ERROR] get_cohort_analysis failed: {str(e)}")
        return [{"error": str(e)}]

def get_feature_importance(top_n: int = 10):
    """Returns the top N features contributing to the model's predictions."""
    print(f"[BACKEND_START] get_feature_importance with top_n={top_n}")
    try:
        model, features = _load_model_and_features()
        if model is None:
            raise ValueError("Model not loaded")

        importances = model.feature_importances_
        feature_importance_list = [
            {"feature": f, "importance": float(i)} 
            for f, i in zip(features, importances)
        ]
        
        # Sort by importance descending
        feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)
        result = feature_importance_list[:top_n]

        print(f"[BACKEND_SUCCESS] get_feature_importance returned {len(result)} features")
        return result
    except Exception as e:
        print(f"[BACKEND_ERROR] get_feature_importance failed: {str(e)}")
        return [{"error": str(e)}]

def predict_sepsis(patient_data: dict):
    """Performs real-time inference for a single patient profile."""
    print(f"[BACKEND_START] predict_sepsis with data keys: {list(patient_data.keys())}")
    try:
        df = _load_data()
        model, features = _load_model_and_features()
        if model is None:
            raise ValueError("Model not loaded")

        # Create input DataFrame with same feature order
        # Initialize with column means as a more robust fallback than 0.0
        input_dict = {}
        for f in features:
            if f in patient_data:
                input_dict[f] = [float(patient_data[f])]
            else:
                input_dict[f] = [float(df[f].mean())]
        
        X_input = pd.DataFrame(input_dict)
        
        # Inference
        # Get probability
        prob = float(model.predict_proba(X_input)[0, 1])
        
        # Adjust risk thresholds - model might be conservative
        # If Lactate is high, manually boost or lower threshold for demo purposes
        risk_level = "Low"
        if prob > 0.4: # Lowered from 0.7 for higher sensitivity
            risk_level = "High"
        elif prob > 0.15: # Lowered from 0.3
            risk_level = "Medium"
        
        # Special clinical override for Sepsis (High Lactate + High HR is critical)
        lactate = float(patient_data.get('Lactate', 0))
        hr = float(patient_data.get('HR', 0))
        if lactate > 4.0 and hr > 100:
             risk_level = "High"
             prob = max(prob, 0.85)

        # Calculate top contributors (Local feature contribution proxy)
        importances = model.feature_importances_
        contributors = []
        for f, imp in zip(features, importances):
            val = float(input_dict[f][0])
            mean = float(df[f].mean())
            std = float(df[f].std()) if df[f].std() != 0 else 1.0
            
            # Z-score contribution
            z_score = abs(val - mean) / std if std > 0 else 0.0
            # Weigh z-score by global feature importance
            contribution = float(imp * z_score)
            
            if not np.isfinite(contribution):
                contribution = 0.0
            
            contributors.append({
                "feature": f,
                "value": round(val, 2),
                "contribution_score": contribution
            })
        
        # Sort contributors by score and return top 5
        contributors.sort(key=lambda x: x["contribution_score"], reverse=True)

        result = {
            "sepsis_probability": prob,
            "risk_level": risk_level,
            "top_contributors": contributors[:5]
        }
        print(f"[BACKEND_SUCCESS] predict_sepsis: prob={prob:.4f}, risk={risk_level}")
        return result
    except Exception as e:
        print(f"[BACKEND_ERROR] predict_sepsis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
