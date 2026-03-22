import pandas as pd
import numpy as np
import os
import glob

def load_and_combine_data(data_dir):
    psv_files = glob.glob(os.path.join(data_dir, "*.psv"))
    dfs = []
    for f in psv_files:
        try:
            # Explicitly setting engine to 'python' to handle potential parsing issues
            temp_df = pd.read_csv(f, sep='|', engine='python')
            temp_df['Patient_ID'] = os.path.basename(f).split('.')[0]
            dfs.append(temp_df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

def preprocess_sepsis_data(df):
    if df.empty:
        return df
        
    to_drop = ['Unit2', 'SaO2', 'BaseExcess', 'EtCO2', 'TroponinI', 'Fibrinogen', 'PTT']
    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    
    # Simple imputation for this task
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df.groupby('Patient_ID')[numeric_cols].transform(lambda x: x.ffill().bfill())
    
    # Fill remaining NaNs with column medians
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Feature engineering (from notebook: 'Measured' flags)
    core_vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
    for col in core_vitals:
        if col in df.columns:
            # Using notna() for boolean check then to int
            df[f'{col}_Measured'] = df[col].notna().astype(int)
            
    return df

if __name__ == "__main__":
    raw_dir = "Datasets/raw"
    processed_dir = "Datasets/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Loading raw data...")
    combined_df = load_and_combine_data(raw_dir)
    
    if combined_df.empty:
        print("No data loaded. Exiting.")
    else:
        print(f"Loaded {len(combined_df)} rows. Preprocessing...")
        cleaned_df = preprocess_sepsis_data(combined_df)
        
        output_path = os.path.join(processed_dir, "sepsis_icu_cleaned.csv")
        cleaned_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
