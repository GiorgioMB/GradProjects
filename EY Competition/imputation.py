import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

def diagnose_and_impute(df):
    print("\n--- Diagnosing Missing Data ---")
    
    # 1. Feature Engineering for Imputer
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)
    df['Month'] = df['Sample Date'].dt.month
    df['Year'] = df['Sample Date'].dt.year
    
    sat_cols = ['nir08', 'green', 'red', 'swir16', 'swir22', 'NDMI', 'MNDWI', 'NDVI']
    context_cols = ['Latitude', 'Longitude', 'Month', 'elevation_mean']
    
    # 2. Diagnosis Test
    df['_is_missing'] = df['nir08'].isna().astype(int)
    
    if df['_is_missing'].sum() > 0:
        X_check = df[context_cols].fillna(0)
        y_check = df['_is_missing']
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_check, y_check)
        auc = roc_auc_score(y_check, rf.predict_proba(X_check)[:, 1])
        
        print(f"   Missingness AUC: {auc:.3f}")
        if auc > 0.65:
            print("   Diagnosis: Data is Missing At Random (Systematic). Imputation required.")
        else:
            print("   Diagnosis: Data is Missing Completely at Random.")
    else:
        print("   No missing data found.")
        # Still run final NaN cleanup (other columns may have NaNs)
        print("--- Final NaN cleanup ---")
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                n_nas = df[col].isna().sum()
                print(f"   Filling {n_nas} remaining NaNs in '{col}' with median")
                df[col] = df[col].fillna(df[col].median())
        df = df.drop(columns=['_is_missing'])
        return df

    # 3. Imputation (MICE)
    print("--- Running Iterative Imputer ---")
    
    impute_cols = context_cols + sat_cols
    valid_cols = [c for c in impute_cols if c in df.columns]
    
    fully_nan_cols = []
    for col in valid_cols:
        if df[col].isna().all():
            fully_nan_cols.append(col)
    
    if fully_nan_cols:
        print(f"    Found {len(fully_nan_cols)} columns with ALL NaN values:")
        for col in fully_nan_cols:
            print(f"      - '{col}' (100% missing, dropping)")
        valid_cols = [c for c in valid_cols if c not in fully_nan_cols]
        df = df.drop(columns=fully_nan_cols)
    
    if not valid_cols:
        print("   No valid columns to impute!")
        return df
    
    data_to_impute = df[valid_cols].copy()
    
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, max_depth=8, n_jobs=-1),
        max_iter=5,
        random_state=42,
        verbose=1
    )
    
    imputed_data = imputer.fit_transform(data_to_impute)
    imputed_df = pd.DataFrame(imputed_data, columns=valid_cols, index=df.index)
    
    # 4. Patch back into main dataframe 
    for col in valid_cols:
        if col in imputed_df.columns:
            df[col] = imputed_df[col]
    
    print("--- Final NaN cleanup ---")
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            n_nas = df[col].isna().sum()
            print(f"   Filling {n_nas} remaining NaNs in '{col}' with median")
            df[col] = df[col].fillna(df[col].median())
            
    df = df.drop(columns=['_is_missing'])
    return df
