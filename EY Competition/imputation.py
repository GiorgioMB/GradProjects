import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import joblib


def diagnose_and_impute(df, fit_imputer=True, imputer_path="imputer_state.joblib"):
    print("\n Diagnosing Missing Data")

    df = df.copy()
    df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)
    if 'Month' not in df.columns:
        df['Month'] = df['Sample Date'].dt.month
    if 'Year' not in df.columns:
        df['Year'] = df['Sample Date'].dt.year

    sat_cols = ['nir08', 'green', 'red', 'swir16', 'swir22',
                'NDMI', 'MNDWI', 'NDVI']
    context_cols = ['Latitude', 'Longitude', 'Month']
    if 'elevation_mean' in df.columns:
        context_cols.append('elevation_mean')

    impute_cols = context_cols + sat_cols
    valid_cols  = [c for c in impute_cols if c in df.columns]

    fully_nan = [c for c in valid_cols if df[c].isna().all()]
    if fully_nan:
        print(f"   Dropping {len(fully_nan)} fully-NaN columns: {fully_nan}")
        valid_cols = [c for c in valid_cols if c not in fully_nan]
        df = df.drop(columns=fully_nan, errors='ignore')

    # Report missingness
    n_missing = df[valid_cols].isna().sum()
    for c in valid_cols:
        if n_missing[c] > 0:
            pct = 100 * n_missing[c] / len(df)
            print(f"   {c}: {n_missing[c]} missing ({pct:.1f}%)")

    if fit_imputer:
        print("── Fitting Iterative Imputer (BayesianRidge) ──")
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=42,
            verbose=1,
        )

        data = df[valid_cols].copy()
        imputed = imputer.fit_transform(data)
        imputed_df = pd.DataFrame(imputed, columns=valid_cols, index=df.index)

        for c in valid_cols:
            df[c] = imputed_df[c]

        train_medians = {}
        for c in df.select_dtypes(include=[np.number]).columns:
            train_medians[c] = df[c].median()

        joblib.dump({
            'imputer':    imputer,
            'valid_cols': valid_cols,
            'medians':    train_medians,
        }, imputer_path)
        print(f"   Saved imputer state to '{imputer_path}'")

    else:
        print("── Loading saved imputer ──")
        state = joblib.load(imputer_path)
        imputer       = state['imputer']
        saved_cols    = state['valid_cols']
        train_medians = state['medians']

        use_cols = [c for c in saved_cols if c in df.columns]
        data = df[use_cols].copy()

        for c in saved_cols:
            if c not in data.columns:
                data[c] = np.nan
        data = data[saved_cols]

        imputed = imputer.transform(data)
        imputed_df = pd.DataFrame(imputed, columns=saved_cols, index=df.index)
        for c in saved_cols:
            if c in df.columns:
                df[c] = imputed_df[c]

    print("── Final NaN cleanup ──")
    if fit_imputer:
        medians = train_medians
    else:
        medians = state['medians']

    for c in df.select_dtypes(include=[np.number]).columns:
        n = df[c].isna().sum()
        if n > 0:
            fill_val = medians.get(c, df[c].median())
            if pd.isna(fill_val):
                fill_val = 0.0
            print(f"   Filling {n} NaNs in '{c}' with {fill_val:.4f}")
            df[c] = df[c].fillna(fill_val)

    return df
