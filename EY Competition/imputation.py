import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import joblib


def diagnose_and_impute(df, fit_imputer=True, imputer_path="imputer_state.joblib"):
    print("\n-- Diagnosing Missing Data --")

    df = df.copy()

    # Ensure temporal features exist for the imputer
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

    # Drop fully-NaN columns
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
        print("-- Fitting Iterative Imputer (BayesianRidge) --")
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=500,
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

        # Save state
        joblib.dump({
            'imputer':    imputer,
            'valid_cols': valid_cols,
            'medians':    train_medians,
        }, imputer_path)
        print(f"   Saved imputer state to '{imputer_path}'")

    else:
        print("-- Loading saved imputer --")
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

    print("-- Final NaN cleanup --")
    if fit_imputer:
        medians = train_medians
    else:
        medians = state['medians']

    # Drop columns that are entirely or almost entirely NaN (>95%)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        frac_missing = df[c].isna().sum() / len(df)
        if frac_missing > 0.95:
            print(f"   Dropping '{c}' ({frac_missing*100:.0f}% NaN — too sparse)")
            df = df.drop(columns=[c], errors='ignore')

    # For remaining NaNs, use location-group median first (same lat/lon), then fall back to global median
    loc_key = None
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        loc_key = (df['Latitude'].round(2).astype(str) + '_' +
                   df['Longitude'].round(2).astype(str))

    for c in df.select_dtypes(include=[np.number]).columns:
        n = df[c].isna().sum()
        if n > 0:
            # Try location-group median first
            filled_by_loc = 0
            if loc_key is not None and n < len(df): 
                loc_medians = df.groupby(loc_key)[c].transform('median')
                was_na = df[c].isna()
                df[c] = df[c].fillna(loc_medians)
                filled_by_loc = was_na.sum() - df[c].isna().sum()

            # Fall back to global median for any remaining
            n_remaining = df[c].isna().sum()
            if n_remaining > 0:
                fill_val = medians.get(c, df[c].median())
                if pd.isna(fill_val):
                    fill_val = 0.0
                df[c] = df[c].fillna(fill_val)
                if filled_by_loc > 0:
                    print(f"   Filling {n} NaNs in '{c}': "
                          f"{filled_by_loc} by location, "
                          f"{n_remaining} by global median ({fill_val:.4f})")
                else:
                    print(f"   Filling {n} NaNs in '{c}' with {fill_val:.4f}")
            elif filled_by_loc > 0:
                print(f"   Filling {n} NaNs in '{c}': "
                      f"all {filled_by_loc} by location median")

    return df
