import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm

import config
import data_fetch
import imputation
import modeling
import eda_report


# HELPERS
def robust_merge(left, right, on_cols, how='left'):
    left  = left.copy()
    right = right.copy()
    for c in on_cols:
        if left[c].dtype in [np.float64, np.float32, float]:
            left[c]  = left[c].round(4)
            right[c] = right[c].round(4)
        else:
            left[c]  = left[c].astype(str).str.strip()
            right[c] = right[c].astype(str).str.strip()
    return pd.merge(left, right, on=on_cols, how=how)


def clean_satellite_csv(csv_path):
    if not os.path.exists(csv_path):
        return
    try:
        expected = ['Latitude', 'Longitude', 'Sample Date',
                    'green', 'red', 'nir08', 'swir16', 'swir22',
                    'NDMI', 'MNDWI', 'NDVI']
        df = pd.read_csv(csv_path)
        if list(df.columns) != expected:
            if set(expected).issubset(set(df.columns)):
                df[expected].to_csv(csv_path, index=False)
                print("   Repaired satellite CSV column order.")
        else:
            print("   Satellite CSV looks good.")
    except Exception as e:
        print(f"   Error checking satellite CSV: {e}")


# MAIN PIPELINE
def main():
    print("=" * 60)
    print("   WATER QUALITY PREDICTION PIPELINE")
    print("=" * 60)

    fast_mode = '--fast' in sys.argv
    skip_fetch = '--skip-fetch' in sys.argv
    enriched_path = "water_quality_post_enrichment.csv"

    if skip_fetch and os.path.exists(enriched_path):
        print("\n   --skip-fetch: Loading pre-enriched data…")
        full_df = pd.read_csv(enriched_path)
        print(f"   Loaded {len(full_df)} rows, {len(full_df.columns)} cols")

    else:
        print("\n1. Loading base data")
        if not os.path.exists(config.TRAIN_FILE):
            print(f"   Error: {config.TRAIN_FILE} not found.")
            return
        train_df = pd.read_csv(config.TRAIN_FILE)
        print(f"   Loaded {len(train_df)} rows.")

        print("\n2. Satellite data")
        if os.path.exists(config.SAT_CACHE):
            clean_satellite_csv(config.SAT_CACHE)
            sat_df = pd.read_csv(config.SAT_CACHE)
            print(f"   Satellite cache: {len(sat_df)} rows")

            merge_cols = ['Latitude', 'Longitude', 'Sample Date']
            full_df = robust_merge(train_df, sat_df, merge_cols, how='left')

            sat_cols = ['green', 'red', 'nir08', 'swir16', 'swir22',
                        'NDMI', 'MNDWI', 'NDVI']
            n_with_sat = full_df[sat_cols[0]].notna().sum()
            print(f"   Merged: {n_with_sat}/{len(full_df)} rows "
                  f"have satellite data ({100*n_with_sat/len(full_df):.0f}%)")

            missing = full_df[full_df['nir08'].isna()]
            if len(missing) > 0 and not fast_mode:
                print(f"   Fetching satellite for {len(missing)} "
                      f"missing rows")
                batch = []
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=8) as exe:
                    futs = {exe.submit(
                        data_fetch.fetch_temporal_satellite, row): idx
                        for idx, row in missing.iterrows()}
                    for fut in tqdm(
                            concurrent.futures.as_completed(futs),
                            total=len(missing)):
                        try:
                            res = fut.result()
                            idx = futs[fut]
                            if not pd.isna(res.get('nir08')):
                                for c in sat_cols:
                                    full_df.loc[idx, c] = res.get(c)
                        except Exception:
                            pass
                n_after = full_df['nir08'].notna().sum()
                print(f"   After fetch: {n_after}/{len(full_df)} "
                      f"have satellite data")
                # Update cache
                sat_out = full_df[full_df['nir08'].notna()][
                    merge_cols + sat_cols].drop_duplicates(
                    subset=merge_cols)
                sat_out.to_csv(config.SAT_CACHE, index=False)
        else:
            print("   No satellite cache found — fetching all…")
            full_df = train_df.copy()
            sat_cols = ['green', 'red', 'nir08', 'swir16', 'swir22',
                        'NDMI', 'MNDWI', 'NDVI']
            for c in sat_cols:
                full_df[c] = np.nan

            if not fast_mode:
                batch = []
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=8) as exe:
                    futs = {exe.submit(
                        data_fetch.fetch_temporal_satellite, row): idx
                        for idx, row in full_df.iterrows()}
                    for i, fut in tqdm(
                            enumerate(
                                concurrent.futures.as_completed(futs)),
                            total=len(full_df)):
                        try:
                            res = fut.result()
                            batch.append(res)
                        except Exception:
                            pass
                        # Save periodically
                        if len(batch) >= 50 or i + 1 == len(full_df):
                            if batch:
                                bdf = pd.DataFrame(batch)
                                hdr = not os.path.exists(config.SAT_CACHE)
                                bdf.to_csv(config.SAT_CACHE, mode='a',
                                           header=hdr, index=False)
                                batch = []
                # Re-merge
                if os.path.exists(config.SAT_CACHE):
                    sat_df = pd.read_csv(config.SAT_CACHE)
                    merge_cols = ['Latitude', 'Longitude', 'Sample Date']
                    full_df = robust_merge(
                        train_df, sat_df, merge_cols, how='left')

        full_df.to_csv("water_quality_post_merge.csv", index=False)
        print(f"   Post-merge: {len(full_df)} rows, "
              f"{len(full_df.columns)} cols")

        if not fast_mode:
            print("\n3. Enrichment (OSM + Terrain + Weather + Geo)")
            pre_cols = set(full_df.columns)
            full_df = data_fetch.enrich_dataset(full_df, config.OSM_CACHE)
            new_cols = set(full_df.columns) - pre_cols
            print(f"   Added {len(new_cols)} columns")
            full_df.to_csv(enriched_path, index=False)
        else:
            print("\n3. --fast mode: Skipping enrichment.")

    print("\n4. Imputation")
    full_df = imputation.diagnose_and_impute(
        full_df, fit_imputer=True, imputer_path="imputer_state.joblib")
    full_df.to_csv("water_quality_processed_final.csv", index=False)

    print("\n4b. Generating EDA report…")
    try:
        eda_report.generate_report(
            input_path="water_quality_processed_final.csv",
            output_path="eda_report.html",
            top_k=30,
        )
    except Exception as e:
        print(f"   EDA report failed (non-fatal): {e}")

    print("\n5. Training models")
    report = modeling.train_models(full_df, log_transform=None)

    print("\n6. Generating submission")
    generate_submission(full_df, report)


# SUBMISSION GENERATOR
def generate_submission(train_df, report):
    sub_path = os.path.join(config.DATA_DIR, "submission_template.csv")
    if not os.path.exists(sub_path):
        print(f"   '{sub_path}' not found – skipping.")
        return

    sub_df = pd.read_csv(sub_path)
    print(f"   {len(sub_df)} submission rows.")

    # Satellite
    print("   Fetching satellite for test rows")
    sat_res = []
    for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
        sat_res.append(data_fetch.fetch_temporal_satellite(row))
    sat_test = pd.DataFrame(sat_res)

    sub_for_merge = sub_df.drop(columns=config.TARGETS, errors='ignore').copy()
    sat_only_cols = [c for c in sat_test.columns
                     if c not in ['Latitude', 'Longitude', 'Sample Date']]
    test_df = sub_for_merge.reset_index(drop=True)
    for c in sat_only_cols:
        test_df[c] = sat_test[c].values

    # Enrich (if not in fast mode)
    if '--fast' not in sys.argv:
        print("   Enriching test rows…")
        test_df = data_fetch.enrich_dataset(test_df, config.OSM_CACHE)

    # Imputation 
    test_df = imputation.diagnose_and_impute(
        test_df, fit_imputer=False, imputer_path="imputer_state.joblib")

    # Feature engineering
    test_df = modeling.engineer_features(test_df)

    # KNN spatial features 
    print("   Adding KNN spatial features")
    knn_enc = report.get('_knn_encoder')
    if knn_enc is not None:
        test_df = knn_enc.transform(test_df, is_train=False)

    # Cross-target features
    xt_featurizer = report.get('_xt_featurizer')
    base_features = report.get('_base_features', [])
    if xt_featurizer is not None and base_features:
        print("   Adding cross-target features for test rows")
        # Use base features that exist in test_df
        xt_feats = [f for f in base_features if f in test_df.columns]
        test_df = xt_featurizer.transform_test(test_df, xt_feats)
        xt_cols = xt_featurizer.get_feature_names()
        print(f"   Cross-target columns added: {xt_cols}")

    # Predict
    for target in config.TARGETS:
        if target not in report:
            print(f"   Skipping {target}, no model.")
            continue

        model    = report[target]['model']
        feats    = report[target]['features']
        use_log  = report[target].get('log_transform', False)

        # Ensure all feature columns exist
        for f in feats:
            if f not in test_df.columns:
                test_df[f] = np.nan

        X_test = test_df[feats]
        raw = model.predict(np.array(X_test, dtype=np.float64))

        preds = np.expm1(raw) if use_log else raw
        preds = np.clip(preds, 0, None)

        sub_df[target] = preds
        print(f"   {target}: mean={preds.mean():.2f}  "
              f"std={preds.std():.2f}  "
              f"min={preds.min():.2f}  max={preds.max():.2f}")

    out = "submission.csv"
    sub_df.to_csv(out, index=False)
    print(f"\n   Submission saved to '{out}' ({len(sub_df)} rows)")


if __name__ == "__main__":
    main()
