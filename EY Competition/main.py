import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm

# Custom Modules
import config
import data_fetch
import imputation
import modeling

def clean_satellite_csv(csv_path):
    """
    Repairs a corrupted satellite CSV by enforcing consistent column order.
    This fixes issues where columns shift positions between appends.
    """
    if not os.path.exists(csv_path):
        return
    
    print(f"   Checking '{csv_path}' for corruption...")
    
    try:
        # Try to read with expected columns
        expected_cols = ['Latitude', 'Longitude', 'Sample Date', 
                        'green', 'red', 'nir08', 'swir16', 'swir22', 
                        'NDMI', 'MNDWI', 'NDVI']
        
        df = pd.read_csv(csv_path)
        
        # Check if columns match
        if list(df.columns) != expected_cols:
            print(f"   Column mismatch detected! Found: {list(df.columns)}")
            print(f"   Attempting repair...")
            
            # If we have all expected columns but in wrong order, reorder them
            if set(expected_cols).issubset(set(df.columns)):
                df = df[expected_cols]
                df.to_csv(csv_path, index=False)
                print(f"   Repaired column order!")
            else:
                missing = set(expected_cols) - set(df.columns)
                print(f"   Cannot repair - missing columns: {missing}")
                print(f"    Consider deleting '{csv_path}' and re-running.")
        else:
            print(f"   CSV structure looks good!")
            
    except Exception as e:
        print(f"   Error reading CSV: {e}")
        print(f"    Consider deleting '{csv_path}' and re-running.")

def main():
    print("==========================================")
    print("   WATER QUALITY PREDICTION PIPELINE")
    print("==========================================")
    
    # --- 1: LOAD BASE DATA ---
    print("\n1. Loading Base Data...")
    if not os.path.exists(config.TRAIN_FILE):
        print(f"Error: {config.TRAIN_FILE} not found.")
        return
    
    train_df = pd.read_csv(config.TRAIN_FILE)
    print(f"   Loaded {len(train_df)} rows.")

    # --- 2: SATELLITE DATA ACQUISITION---
    print("\n2. Processing Satellite Data...")
    
    if os.path.exists(config.SAT_CACHE):
        clean_satellite_csv(config.SAT_CACHE)
    
    # Identify what's already done
    finished_keys = set()
    if os.path.exists(config.SAT_CACHE):
        print(f"   Resuming from '{config.SAT_CACHE}'...")
        # Minimal load to check keys
        existing_sat = pd.read_csv(config.SAT_CACHE, usecols=['Latitude', 'Longitude', 'Sample Date'])
        finished_keys = set(
            existing_sat['Latitude'].astype(str) + "_" + 
            existing_sat['Longitude'].astype(str) + "_" + 
            existing_sat['Sample Date'].astype(str)
        )

    # Generate keys for current data
    train_df['key'] = (
        train_df['Latitude'].astype(str) + "_" + 
        train_df['Longitude'].astype(str) + "_" + 
        train_df['Sample Date'].astype(str)
    )

    rows_to_process = train_df[~train_df['key'].isin(finished_keys)]
    print(f"   Remaining rows to fetch: {len(rows_to_process)}")
    
    # Run Fetcher if needed
    if not rows_to_process.empty:
        batch_results = []
        save_interval = 50 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_idx = {
                executor.submit(data_fetch.fetch_temporal_satellite, row): idx 
                for idx, row in rows_to_process.iterrows()
            }
            
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(future_to_idx)), total=len(rows_to_process)):
                try:
                    result = future.result()
                    # Ensure identifiers are present with proper types
                    idx = future_to_idx[future]
                    row = train_df.loc[idx]
                    result['Latitude'] = float(row['Latitude'])
                    result['Longitude'] = float(row['Longitude'])
                    result['Sample Date'] = str(row['Sample Date'])
                    
                    batch_results.append(result)
                except Exception as e:
                    pass # Skip failures here, we catch them in rescue phase

                # Save Batch with ENFORCED column order
                if len(batch_results) >= save_interval or (i + 1) == len(rows_to_process):
                    new_df = pd.DataFrame(batch_results)
                    
                    # Enforce consistent column order to prevent CSV corruption
                    column_order = ['Latitude', 'Longitude', 'Sample Date', 
                                  'green', 'red', 'nir08', 'swir16', 'swir22', 
                                  'NDMI', 'MNDWI', 'NDVI']
                    new_df = new_df[column_order]
                    
                    header = not os.path.exists(config.SAT_CACHE)
                    new_df.to_csv(config.SAT_CACHE, mode='a', header=header, index=False)
                    batch_results = []
        print("   Primary fetch complete.")

    # --- 3: REPAIR FAILED FETCHES ---
    print("\n3. Verifying & Rescuing Failed Data...")
    sat_df = pd.read_csv(config.SAT_CACHE)
    
    # Ensure proper types after loading from CSV
    sat_df['Latitude'] = sat_df['Latitude'].astype(float)
    sat_df['Longitude'] = sat_df['Longitude'].astype(float)
    
    # Check for missing data (using nir08 as proxy for valid download)
    failed_rows = sat_df[sat_df['nir08'].isna()]
    
    if not failed_rows.empty:
        print(f"   Found {len(failed_rows)} failed downloads. Attempting rescue (relaxed constraints)...")
        
        rescued_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # We pass relaxed_mode=True to widen the search window
            futures = [executor.submit(data_fetch.fetch_temporal_satellite, row, relaxed_mode=True) 
                       for _, row in failed_rows.iterrows()]
            
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(failed_rows)):
                try:
                    res = f.result()
                    # Only keep if we actually got data this time
                    if not pd.isna(res.get('nir08')):  
                        rescued_results.append(res)
                except:
                    continue
        
        if rescued_results:
            print(f"   Rescued {len(rescued_results)} rows!")
            rescued_df = pd.DataFrame(rescued_results)
            
            # Enforce consistent column order for rescued data
            column_order = ['Latitude', 'Longitude', 'Sample Date', 
                          'green', 'red', 'nir08', 'swir16', 'swir22', 
                          'NDMI', 'MNDWI', 'NDVI']
            rescued_df = rescued_df[column_order]
            
            # Remove the still-broken rows from original and append the fixed ones
            clean_df = sat_df.dropna(subset=['nir08']) 
            
            sat_df = pd.concat([clean_df, rescued_df], ignore_index=True)
            sat_df = sat_df.drop_duplicates(subset=['Latitude', 'Longitude', 'Sample Date'])
            sat_df.to_csv(config.SAT_CACHE, index=False)
            print(f"   Updated satellite cache saved.")
        else:
            print("   Rescue attempt yielded no new data. Proceeding to imputation.")
    else:
        print("   All satellite data looks good.")
    
    # Reload satellite data to ensure we have the latest version
    print("   Reloading satellite data for merge...")
    sat_df = pd.read_csv(config.SAT_CACHE)
    sat_df['Latitude'] = sat_df['Latitude'].astype(float)
    sat_df['Longitude'] = sat_df['Longitude'].astype(float)

    # Merge Satellite Data
    train_df = train_df.drop(columns=['key'])
    
    # Fix data types before merge
    print("   Ensuring consistent data types for merge...")
    train_df['Latitude'] = train_df['Latitude'].astype(float)
    train_df['Longitude'] = train_df['Longitude'].astype(float)
    sat_df['Latitude'] = sat_df['Latitude'].astype(float)
    sat_df['Longitude'] = sat_df['Longitude'].astype(float)
    
    # Verify satellite data columns
    sat_cols = [c for c in sat_df.columns if c not in ['Latitude', 'Longitude', 'Sample Date']]
    print(f"   Satellite columns to merge: {sat_cols}")
    
    # Save satellite data checkpoint
    print(f"   Saving satellite data checkpoint to '{config.SAT_CACHE}'...")
    sat_df.to_csv(config.SAT_CACHE, index=False)
    
    full_df = pd.merge(train_df, sat_df, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    print(f"   Merged dataset has {len(full_df)} rows and {len(full_df.columns)} columns.")
    
    # Save post-merge checkpoint
    full_df.to_csv("water_quality_post_merge.csv", index=False)
    print("   Saved 'water_quality_post_merge.csv'")

    # --- 4: ENRICHMENT ---
    print("\n4. Enriching with OSM & Terrain Data...")
    pre_enrich_cols = set(full_df.columns)
    full_df = data_fetch.enrich_dataset(full_df, config.OSM_CACHE)
    
    new_cols = set(full_df.columns) - pre_enrich_cols
    print(f"   Added {len(new_cols)} new columns: {sorted(new_cols)}")
    
    full_df.to_csv("water_quality_post_enrichment.csv", index=False)
    print("   Saved 'water_quality_post_enrichment.csv'")

    # --- 5: IMPUTATION ---
    print("\n5. Running Statistical Imputation...")
    full_df = imputation.diagnose_and_impute(full_df)
    
    # Save final processed dataset
    full_df.to_csv("water_quality_processed_final.csv", index=False)
    print("   Saved 'water_quality_processed_final.csv'")
        
    # --- STEP 6: MODELING ---
    # Set to True to train on log1p(y) and convert predictions back via expm1.
    # Set to False to train on raw y directly.
    LOG_TRANSFORM = False

    print("\n6. Training Models...")
    print(f"   LOG_TRANSFORM = {LOG_TRANSFORM}")
    performance_report = modeling.train_models(full_df, log_transform=LOG_TRANSFORM)

    # --- 7: GENERATE SUBMISSION ---
    print("\n7. Generating Submission Predictions...")
    generate_submission(full_df, performance_report)


def generate_submission(train_df, performance_report):
    """
    Loads submission_template.csv, fetches features for test rows (re-using
    cached data where possible), engineers features, and predicts using
    the trained models.
    """

    sub_path = os.path.join(config.DATA_DIR, "submission_template.csv")
    if not os.path.exists(sub_path):
        print(f"   Warning: '{sub_path}' not found. Skipping submission.")
        return

    sub_df = pd.read_csv(sub_path)
    print(f"   Loaded {len(sub_df)} submission rows.")

    # --- Fetch satellite data for test rows ---
    print("   Fetching satellite data for submission rows...")
    sat_results = []
    for _, row in sub_df.iterrows():
        result = data_fetch.fetch_temporal_satellite(row)
        sat_results.append(result)
    sat_test = pd.DataFrame(sat_results)

    sub_for_merge = sub_df.drop(columns=config.TARGETS, errors='ignore').copy()
    # Round lat/lon to avoid float precision mismatches during merge
    sub_for_merge['Latitude'] = sub_for_merge['Latitude'].astype(float).round(6)
    sub_for_merge['Longitude'] = sub_for_merge['Longitude'].astype(float).round(6)
    sat_test['Latitude'] = sat_test['Latitude'].astype(float).round(6)
    sat_test['Longitude'] = sat_test['Longitude'].astype(float).round(6)

    test_df = pd.merge(sub_for_merge, sat_test,
                       on=['Latitude', 'Longitude', 'Sample Date'], how='left')

    # --- Enrich with OSM / terrain / weather / soil ---
    print("   Enriching submission rows...")
    test_df = data_fetch.enrich_dataset(test_df, config.OSM_CACHE)

    # --- Imputation ---
    test_df = imputation.diagnose_and_impute(test_df)

    # --- Feature engineering ---
    test_df = modeling.engineer_features(test_df)

    # --- Predict each target ---
    for target in config.TARGETS:
        if target not in performance_report:
            print(f"   Skipping {target} — no trained model.")
            continue

        model = performance_report[target]['model']
        features = performance_report[target]['features']

        for f in features:
            if f not in test_df.columns:
                test_df[f] = np.nan

        X_test = test_df[features]
        raw_preds = model.predict(np.array(X_test))

        use_log = performance_report[target].get('log_transform', False)
        if use_log:
            preds = np.expm1(raw_preds)
            n_nan = np.isnan(preds).sum()
            if n_nan > 0:
                print(f"     expm1 produced {n_nan} NaN predictions for {target}!")
        else:
            preds = raw_preds

        preds = np.clip(preds, 0, None)
        sub_df[target] = preds
        print(f"   {target}: mean={preds.mean():.2f}, std={preds.std():.2f}")

    # --- Save submission ---
    out_path = "submission.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"\n   Submission saved to '{out_path}' ({len(sub_df)} rows)")

if __name__ == "__main__":
    main()
