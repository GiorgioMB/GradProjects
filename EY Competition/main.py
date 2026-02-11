import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
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
    
    # --------------------------------------------------------
    # STEP 1: LOAD BASE DATA
    # --------------------------------------------------------
    print("\n1. Loading Base Data...")
    if not os.path.exists(config.TRAIN_FILE):
        print(f"Error: {config.TRAIN_FILE} not found.")
        return
    
    train_df = pd.read_csv(config.TRAIN_FILE)
    print(f"   Loaded {len(train_df)} rows.")

    # --------------------------------------------------------
    # STEP 2: SATELLITE DATA ACQUISITION
    # --------------------------------------------------------
    print("\n2. Processing Satellite Data...")
    
    # First, clean/validate existing satellite CSV if it exists
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
                    idx = future_to_idx[future]
                    row = train_df.loc[idx]
                    result['Latitude'] = float(row['Latitude'])
                    result['Longitude'] = float(row['Longitude'])
                    result['Sample Date'] = str(row['Sample Date'])
                    
                    batch_results.append(result)
                except Exception as e:
                    pass 
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

    # --------------------------------------------------------
    # STEP 3: REPAIR FAILED FETCHES
    # --------------------------------------------------------
    print("\n3. Verifying & Rescuing Failed Data...")
    sat_df = pd.read_csv(config.SAT_CACHE)
    
    sat_df['Latitude'] = sat_df['Latitude'].astype(float)
    sat_df['Longitude'] = sat_df['Longitude'].astype(float)
    
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
                    if not pd.isna(res.get('nir08')):
                        rescued_results.append(res)
                except:
                    continue
        
        if rescued_results:
            print(f"   Rescued {len(rescued_results)} rows!")
            rescued_df = pd.DataFrame(rescued_results)
            
            column_order = ['Latitude', 'Longitude', 'Sample Date', 
                          'green', 'red', 'nir08', 'swir16', 'swir22', 
                          'NDMI', 'MNDWI', 'NDVI']
            rescued_df = rescued_df[column_order]
            
            clean_df = sat_df.dropna(subset=['nir08'])  
            sat_df = pd.concat([clean_df, rescued_df], ignore_index=True)
            sat_df = sat_df.drop_duplicates(subset=['Latitude', 'Longitude', 'Sample Date'])
            sat_df.to_csv(config.SAT_CACHE, index=False)
            print(f"   Updated satellite cache saved.")
        else:
            print("   Rescue attempt yielded no new data. Proceeding to imputation.")
    else:
        print("   All satellite data looks good.")
    
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

    # --------------------------------------------------------
    # STEP 4: ENRICHMENT (OSM & TERRAIN)
    # --------------------------------------------------------
    print("\n4. Enriching with OSM & Terrain Data...")
    pre_enrich_cols = set(full_df.columns)
    full_df = data_fetch.enrich_dataset(full_df, config.OSM_CACHE)
    
    # Verify what was added
    new_cols = set(full_df.columns) - pre_enrich_cols
    print(f"   Added {len(new_cols)} new columns: {sorted(new_cols)}")
    
    # Save post-enrichment checkpoint
    full_df.to_csv("water_quality_post_enrichment.csv", index=False)
    print("   Saved 'water_quality_post_enrichment.csv'")

    # --------------------------------------------------------
    # STEP 5: IMPUTATION
    # --------------------------------------------------------
    print("\n5. Running Statistical Imputation...")
    full_df = imputation.diagnose_and_impute(full_df)
    
    # Save final processed dataset
    full_df.to_csv("water_quality_processed_final.csv", index=False)
    print("   Saved 'water_quality_processed_final.csv'")

    # --------------------------------------------------------
    # STEP 6: MODELING
    # --------------------------------------------------------
    print("\n6. Training Models...")
    modeling.train_models(full_df)

if __name__ == "__main__":
    main()
