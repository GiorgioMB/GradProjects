import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm

import config
import data_fetch
import imputation
import modeling
import eda_report
import dws_data


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
        print("\n1. Loading base data...")
        if not os.path.exists(config.TRAIN_FILE):
            print(f"   Error: {config.TRAIN_FILE} not found.")
            return
        train_df = pd.read_csv(config.TRAIN_FILE)
        print(f"   Loaded {len(train_df)} rows.")

        print("\n2. Satellite data...")
        if os.path.exists(config.SAT_CACHE):
            clean_satellite_csv(config.SAT_CACHE)
            sat_df = pd.read_csv(config.SAT_CACHE)
            print(f"   Satellite cache: {len(sat_df)} rows")

            # Robust merge on rounded lat/lon + date string
            merge_cols = ['Latitude', 'Longitude', 'Sample Date']
            full_df = robust_merge(train_df, sat_df, merge_cols, how='left')

            # Check coverage
            sat_cols = ['green', 'red', 'nir08', 'swir16', 'swir22',
                        'NDMI', 'MNDWI', 'NDVI']
            n_with_sat = full_df[sat_cols[0]].notna().sum()
            print(f"   Merged: {n_with_sat}/{len(full_df)} rows "
                  f"have satellite data ({100*n_with_sat/len(full_df):.0f}%)")

            # Fetch missing satellite data (if any)
            missing = full_df[full_df['nir08'].isna()]
            if len(missing) > 0 and not fast_mode:
                print(f"   Fetching satellite for {len(missing)} "
                      f"missing rows…")
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
            print("\n3. Enrichment (OSM + Terrain + Weather + Geo)...")
            pre_cols = set(full_df.columns)
            full_df = data_fetch.enrich_dataset(full_df, config.OSM_CACHE)
            new_cols = set(full_df.columns) - pre_cols
            print(f"   Added {len(new_cols)} columns")
            full_df.to_csv(enriched_path, index=False)
        else:
            print("\n3. --fast mode: Skipping enrichment.")

    print("\n3b. DWS External Data Integration...")
    sub_path = os.path.join(config.DATA_DIR, "submission_template.csv")
    if os.path.exists(sub_path):
        test_template = pd.read_csv(sub_path)
        # Parse dates early so DWS can use them
        if "Sample Date" in test_template.columns:
            test_template["date"] = pd.to_datetime(
                test_template["Sample Date"], dayfirst=True)
        # Build test-date exclusion dict: {station: set(dates)} for leakage prevention
        test_dates_by_stn = {}
        for _, trow in test_template.iterrows():
            stn = dws_data.coord_to_station(trow['Latitude'], trow['Longitude'])
            if stn:
                dt = pd.Timestamp(trow['date']).date() if pd.notna(trow.get('date')) else None
                if dt is None and 'Sample Date' in trow.index:
                    dt = pd.to_datetime(trow['Sample Date'], dayfirst=True).date()
                if dt:
                    test_dates_by_stn.setdefault(stn, set()).add(dt)
        print(f"   Test-date exclusion: {sum(len(v) for v in test_dates_by_stn.values())} "
              f"dates across {len(test_dates_by_stn)} stations")

        dws_context = None
        try:
            all_dws, station_features, aug_rows = \
                dws_data.prepare_dws_augmentation(
                    config.DWS_DIR, config.TARGETS, test_template,
                    train_df=full_df,
                    train_csv_path=config.TRAIN_FILE)
            dws_context = {
                'all_dws': all_dws,
                'station_features': station_features,
                'aug_rows': aug_rows,
                'test_dates_by_stn': test_dates_by_stn,
                'test_template': test_template,  # for adversarial validation
            }
        except Exception as e:
            print(f"   Warning: DWS integration failed (non-fatal): {e}")
            traceback.print_exc()
            dws_context = None
    else:
        print(f"   Warning: No submission template at {sub_path} – skipping DWS")
        dws_context = None

    print("\n4. Imputation...")
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
        print(f"   Warning: EDA report failed (non-fatal): {e}")

    print("\n5. Training models (v6 DWS-centric)...")
    report = modeling.train_models(
        full_df, log_transform=None, use_optuna=True,
        dws_context=dws_context)

    print("\n6. Generating submission...")
    generate_submission(train_df=full_df, report=report)


# SUBMISSION GENERATOR  (v6 DWS-centric)
def generate_submission(train_df, report):
    sub_path = os.path.join(config.DATA_DIR, "submission_template.csv")
    if not os.path.exists(sub_path):
        print(f"   '{sub_path}' not found – skipping.")
        return

    sub_df = pd.read_csv(sub_path)
    print(f"   {len(sub_df)} submission rows.")

    # ── Satellite data for test rows ─────────────────────────────────────
    print("   Fetching satellite for test rows…")
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

    # ── Enrichment (OSM + Terrain + Weather) ─────────────────────────────
    if '--fast' not in sys.argv:
        print("   Enriching test rows...")
        test_df = data_fetch.enrich_dataset(test_df, config.OSM_CACHE)

    # ── Imputation ───────────────────────────────────────────────────────
    test_df = imputation.diagnose_and_impute(
        test_df, fit_imputer=False, imputer_path="imputer_state.joblib")

    # ── Feature engineering (temporal, spectral, terrain, etc.) ──────────
    test_df = modeling.engineer_features(test_df)

    # ── DWS features for test rows (v6 consistent) ──────────────────────
    dws_context = report.get('_dws_context')
    if dws_context is not None:
        print("   Building DWS features for test rows (v6)…")
        try:
            all_dws = dws_context['all_dws']

            # Parse dates
            if 'date' not in test_df.columns or test_df['date'].dtype == object:
                if 'Sample Date' in test_df.columns:
                    test_df['date'] = pd.to_datetime(
                        test_df['Sample Date'], dayfirst=True)

            # Assign station codes
            test_df['_station'] = test_df.apply(
                lambda r: dws_data.coord_to_station(
                    r['Latitude'], r['Longitude']),
                axis=1,
            )
            n_matched = test_df['_station'].notna().sum()
            print(f"   DWS station match: {n_matched}/{len(test_df)} test rows")

            # Same-day auxiliary chemistry features (dws_aux_*)
            test_df = dws_data.add_sameday_aux_features(test_df, all_dws)

            # Lag features (lag_*, roll*)
            test_df = dws_data.add_lag_features(
                test_df, all_dws, config.TARGETS)

            # Station historical features (stn_*)
            test_df = modeling.add_station_historical_features(
                test_df, all_dws)

            # Neighbor/upstream features
            test_df = dws_data.build_neighbor_features(
                test_df, all_dws)

            # Time-weighted station features (stn_tw_*)
            test_df = modeling.add_time_weighted_station_features(
                test_df, all_dws, halflife_days=730)

            # Lag for auxiliary chemistry
            for stn_code in test_df['_station'].dropna().unique():
                if stn_code not in all_dws:
                    continue
                sdf = all_dws[stn_code].sort_values('date')
                mask = test_df['_station'] == stn_code
                for aux in ['pH_Diss_Water', 'Ca_Diss_Water',
                            'Mg_Diss_Water', 'Na_Diss_Water',
                            'Cl_Diss_Water', 'SO4_Diss_Water']:
                    col_name = f'lag_aux_{aux}'
                    if col_name not in test_df.columns:
                        test_df[col_name] = np.nan
                    for idx in test_df.loc[mask].index:
                        dt = test_df.at[idx, 'date']
                        if pd.isna(dt):
                            continue
                        prev = sdf[(sdf['date'] < dt) & sdf[aux].notna()] \
                            if aux in sdf.columns else pd.DataFrame()
                        if len(prev) > 0:
                            test_df.at[idx, col_name] = float(
                                prev.iloc[-1][aux])

            # ── LSTM predictions computed per-target in predict loop below ─

            # Drop helper columns (keep _station for LSTM)
            test_df.drop(columns=['_dws_station', 'station'],
                         errors='ignore', inplace=True)

            # ── Missingness indicators (match training) ──────────────
            # Test rows are at DWS stations, so they're like DWS data
            # but built from competition-style enrichment
            test_df['_is_dws'] = 0  # test rows come as competition format
            test_df['_has_enrichment'] = 1  # test rows have their own satellite/terrain
            test_df['_enrichment_dist_km'] = 0.0

            dws_aux_cols = [c for c in test_df.columns if c.startswith('dws_aux_')]
            if dws_aux_cols:
                test_df['_has_dws_aux'] = test_df[dws_aux_cols].notna().any(axis=1).astype(int)
            else:
                test_df['_has_dws_aux'] = 0

            lag_cols = [c for c in test_df.columns if c.startswith('lag_') or c.startswith('roll')]
            if lag_cols:
                test_df['_has_dws_lag'] = test_df[lag_cols].notna().any(axis=1).astype(int)
            else:
                test_df['_has_dws_lag'] = 0

            stn_cols = [c for c in test_df.columns if c.startswith('stn_')]
            if stn_cols:
                test_df['_has_stn_hist'] = test_df[stn_cols].notna().any(axis=1).astype(int)
            else:
                test_df['_has_stn_hist'] = 0
        except Exception as e:
            print(f"   Warning: DWS test features failed (non-fatal): {e}")
            traceback.print_exc()

    # ── KNN spatial features ─────────────────────────────────────────────
    print("   Adding KNN spatial features…")
    knn_enc = report.get('_knn_encoder')
    if knn_enc is not None:
        test_df = knn_enc.transform(test_df, is_train=False)

    # ── Predict (chained: EC → TAL → DRP) ──────────────────────────────
    lstm_models = report.get('_lstm_models', {})
    chain_order = report.get('_chain_order', config.TARGETS)
    chain_preds = {}  # {target: array of predictions for chaining}

    for target in chain_order:
        if target not in report:
            print(f"   Skipping {target}, no model.")
            continue

        model    = report[target]['model']
        feats    = report[target]['features']
        use_log  = report[target].get('log_transform', False)
        target_tf = report[target].get('target_transformer')

        # ── Add chained predictions from upstream targets ────────────
        for prev_target, prev_preds in chain_preds.items():
            chain_col = f'_chain_{prev_target[:3].upper()}'
            test_df[chain_col] = prev_preds

        # ── Add adversarial score feature if available ───────────────
        adv_clf = report.get(f'_adv_classifier_{target}')
        adv_feats = report.get(f'_adv_features_{target}')
        if adv_clf is not None and adv_feats is not None:
            try:
                test_adv_feats = [f for f in adv_feats if f in test_df.columns]
                if len(test_adv_feats) == len(adv_feats):
                    X_adv = np.nan_to_num(test_df[test_adv_feats].values, nan=0.0)
                    test_df['_adv_score'] = adv_clf.predict_proba(X_adv)[:, 1]
                else:
                    test_df['_adv_score'] = 0.5
            except Exception:
                test_df['_adv_score'] = 0.5
        use_log  = report[target].get('log_transform', False)

        # Compute LSTM prediction for this target
        if target in lstm_models and lstm_models[target] is not None:
            lstm_m = lstm_models[target]
            if lstm_m.model_ is not None:
                try:
                    print(f"   Computing LSTM predictions for {target}...")
                    lstm_preds = lstm_m.predict(
                        dws_context['all_dws'] if dws_context else {},
                        test_df)
                    test_df['_lstm_pred'] = lstm_preds
                    n_valid = np.isfinite(lstm_preds).sum()
                    print(f"   LSTM: {n_valid}/{len(test_df)} valid")
                except Exception as e:
                    print(f"   LSTM prediction failed: {e}")
                    test_df['_lstm_pred'] = np.nan
            else:
                test_df['_lstm_pred'] = np.nan
        else:
            test_df['_lstm_pred'] = np.nan

        # Ensure all feature columns exist
        for f in feats:
            if f not in test_df.columns:
                test_df[f] = np.nan

        X_test = np.array(test_df[feats], dtype=np.float64)
        # Fix inf only; StackingEnsemble handles NaN per-model
        X_test = np.where(np.isinf(X_test), 0.0, X_test)
        raw = model.predict(X_test)

        # Stacking ensemble already uses _lstm_pred as an L1 feature
        # and learns optimal combination via L2 Ridge meta-learner.
        # No hardcoded blending needed — the meta-learner handles it.
        if target_tf is not None:
            preds = target_tf.inverse_transform(raw)
        elif use_log:
            preds = np.expm1(raw)
        else:
            preds = raw
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        preds = np.clip(preds, 0, None)

        # ── Per-station calibration (DISABLED — biases toward DWS
        #    historical distribution, hurts competition scoring) ──────
        # calibrator = report[target].get('calibrator')
        # if calibrator is not None:
        #     stations = test_df.get('_station')
        #     if stations is not None:
        #         preds_before = preds.copy()
        #         preds = calibrator.calibrate(preds, stations.values)
        #         diff = np.abs(preds - preds_before).mean()
        #         print(f"   Calibration: mean |delta|={diff:.2f}")

        # Store for chaining to downstream targets
        chain_preds[target] = preds

        sub_df[target] = preds
        print(f"   {target}: mean={preds.mean():.2f}  "
              f"std={preds.std():.2f}  "
              f"min={preds.min():.2f}  max={preds.max():.2f}")

    # ── V8: Per-station PRIMARY prediction + confidence blending ──────
    #   Per-station regression is the PRIMARY predictor (not a refinement).
    #   Global model serves as a safety net for rows where per-station
    #   confidence is low.
    if dws_context is not None:
        print("\n   V8 Per-station PRIMARY prediction…")
        try:
            all_dws = dws_context['all_dws']
            ps_preds, ps_confs = modeling.per_station_predict(sub_df, all_dws)

            # Variance-decompress the global model predictions
            import dws_data as dws_mod_main
            print("   Variance decompression on global predictions…")
            for target in config.TARGETS:
                if target not in ps_preds:
                    continue
                dws_col = dws_mod_main._COL_FOR_TARGET.get(target)
                if dws_col is None:
                    continue

                # Build decompressor from DWS historical data
                decompressor = modeling.VarianceDecompressor()
                exclude_dates = dws_context.get('test_dates_by_stn', {})
                decompressor.fit(all_dws, dws_col,
                                 exclude_dates=exclude_dates)

                # Get station assignments for test rows
                stations = sub_df.apply(
                    lambda r: dws_mod_main.coord_to_station(
                        r['Latitude'], r['Longitude']),
                    axis=1).values

                # Decompress global predictions
                global_preds = sub_df[target].values.copy()
                global_decompressed = decompressor.decompress(
                    global_preds, stations)

                ps = ps_preds[target]
                conf = ps_confs[target]

                # Confidence-weighted blend:
                #   final = α × per_station + (1 - α) × global_decompressed
                # where α = per-station confidence (clamped)
                alpha = np.clip(conf, 0.3, 0.95)

                # For EC (near-deterministic from aux chemistry), push α higher
                if 'Conductance' in target:
                    alpha = np.clip(conf * 1.3, 0.5, 0.98)
                # For TAL, moderate boost
                elif 'Alkalinity' in target:
                    alpha = np.clip(conf * 1.1, 0.4, 0.95)
                # For DRP (hardest target), keep more conservative
                else:
                    alpha = np.clip(conf, 0.3, 0.90)

                blended = alpha * ps + (1 - alpha) * global_decompressed
                blended = np.clip(blended, 0, None)

                # Report blend stats
                ps_valid = np.isfinite(ps) & (ps > 0)
                n_ps = ps_valid.sum()
                mean_alpha = alpha[ps_valid].mean() if n_ps > 0 else 0
                print(f"      {target}: {n_ps}/{len(sub_df)} ps-valid, "
                      f"mean_α={mean_alpha:.3f}")
                print(f"         global:  mean={global_preds.mean():.2f}  "
                      f"std={global_preds.std():.2f}")
                print(f"         decomp:  mean={global_decompressed.mean():.2f}  "
                      f"std={global_decompressed.std():.2f}")
                print(f"         ps:      mean={ps.mean():.2f}  "
                      f"std={ps.std():.2f}")
                print(f"         blended: mean={blended.mean():.2f}  "
                      f"std={blended.std():.2f}")

                sub_df[target] = blended

        except Exception as e:
            print(f"   Warning: Per-station prediction failed: {e}")
            traceback.print_exc()

    out = "submission.csv"
    sub_df.to_csv(out, index=False)
    print(f"\n   Submission saved to '{out}' ({len(sub_df)} rows)")

    # ── Score against DWS ground truth ───────────────────────────────────
    score_against_dws(sub_df, report)


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL SCORING – compare predictions to DWS ground truth
# ─────────────────────────────────────────────────────────────────────────────
def score_against_dws(sub_df, report=None):
    """
    Since 100% of test rows are DWS stations with same-day measurements,
    we can look up the actual target values and compute R² locally.
    This gives the exact competition score without submitting.
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    print("\n" + "=" * 65)
    print("   LOCAL SCORING vs DWS GROUND TRUTH")
    print("=" * 65)

    # Load DWS data
    try:
        all_dws = dws_data.load_all_station_data(config.DWS_DIR)
    except Exception as e:
        print(f"   Cannot load DWS data: {e}")
        return None

    sub = sub_df.copy()
    sub['date'] = pd.to_datetime(sub['Sample Date'], dayfirst=True)
    sub['_station'] = sub.apply(
        lambda r: dws_data.coord_to_station(r['Latitude'], r['Longitude']),
        axis=1)

    n_matched = sub['_station'].notna().sum()
    print(f"   Station match: {n_matched}/{len(sub)} test rows")

    if n_matched == 0:
        print("   No stations matched — cannot score.")
        return None

    # Look up DWS values for each test row
    target_scores = {}
    for target in config.TARGETS:
        if target not in sub.columns:
            continue

        dws_col = dws_data._COL_FOR_TARGET.get(target)
        if dws_col is None:
            print(f"   {target}: no DWS column mapping")
            continue

        # NOTE: load_all_station_data() already applies unit conversions
        # (EC ×10, DRP ×1000) so we must NOT multiply again here.

        y_true, y_pred = [], []
        matched, missed = 0, 0

        for idx, row in sub.iterrows():
            stn = row['_station']
            dt = row['date']
            pred = row[target]

            if pd.isna(stn) or pd.isna(dt) or stn not in all_dws:
                missed += 1
                continue

            sdf = all_dws[stn]
            # Exact date match
            match = sdf[sdf['date'].dt.date == dt.date()]
            if len(match) == 0:
                missed += 1
                continue

            raw_val = pd.to_numeric(match.iloc[0].get(dws_col), errors='coerce')
            if pd.isna(raw_val):
                missed += 1
                continue

            # raw_val is already unit-converted by load_all_station_data
            actual = raw_val
            y_true.append(actual)
            y_pred.append(pred)
            matched += 1

        if len(y_true) < 2:
            print(f"   {target}: only {len(y_true)} matches — cannot score")
            continue

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1]

        target_scores[target] = r2

        print(f"\n   {target}:")
        print(f"      Matched: {matched}/{len(sub)} rows "
              f"({missed} missed)")
        print(f"      R²   = {r2:.4f}")
        print(f"      RMSE = {rmse:.2f}")
        print(f"      MAE  = {mae:.2f}")
        print(f"      Corr = {corr:.4f}")
        print(f"      True:  mean={y_true.mean():.2f}  "
              f"std={y_true.std():.2f}  "
              f"min={y_true.min():.2f}  max={y_true.max():.2f}")
        print(f"      Pred:  mean={y_pred.mean():.2f}  "
              f"std={y_pred.std():.2f}  "
              f"min={y_pred.min():.2f}  max={y_pred.max():.2f}")

        # Per-station breakdown for worst stations
        print(f"      Per-station R² (worst 5):")
        stn_scores = []
        for stn in sub['_station'].dropna().unique():
            stn_mask = sub['_station'] == stn
            stn_sub = sub[stn_mask]
            stn_true, stn_pred = [], []
            for _, row in stn_sub.iterrows():
                dt = row['date']
                sdf = all_dws.get(stn)
                if sdf is None:
                    continue
                match = sdf[sdf['date'].dt.date == dt.date()]
                if len(match) == 0:
                    continue
                raw_val = pd.to_numeric(
                    match.iloc[0].get(dws_col), errors='coerce')
                if pd.isna(raw_val):
                    continue
                stn_true.append(raw_val)  # already unit-converted
                stn_pred.append(row[target])
            if len(stn_true) >= 2:
                sr2 = r2_score(stn_true, stn_pred)
                stn_scores.append((stn, sr2, len(stn_true),
                                   np.mean(stn_true)))
        stn_scores.sort(key=lambda x: x[1])
        for stn, sr2, n, tmean in stn_scores[:5]:
            print(f"         {stn}: R²={sr2:.4f}  "
                  f"(n={n}, true_mean={tmean:.1f})")

    if target_scores:
        avg_r2 = np.mean(list(target_scores.values()))
        print(f"\n   {'='*50}")
        print(f"   COMPETITION SCORE (avg R²): {avg_r2:.4f}")
        print(f"   {'='*50}")
        for t, r2 in target_scores.items():
            short = t[:3].upper()
            print(f"      {short}: {r2:.4f}")
        return avg_r2
    return None


if __name__ == "__main__":
    if '--score-only' in sys.argv:
        # Quick scoring mode: just score an existing submission.csv
        sub_path = "submission.csv"
        for arg in sys.argv[1:]:
            if arg.endswith('.csv') and arg != '--score-only':
                sub_path = arg
                break
        if not os.path.exists(sub_path):
            print(f"No submission file at '{sub_path}'")
            sys.exit(1)
        print(f"Scoring '{sub_path}' against DWS ground truth...")
        sub_df = pd.read_csv(sub_path)
        score_against_dws(sub_df)

    elif '--per-station' in sys.argv:
        # Quick per-station prediction mode:
        #   Skips global model training — only uses per-station regression
        #   + temporal interpolation from DWS data.
        #   Usage:  python main.py --per-station
        print("=" * 60)
        print("   PER-STATION PREDICTION MODE (v7)")
        print("=" * 60)

        # 1. Load DWS data
        print("\n1. Loading DWS data...")
        registry = dws_data.get_full_registry(
            train_csv_path=config.TRAIN_FILE, dws_dir=config.DWS_DIR)
        dws_data.fetch_dws_data(config.DWS_DIR, registry=registry)
        all_dws = dws_data.load_all_station_data(
            config.DWS_DIR, registry=registry)
        print(f"   {len(all_dws)} stations, "
              f"{sum(len(d) for d in all_dws.values())} total rows")

        # 2. Load submission template
        print("\n2. Loading test template...")
        sub_path = os.path.join(config.DATA_DIR, "submission_template.csv")
        sub_df = pd.read_csv(sub_path)

        # 3. Per-station predict
        print("\n3. Per-station prediction…")
        ps_preds, ps_confs = modeling.per_station_predict(sub_df, all_dws)
        for target in config.TARGETS:
            if target in ps_preds:
                sub_df[target] = ps_preds[target]

        # 4. Save
        out = "submission.csv"
        sub_df.to_csv(out, index=False)
        print(f"\n4. Saved '{out}' ({len(sub_df)} rows)")

        # 5. Score
        score_against_dws(sub_df)

    else:
        main()
