import numpy as np
import pandas as pd
import traceback
import xgboost as xgb
import warnings

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    warnings.warn("LightGBM not installed – ensemble will skip it.")
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import BallTree
from scipy import stats as sp_stats
import joblib
import config

try:
    from sarimax_model import StationSARIMAX, blend_predictions
    HAS_SARIMAX = True
except ImportError:
    HAS_SARIMAX = False


# 0.  FEATURE ENGINEERING  
def engineer_features(df):
    df = df.copy()
    eps = 1e-8

    if 'Sample Date' in df.columns:
        dt = pd.to_datetime(df['Sample Date'], dayfirst=True)
        df['Month']     = dt.dt.month
        df['DayOfYear'] = dt.dt.dayofyear
        df['Month_sin']     = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos']     = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

    if 'nir08' in df.columns and 'red' in df.columns:
        df['NDVI'] = ((df['nir08'] - df['red']) /
                      (df['nir08'] + df['red'] + eps))
    if 'green' in df.columns and 'nir08' in df.columns:
        df['NDWI'] = ((df['green'] - df['nir08']) /
                      (df['green'] + df['nir08'] + eps))
    if 'nir08' in df.columns and 'swir16' in df.columns:
        df['NDMI_calc'] = ((df['nir08'] - df['swir16']) /
                           (df['nir08'] + df['swir16'] + eps))
    if 'red' in df.columns and 'green' in df.columns:
        df['Turbidity_proxy'] = df['red'] / (df['green'] + eps)
    if 'nir08' in df.columns and 'red' in df.columns:
        df['Chl_proxy'] = df['nir08'] / (df['red'] + eps)
    if 'nir08' in df.columns and 'green' in df.columns:
        df['NIR_Green_ratio'] = df['nir08'] / (df['green'] + eps)
    if all(c in df.columns for c in ['swir16', 'red', 'nir08', 'green']):
        df['BSI'] = (((df['swir16'] + df['red']) - (df['nir08'] + df['green'])) /
                     ((df['swir16'] + df['red']) + (df['nir08'] + df['green']) + eps))
    if 'swir16' in df.columns and 'swir22' in df.columns:
        df['SWIR_ratio'] = df['swir16'] / (df['swir22'] + eps)
    if 'nir08' in df.columns and 'swir22' in df.columns:
        df['NBR'] = ((df['nir08'] - df['swir22']) /
                     (df['nir08'] + df['swir22'] + eps))
    if 'green' in df.columns and 'red' in df.columns:
        df['Green_Red_ratio'] = df['green'] / (df['red'] + eps)

    if 'elevation_mean' in df.columns:
        df['log_Elev'] = np.log1p(np.clip(df['elevation_mean'], 0, None))
        if 'slope_mean' in df.columns:
            df['Elev_Slope'] = df['elevation_mean'] * df['slope_mean']

    if 'rain_7d_sum' in df.columns and 'rain_30d_sum' in df.columns:
        df['Rain_7d_frac'] = df['rain_7d_sum'] / (df['rain_30d_sum'] + eps)
    if 'elevation_mean' in df.columns and 'rain_30d_sum' in df.columns:
        df['Elev_Rain'] = df['elevation_mean'] * df['rain_30d_sum']

    if 'lc_cropland' in df.columns and 'rain_30d_sum' in df.columns:
        df['Cropland_Rain'] = df['lc_cropland'] * df['rain_30d_sum']
    if 'lc_built_up' in df.columns and 'rain_30d_sum' in df.columns:
        df['Urban_Rain'] = df['lc_built_up'] * df['rain_30d_sum']
    if 'lc_cropland' in df.columns:
        natural   = (df.get('lc_tree_cover', 0) +
                     df.get('lc_shrubland', 0) +
                     df.get('lc_grassland', 0))
        disturbed = (df.get('lc_cropland', 0) +
                     df.get('lc_built_up', 0) +
                     df.get('lc_bare_sparse', 0))
        df['Natural_vs_Disturbed'] = natural / (disturbed + eps)

    if 'water_fraction' in df.columns and 'elevation_mean' in df.columns:
        df['Water_Elev'] = df['water_fraction'] * df['elevation_mean']
    if 'geo_is_karst' in df.columns and 'rain_30d_sum' in df.columns:
        df['Karst_Rain'] = df['geo_is_karst'] * df['rain_30d_sum']

    if 'green' in df.columns and 'swir16' in df.columns:
        df['Green_SWIR_ratio'] = df['green'] / (df['swir16'] + eps)
    if 'red' in df.columns and 'swir22' in df.columns:
        df['Red_SWIR22_ratio'] = df['red'] / (df['swir22'] + eps)
    if all(c in df.columns for c in ['green', 'red', 'nir08']):
        df['Brightness'] = (df['green'] + df['red'] + df['nir08']) / 3.0
        df['Greenness']  = df['nir08'] - (df['green'] + df['red']) / 2.0
    if all(c in df.columns for c in ['swir16', 'nir08', 'red']):
        df['SAVI'] = 1.5 * (df['nir08'] - df['red']) / (df['nir08'] + df['red'] + 0.5 + eps)
        df['EVI']  = 2.5 * (df['nir08'] - df['red']) / (df['nir08'] + 6*df['red'] - 7.5*0.1*df['swir16'] + 1 + eps)
    if 'red' in df.columns and 'swir16' in df.columns:
        df['Red_SWIR16_diff'] = df['red'] - df['swir16']

    if 'elevation_std' in df.columns:
        df['Terrain_Roughness'] = df['elevation_std']
    if 'slope_mean' in df.columns:
        df['log_Slope'] = np.log1p(np.clip(df['slope_mean'], 0, None))
    if 'aspect_mean' in df.columns:
        df['Aspect_sin'] = np.sin(np.deg2rad(df['aspect_mean']))
        df['Aspect_cos'] = np.cos(np.deg2rad(df['aspect_mean']))
    if 'slope_std' in df.columns:
        df['Slope_variability'] = df['slope_std']
    if 'elevation_mean' in df.columns and 'elevation_std' in df.columns:
        df['Elev_CV'] = df['elevation_std'] / (df['elevation_mean'] + eps)

    if 'rain_30d_sum' in df.columns and 'temp_30d_mean' in df.columns:
        df['Rain_Temp'] = df['rain_30d_sum'] * df['temp_30d_mean']
    if 'rain_7d_sum' in df.columns and 'temp_30d_mean' in df.columns:
        df['Rain7d_Temp'] = df['rain_7d_sum'] * df['temp_30d_mean']
    if 'et_30d_sum' in df.columns and 'rain_30d_sum' in df.columns:
        df['ET_Rain_ratio'] = df['et_30d_sum'] / (df['rain_30d_sum'] + eps)
    if 'et_7d_sum' in df.columns and 'rain_7d_sum' in df.columns:
        df['ET7d_Rain7d_ratio'] = df['et_7d_sum'] / (df['rain_7d_sum'] + eps)
    if 'wind_30d_mean' in df.columns and 'rain_30d_sum' in df.columns:
        df['Wind_Rain'] = df['wind_30d_mean'] * df['rain_30d_sum']
    if 'radiation_30d_mean' in df.columns and 'temp_30d_mean' in df.columns:
        df['Radiation_Temp'] = df['radiation_30d_mean'] * df['temp_30d_mean']
    if 'humidity_30d_mean' in df.columns and 'rain_30d_sum' in df.columns:
        df['Humidity_Rain'] = df['humidity_30d_mean'] * df['rain_30d_sum']

    if 'water_balance_30d' in df.columns and 'elevation_mean' in df.columns:
        df['WaterBal_Elev'] = df['water_balance_30d'] * df['elevation_mean']
    # Humidity_Temp already created in 'Rainfall / weather interactions' above
    if 'water_balance_30d' in df.columns and 'lc_cropland' in df.columns:
        df['WaterBal_Crop'] = df['water_balance_30d'] * df['lc_cropland']

    if 'lc_tree_cover' in df.columns and 'rain_30d_sum' in df.columns:
        df['Forest_Rain'] = df['lc_tree_cover'] * df['rain_30d_sum']
    if 'lc_grassland' in df.columns and 'rain_30d_sum' in df.columns:
        df['Grass_Rain'] = df['lc_grassland'] * df['rain_30d_sum']
    if 'lc_bare_sparse' in df.columns and 'rain_30d_sum' in df.columns:
        df['Bare_Rain'] = df['lc_bare_sparse'] * df['rain_30d_sum']
    if 'lc_herbaceous_wetland' in df.columns:
        df['Wetland_frac'] = df['lc_herbaceous_wetland']
    if 'lc_water' in df.columns and 'water_fraction' in df.columns:
        df['LC_Water_x_JRC'] = df['lc_water'] * df['water_fraction']
    if 'lc_cropland' in df.columns and 'lc_built_up' in df.columns:
        df['Anthropogenic_total'] = df['lc_cropland'] + df['lc_built_up']
    if 'lc_cropland' in df.columns and 'elevation_mean' in df.columns:
        df['Cropland_Elev'] = df['lc_cropland'] * df['elevation_mean']

    if 'water_occurrence_mean' in df.columns and 'rain_30d_sum' in df.columns:
        df['WaterOcc_Rain'] = df['water_occurrence_mean'] * df['rain_30d_sum']
    if 'water_seasonality' in df.columns and 'temp_30d_mean' in df.columns:
        df['WaterSeason_Temp'] = df['water_seasonality'] * df['temp_30d_mean']
    if 'nearest_dam_dist_m' in df.columns:
        df['log_Dam_dist'] = np.log1p(np.clip(df['nearest_dam_dist_m'], 0, None))

    if 'geo_lith_category' in df.columns and 'rain_30d_sum' in df.columns:
        df['Geology_Rain'] = df['geo_lith_category'] * df['rain_30d_sum']
    if 'geo_rock_age_ma' in df.columns:
        df['log_RockAge'] = np.log1p(np.clip(df['geo_rock_age_ma'], 0, None))
    if 'geo_is_karst' in df.columns and 'elevation_mean' in df.columns:
        df['Karst_Elev'] = df['geo_is_karst'] * df['elevation_mean']

    if 'pop_density_proxy' in df.columns:
        df['log_PopDensity'] = np.log1p(np.clip(df['pop_density_proxy'], 0, None))
    if 'pop_built_up_5km' in df.columns and 'rain_30d_sum' in df.columns:
        df['Urban5km_Rain'] = df['pop_built_up_5km'] * df['rain_30d_sum']

    return df


# 0b.  KNN SPATIAL FEATURES  (gentle spatial prior)
class SpatialKNNEncoder:
    def __init__(self, targets, k_values=(5, 10, 15, 25)):
        self.targets  = targets
        self.k_values = k_values
        self.tree_    = None

    def fit(self, df):
        coords = np.deg2rad(df[['Latitude', 'Longitude']].values)
        self.tree_ = BallTree(coords, metric='haversine')
        self.train_coords_ = coords
        self.train_targets_ = {}
        for t in self.targets:
            if t in df.columns:
                self.train_targets_[t] = df[t].values.copy()
        return self

    def transform(self, df, is_train=False):
        df = df.copy()
        coords = np.deg2rad(df[['Latitude', 'Longitude']].values)
        max_k = max(self.k_values)
        k_q = max_k + 1 if is_train else max_k
        # Clamp to number of available training points
        k_q = min(k_q, self.tree_.data.shape[0])
        if k_q < 2:
            return df
        dists_all, inds_all = self.tree_.query(coords, k=k_q)

        if is_train:
            inds_all  = inds_all[:, 1:]
            dists_all = dists_all[:, 1:]

        for t in self.targets:
            if t not in self.train_targets_:
                continue
            y_all = self.train_targets_[t]

            for k in self.k_values:
                if k > inds_all.shape[1]:
                    continue
                inds  = inds_all[:, :k]
                dists = dists_all[:, :k]
                vals = y_all[inds]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[f'{t}_knn{k}_mean']   = np.nanmean(vals, axis=1)
                    df[f'{t}_knn{k}_median'] = np.nanmedian(vals, axis=1)
                    df[f'{t}_knn{k}_std']    = np.nanstd(vals, axis=1)

                # Distance-weighted mean
                w = 1.0 / (dists * 6371.0 + 1.0)
                valid = ~np.isnan(vals)
                wv = np.where(valid, vals * w, 0.0)
                ws = np.where(valid, w, 0.0)
                df[f'{t}_knn{k}_wmean'] = wv.sum(axis=1) / (ws.sum(axis=1) + 1e-8)

            # Nearest-neighbour distance
            df[f'{t}_nn_dist_km'] = dists_all[:, 0] * 6371.0

        return df


# 1.  MODEL CONFIGS 
def _get_models_for_target(target_name):
    use_log = True  # All targets are right-skewed

    if 'Phosphorus' in target_name:
        xgb_p = dict(
            n_estimators=3000, learning_rate=0.01, max_depth=3,
            min_child_weight=40, subsample=0.6, colsample_bytree=0.3,
            colsample_bylevel=0.5, reg_alpha=5.0, reg_lambda=15.0,
            gamma=2.0, n_jobs=-1, tree_method='hist', random_state=42,
        )
        lgb_p = dict(
            n_estimators=3000, learning_rate=0.01, num_leaves=8,
            max_depth=3, min_child_samples=60, subsample=0.6,
            colsample_bytree=0.3, reg_alpha=5.0, reg_lambda=15.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )
    elif 'Conductance' in target_name:
        xgb_p = dict(
            n_estimators=3000, learning_rate=0.01, max_depth=4,
            min_child_weight=30, subsample=0.65, colsample_bytree=0.35,
            colsample_bylevel=0.5, reg_alpha=3.0, reg_lambda=10.0,
            gamma=1.5, n_jobs=-1, tree_method='hist', random_state=42,
        )
        lgb_p = dict(
            n_estimators=3000, learning_rate=0.01, num_leaves=12,
            max_depth=4, min_child_samples=40, subsample=0.65,
            colsample_bytree=0.35, reg_alpha=3.0, reg_lambda=10.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )
    else:  # Alkalinity
        xgb_p = dict(
            n_estimators=3000, learning_rate=0.01, max_depth=4,
            min_child_weight=25, subsample=0.65, colsample_bytree=0.35,
            colsample_bylevel=0.5, reg_alpha=2.0, reg_lambda=8.0,
            gamma=1.0, n_jobs=-1, tree_method='hist', random_state=42,
        )
        lgb_p = dict(
            n_estimators=3000, learning_rate=0.01, num_leaves=12,
            max_depth=4, min_child_samples=35, subsample=0.65,
            colsample_bytree=0.35, reg_alpha=2.0, reg_lambda=8.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )

    estimators = [('xgb', xgb.XGBRegressor(**xgb_p))]

    if HAS_LGB:
        estimators.append(('lgb', lgb.LGBMRegressor(**lgb_p)))

    if HAS_CB:
        cb_p = dict(
            iterations=2000, learning_rate=0.01, depth=4,
            l2_leaf_reg=15.0, random_seed=42, verbose=0,
            subsample=0.65, colsample_bylevel=0.35,
            min_data_in_leaf=40,
        )
        estimators.append(('cb', CatBoostRegressor(**cb_p)))

    # Extra-Trees
    et_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('et', ExtraTreesRegressor(
            n_estimators=500, max_depth=8, min_samples_leaf=20,
            max_features=0.3, n_jobs=-1, random_state=42,
        )),
    ])
    estimators.append(('et', et_pipe))

    # Ridge regression — completely different inductive bias
    ridge_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0)),
    ])
    estimators.append(('ridge', ridge_pipe))

    return estimators, use_log


# 2.  WEIGHTED ENSEMBLE  
class WeightedEnsemble(BaseEstimator, RegressorMixin):

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y, groups=None):
        X_arr = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.array(y, dtype=np.float64)

        n_splits = 5
        if groups is not None:
            n_unique = len(set(groups))
            n_splits = min(n_splits, n_unique)
            gkf = GroupKFold(n_splits=n_splits)
            split_fn = lambda: gkf.split(X_arr, y_arr, groups=groups)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            split_fn = lambda: kf.split(X_arr, y_arr)

        # Evaluate each model via OOF to get weights
        model_scores = []
        for name, est in self.estimators:
            fold_r2s = []
            for tr, va in split_fn():
                try:
                    m = clone(est)
                    m.fit(X_arr[tr], y_arr[tr])
                    preds = m.predict(X_arr[va])
                    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                    fold_r2s.append(max(r2_score(y_arr[va], preds), 0.0))
                except Exception as e:
                    warnings.warn(f"   {name} failed in fold: {e}")
                    fold_r2s.append(0.0)
            mean_r2 = np.mean(fold_r2s)
            model_scores.append(mean_r2)
            print(f"      {name:>8s}: OOF R²={mean_r2:.4f} "
                  f"(±{np.std(fold_r2s):.3f})")

        # Compute weights (proportional to R2, floor at 0)
        scores = np.array(model_scores)
        scores = np.maximum(scores, 0.0)
        if scores.sum() > 0:
            self.weights_ = scores / scores.sum()
        else:
            self.weights_ = np.ones(len(scores)) / len(scores)

        print(f"      Weights: {dict(zip([n for n,_ in self.estimators], [f'{w:.3f}' for w in self.weights_]))}")

        # Fit all models on full data
        self.fitted_models_ = []
        self.model_names_   = []
        for name, est in self.estimators:
            m = clone(est)
            m.fit(X_arr, y_arr)
            self.fitted_models_.append(m)
            self.model_names_.append(name)

        return self

    def predict(self, X):
        X_arr = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        preds = np.zeros(len(X_arr))
        for m, w in zip(self.fitted_models_, self.weights_):
            p = np.nan_to_num(m.predict(X_arr), nan=0.0, posinf=0.0, neginf=0.0)
            preds += w * p
        return preds


# 3.  EVALUATION  (spatial CV with per-fold KNN)
def evaluate_model(df_full, features, target, groups, estimators, use_log,
                   knn_targets):

    print(f"\n   Evaluating {target} (spatial CV, log={use_log})…")

    y_arr = df_full[target].values.astype(np.float64)
    g_arr = np.array(groups)
    n_groups = len(set(g_arr))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    fold_r2, fold_rmse, fold_mae = [], [], []
    fold_train_r2 = []

    for fi, (tri, tei) in enumerate(gkf.split(np.zeros(len(y_arr)), y_arr, g_arr)):
        df_tr = df_full.iloc[tri].copy()
        df_te = df_full.iloc[tei].copy()
        y_tr  = y_arr[tri]
        y_te  = y_arr[tei]
        g_tr  = g_arr[tri]

        # Refit KNN on training fold only
        fold_knn = SpatialKNNEncoder(knn_targets, k_values=(5, 10, 15, 25))
        fold_knn.fit(df_tr)
        df_tr = fold_knn.transform(df_tr, is_train=True)
        df_te = fold_knn.transform(df_te, is_train=False)

        # Build feature list including KNN columns
        all_feats = [f for f in df_tr.columns
                     if f in features
                     or '_knn' in f or '_nn_dist' in f]
        all_feats = [f for f in all_feats if f not in config.TARGETS]
        all_feats = list(dict.fromkeys(all_feats))

        X_tr = np.nan_to_num(np.array(df_tr[all_feats], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(np.array(df_te[all_feats], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        y_fit = np.log1p(np.clip(y_tr, 0, None)) if use_log else y_tr

        ens = WeightedEnsemble(estimators)
        print(f"\n   ── Fold {fi+1}/{n_splits} "
              f"(train={len(tri)}, test={len(tei)}, "
              f"train_locs={len(set(g_tr))}, "
              f"test_locs={len(set(g_arr[tei]))}) ──")
        ens.fit(X_tr, y_fit, groups=g_tr)

        preds_raw = ens.predict(X_te)
        preds = np.expm1(preds_raw) if use_log else preds_raw
        y_cap = float(np.max(y_tr)) * 2          # safety cap
        preds = np.clip(preds, 0, y_cap)
        preds = np.nan_to_num(preds, nan=0.0, posinf=y_cap, neginf=0.0)

        r2  = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mae  = mean_absolute_error(y_te, preds)
        fold_r2.append(r2); fold_rmse.append(rmse); fold_mae.append(mae)

        # Train diagnostic
        tr_p = ens.predict(X_tr)
        if use_log:
            tr_p = np.expm1(tr_p)
        tr_p = np.nan_to_num(np.clip(tr_p, 0, y_cap), nan=0.0, posinf=y_cap, neginf=0.0)
        tr_r2 = r2_score(y_tr, tr_p)
        fold_train_r2.append(tr_r2)
        gap = tr_r2 - r2
        print(f"      Fold {fi+1}: Train R²={tr_r2:.4f}  Test R²={r2:.4f}  "
              f"Gap={gap:.4f}  RMSE={rmse:.1f}")
        if gap > 0.3:
            print(f"      WARNING: Train-Test gap > 0.3 -> overfitting!")

    mr2 = np.mean(fold_r2)
    mg  = np.mean(fold_train_r2) - mr2
    print(f"\n   CV ({n_splits}-fold):  R²={mr2:.4f} (±{np.std(fold_r2):.3f})  "
          f"RMSE={np.mean(fold_rmse):.1f}  MAE={np.mean(fold_mae):.1f}")
    print(f"   Avg Train R²={np.mean(fold_train_r2):.4f}  Avg Gap={mg:.4f}")
    return mr2, np.mean(fold_rmse)


# 4.  FEATURE IMPORTANCE PRUNING
def prune_features(X, y, features, use_log, min_features=50):
    y_fit = np.log1p(np.clip(y, 0, None)) if use_log else y
    m = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_child_weight=20, subsample=0.7, colsample_bytree=0.5,
        tree_method='hist', random_state=42, n_jobs=-1,
    )
    X_arr = np.array(X, dtype=np.float64)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    m.fit(X_arr, np.array(y_fit, dtype=np.float64))
    imps = m.feature_importances_

    feat_imp = sorted(zip(features, imps), key=lambda x: x[1], reverse=True)
    keep = [f for f, i in feat_imp if i > 0]
    dropped = [f for f, i in feat_imp if i == 0]

    if len(keep) < min_features and len(features) >= min_features:
        need = min_features - len(keep)
        keep.extend(dropped[:need])
        dropped = dropped[need:]
        print(f"   Kept {need} zero-gain features to meet min_features={min_features}")

    if dropped:
        print(f"   Pruned {len(dropped)} zero-gain features: "
              f"{dropped[:10]}{'...' if len(dropped) > 10 else ''}")
    print(f"   Keeping {len(keep)} features (min target: {min_features})")
    return keep


# 4b. CROSS-TARGET FEATURES  (exploit inter-target correlations)
def check_target_correlations(df, targets):
    present = [t for t in targets if t in df.columns]
    if len(present) < 2:
        print("   Only one target present — skipping correlation check.")
        return {}

    valid = df[present].dropna()
    print(f"   Rows with all targets present: {len(valid)}/{len(df)}")
    if len(valid) < 30:
        print("   Too few complete rows — skipping.")
        return {}

    corr_info = {}
    print("   ┌────────────────────────────────────────────────────────────────────┐")
    print("   │                  Target–Target Correlations                        │")
    print("   ├───────────────────────────────┬──────────┬───────────┬─────────────┤")
    print("   │ Pair                          │ Pearson  │ Spearman  │ Exploitable │")
    print("   ├───────────────────────────────┼──────────┼───────────┼─────────────┤")
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            t1, t2 = present[i], present[j]
            pair = valid[[t1, t2]].dropna()
            pearson_r, _ = sp_stats.pearsonr(pair[t1], pair[t2])
            spearman_r, _ = sp_stats.spearmanr(pair[t1], pair[t2])
            exploitable = abs(spearman_r) > 0.15
            flag = " YES" if exploitable else " NO"
            # Truncate names for display
            n1 = t1[:12]
            n2 = t2[:12]
            label = f"{n1}<->{n2}"
            print(f"   │ {label:<25s} │ {pearson_r:>+7.3f}  │ {spearman_r:>+8.3f}  │ {flag:>11s} │")
            corr_info[(t1, t2)] = {
                'pearson': pearson_r, 'spearman': spearman_r,
                'exploitable': exploitable,
            }
    print("   └───────────────────────────────┴──────────┴───────────┴─────────────┘")

    any_exploit = any(v['exploitable'] for v in corr_info.values())
    if any_exploit:
        print("   At least one pair has |Spearman rho| > 0.15 -> enabling cross-target features.")
    else:
        print("   No strong cross-target correlations -> cross-target features would add noise.")
    return corr_info


class CrossTargetFeaturizer:

    def __init__(self, targets, enabled_pairs=None):
        self.targets = targets
        self.enabled_pairs = enabled_pairs
        self.fitted_models_ = {}   # target → fitted model
        self.oof_predictions_ = {} # target → OOF array (indexed like df)

    def _is_useful(self, src_target, dst_target):
        if self.enabled_pairs is None:
            return True
        key = (src_target, dst_target)
        rev_key = (dst_target, src_target)
        info = self.enabled_pairs.get(key) or self.enabled_pairs.get(rev_key)
        return info is not None and info['exploitable']

    def fit_oof(self, df, features, groups):
        df = df.copy()
        self.features_ = list(features)  # Save for test-time alignment
        X = np.nan_to_num(np.array(df[features], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        g = np.array(groups)
        n_groups = len(set(g))
        n_splits = min(5, n_groups)
        if n_splits < 2:
            print("   CrossTarget: too few groups for OOF — skipping.")
            return df

        gkf = GroupKFold(n_splits=n_splits)

        for t in self.targets:
            if t not in df.columns:
                continue

            y = df[t].values.astype(np.float64)
            mask = ~np.isnan(y)
            if mask.sum() < 50:
                continue

            # Lightweight Ridge pipeline for OOF
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=10.0)),
            ])

            # Generate OOF predictions
            oof = np.full(len(df), np.nan)
            for tri, tei in gkf.split(X, y, g):
                # Only fit on rows where target is not NaN
                tri_valid = tri[mask[tri]]
                if len(tri_valid) < 10:
                    continue
                p = clone(pipe)
                p.fit(X[tri_valid], np.log1p(np.clip(y[tri_valid], 0, None)))
                oof[tei] = p.predict(X[tei])

            self.oof_predictions_[t] = oof

            # Fit final model on all data for test-time predictions
            valid_idx = np.where(mask)[0]
            final_pipe = clone(pipe)
            final_pipe.fit(X[valid_idx], np.log1p(np.clip(y[valid_idx], 0, None)))
            self.fitted_models_[t] = final_pipe

        # Add cross-target columns
        added = 0
        for dst_target in self.targets:
            for src_target in self.targets:
                if src_target == dst_target:
                    continue
                if not self._is_useful(src_target, dst_target):
                    continue
                col_name = f'xt_{src_target[:8].replace(" ", "")}'
                if src_target in self.oof_predictions_:
                    df[col_name] = self.oof_predictions_[src_target]
                    added += 1

        # De-duplicate column names (some might appear more than once)
        added_cols = [c for c in df.columns if c.startswith('xt_')]
        added_cols = list(dict.fromkeys(added_cols))
        print(f"   CrossTarget: added {len(added_cols)} OOF feature columns: {added_cols}")
        return df

    def transform_test(self, test_df, features):
        test_df = test_df.copy()

        aligned_feats = self.features_ if hasattr(self, 'features_') else features
        for f in aligned_feats:
            if f not in test_df.columns:
                test_df[f] = np.nan
        X = np.nan_to_num(np.array(test_df[aligned_feats], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        for t in self.targets:
            if t not in self.fitted_models_:
                continue
            col_name = f'xt_{t[:8].replace(" ", "")}'
            test_df[col_name] = self.fitted_models_[t].predict(X)

        return test_df

    def get_feature_names(self):
        names = []
        for t in self.targets:
            if t in self.fitted_models_:
                names.append(f'xt_{t[:8].replace(" ", "")}')
        return list(dict.fromkeys(names))


# 5.  OPTUNA SEARCH  (conservative search space)
def _optuna_search(X, y, groups, use_log, target_name, n_trials=30):
    if not HAS_OPTUNA:
        print("   Optuna not installed – using default hyperparams.")
        return None

    print(f"   Running Optuna ({n_trials} trials, conservative bounds)…")
    y_fit = np.log1p(np.clip(y, 0, None)) if use_log else y
    X_arr = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.array(y_fit, dtype=np.float64)
    g_arr = np.array(groups)

    n_groups = len(set(g_arr))
    n_splits = min(4, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial):
        params = {
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('lr', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'min_child_weight': trial.suggest_int('mcw', 20, 60),
            'subsample': trial.suggest_float('subsample', 0.5, 0.7),
            'colsample_bytree': trial.suggest_float('colsample', 0.25, 0.5),
            'reg_alpha': trial.suggest_float('alpha', 1.0, 20.0, log=True),
            'reg_lambda': trial.suggest_float('lambda', 2.0, 30.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.5, 5.0),
            'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42,
        }
        scores = []
        for tr, va in gkf.split(X_arr, y_arr, g_arr):
            m = xgb.XGBRegressor(**params)
            m.fit(X_arr[tr], y_arr[tr],
                  eval_set=[(X_arr[va], y_arr[va])], verbose=False)
            preds = m.predict(X_arr[va])
            if use_log:
                preds_real = np.expm1(preds)
                y_real = np.expm1(y_arr[va])
            else:
                preds_real, y_real = preds, y_arr[va]
            y_cap_opt = float(np.max(y_real)) * 2
            preds_real = np.nan_to_num(np.clip(preds_real, 0, y_cap_opt),
                                       nan=0.0, posinf=y_cap_opt, neginf=0.0)
            scores.append(r2_score(y_real, preds_real))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    # Rename back from short names
    best['learning_rate'] = best.pop('lr')
    best['min_child_weight'] = best.pop('mcw')
    best['colsample_bytree'] = best.pop('colsample')
    best['reg_alpha'] = best.pop('alpha')
    best['reg_lambda'] = best.pop('lambda')
    best['n_estimators'] = 3000
    print(f"   Best Optuna R²: {study.best_value:.4f}")
    print(f"   Best params: {best}")
    return best


# 6.  MAIN ENTRY POINT
def train_models(df, log_transform=None, use_optuna=True, dws_context=None):
    print("\n== Feature Engineering ==")
    df = engineer_features(df)

    if dws_context is not None:
        import dws_data as dws_mod

        all_dws          = dws_context['all_dws']
        station_features = dws_context['station_features']
        aug_rows         = dws_context['aug_rows']

        print("\n== DWS Integration ==")

        # 1. Add augmented training rows from DWS
        if aug_rows is not None and len(aug_rows) > 0:
            aug = aug_rows.copy()
            aug = engineer_features(aug)
            pre_len = len(df)
            df = pd.concat([df, aug], ignore_index=True)
            print(f"   Augmented training: {pre_len} -> {len(df)} rows "
                  f"(+{len(aug)} DWS rows at test locations)")

        # 2. Assign DWS station codes
        df['_dws_station'] = df.apply(
            lambda r: dws_mod.coord_to_station(r['Latitude'], r['Longitude']),
            axis=1,
        )
        n_matched = df['_dws_station'].notna().sum()
        print(f"   DWS station match: {n_matched}/{len(df)} rows")

        # 3. Merge station-level features (IDW for non-matched rows)
        df = dws_mod.merge_station_features(df, station_features)

        # 4. Add per-row lag features
        df = dws_mod.add_lag_features(df, all_dws, config.TARGETS)

        # 5. Add same-day auxiliary features to ALL training rows
        df = dws_mod.add_sameday_aux_features(df, all_dws)

    print("\n== Cross-Target Correlation Analysis ==")
    corr_info = check_target_correlations(df, config.TARGETS)
    use_cross_target = any(v['exploitable'] for v in corr_info.values()) if corr_info else False

    # KNN 
    print("\n== KNN Spatial Features ==")
    knn_enc = SpatialKNNEncoder(config.TARGETS, k_values=(5, 10, 15, 25))
    knn_enc.fit(df)
    df_with_knn = knn_enc.transform(df, is_train=True)
    knn_cols = [c for c in df_with_knn.columns
                if '_knn' in c or '_nn_dist' in c]
    print(f"   Added {len(knn_cols)} KNN features")

    # Feature selection
    always_drop = ['Sample Date', 'key', '_dt', '_loc_id', '_enc_loc_id',
                   '_is_missing', '_geo_key', 'Latitude', 'Longitude',
                   '_dws_station', 'date', 'station']
    dead_feats = getattr(config, 'DEAD_FEATURES', [])
    drop_cols = config.TARGETS + always_drop + dead_feats
    base_features = [c for c in df.columns
                     if c not in drop_cols
                     and pd.api.types.is_numeric_dtype(df[c])]
    print(f"   Base features ({len(base_features)}): {base_features[:15]}...")

    # ── Cross-target OOF features ────────────────────────────────────────
    xt_featurizer = None
    if use_cross_target:
        print("\n══ Cross-Target OOF Features ══")
        # Use full df with all rows to generate OOF cross-target predictions
        groups_all = (df['Latitude'].round(2).astype(str) + "_" +
                      df['Longitude'].round(2).astype(str))
        xt_featurizer = CrossTargetFeaturizer(
            config.TARGETS, enabled_pairs=corr_info)
        df = xt_featurizer.fit_oof(df, base_features, groups_all)
        xt_cols = xt_featurizer.get_feature_names()
    else:
        xt_cols = []
        print("\n   Cross-target features: DISABLED (correlations too weak)")

    performance_report = {
        '_knn_encoder': knn_enc,
        '_xt_featurizer': xt_featurizer,
        '_base_features': base_features,
        '_dws_context': dws_context,
    }

    for target in config.TARGETS:
        if target not in df.columns:
            continue

        print(f"\n{'='*65}")
        print(f"   TARGET: {target}")
        print(f"{'='*65}")

        estimators, auto_log = _get_models_for_target(target)
        use_log = auto_log if log_transform is None else log_transform

        tdf = df.dropna(subset=[target]).copy()
        groups = (tdf['Latitude'].round(2).astype(str) + "_" +
                  tdf['Longitude'].round(2).astype(str))

        y = tdf[target].copy()

        # Winsorise at 1st/99th percentile
        lo, hi = y.quantile(0.01), y.quantile(0.99)
        y = y.clip(lower=lo, upper=hi)
        tdf[target] = y

        keep = ~tdf[base_features].isna().all(axis=1)
        tdf, y, groups = tdf[keep], y[keep], groups[keep]

        print(f"   Samples: {len(tdf)},  Locations: {groups.nunique()}")
        print(f"   y: mean={y.mean():.2f}  std={y.std():.2f}  "
              f"skew={y.skew():.2f}  log={use_log}")
        print(f"   y range: [{y.min():.1f}, {y.max():.1f}] (winsorised)")

        # Cross-target features relevant for THIS target
        xt_feats_for_target = [c for c in xt_cols
                               if c in tdf.columns
                               and c != f'xt_{target[:8].replace(" ", "")}']
        if xt_feats_for_target:
            print(f"   Cross-target features for this target: {xt_feats_for_target}")

        # Prune
        X_base = tdf[base_features]
        tgt_base_feats = prune_features(X_base, y, base_features, use_log)
        tgt_base_feats = tgt_base_feats + xt_feats_for_target  # always include xt features
        print(f"   Features after pruning + cross-target: {len(tgt_base_feats)}")

        if use_optuna and HAS_OPTUNA:
            tdf_knn = knn_enc.transform(tdf, is_train=True)
            optuna_feats = [f for f in tdf_knn.columns
                           if (f in tgt_base_feats
                               or '_knn' in f or '_nn_dist' in f)
                           and f not in config.TARGETS]
            optuna_feats = list(dict.fromkeys(optuna_feats))
            best_xgb_params = _optuna_search(
                tdf_knn[optuna_feats], y, groups, use_log, target,
                n_trials=30)
            if best_xgb_params is not None:
                best_xgb_params['tree_method'] = 'hist'
                best_xgb_params['n_jobs'] = -1
                best_xgb_params['random_state'] = 42
                estimators = [(n, e) if n != 'xgb' else
                              ('xgb', xgb.XGBRegressor(**best_xgb_params))
                              for n, e in estimators]

        r2, rmse = evaluate_model(tdf, tgt_base_feats, target, groups,
                                   estimators, use_log, config.TARGETS)

        print("   Training final model on ALL data…")
        tdf_final = knn_enc.transform(tdf, is_train=True)
        all_feats = [f for f in tdf_final.columns
                     if (f in tgt_base_feats
                         or '_knn' in f or '_nn_dist' in f)
                     and f not in config.TARGETS]
        all_feats = list(dict.fromkeys(all_feats))

        y_fit = (np.log1p(np.clip(y, 0, None)) if use_log
                 else np.array(y, dtype=np.float64))
        final = WeightedEnsemble(estimators)
        final.fit(np.nan_to_num(np.array(tdf_final[all_feats], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
                  np.array(y_fit, dtype=np.float64),
                  groups=np.array(groups))

        performance_report[target] = {
            'R2': r2, 'RMSE': rmse,
            'model': final,
            'features': all_feats,
            'log_transform': use_log,
            'global_mean': float(y.mean()),
        }

        safe = target.replace(' ', '_')
        joblib.dump(final, f"model_{safe}.joblib")
        print(f"   Saved model_{safe}.joblib")

    joblib.dump(knn_enc, "knn_encoder.joblib")
    if xt_featurizer is not None:
        joblib.dump(xt_featurizer, "cross_target_featurizer.joblib")
        print("   Saved cross_target_featurizer.joblib")

    sarimax_fitted = None
    if HAS_SARIMAX and dws_context is not None:
        all_dws = dws_context.get('all_dws')
        if all_dws and len(all_dws) > 0:
            try:
                sarimax_fitted = StationSARIMAX(
                    config.TARGETS, n_optuna_trials=20)
                sarimax_fitted.fit(all_dws)
                sarimax_cv = sarimax_fitted.evaluate_cv(all_dws)
                joblib.dump(sarimax_fitted, "sarimax_models.joblib")
                print("   Saved sarimax_models.joblib")
            except Exception as e:
                print(f"   Warning: SARIMAX fitting failed (non-fatal): {e}")
                traceback.print_exc()
                sarimax_fitted = None

    performance_report['_sarimax'] = sarimax_fitted

    print(f"\n{'='*65}")
    print("   FINAL CV RESULTS")
    print(f"{'='*65}")
    for t in config.TARGETS:
        if t in performance_report:
            m = performance_report[t]
            sarimax_tag = ""
            if sarimax_fitted is not None and hasattr(sarimax_fitted, 'models_'):
                n_sarimax = sum(1 for k in sarimax_fitted.models_ if k[1] == t)
                sarimax_tag = f"  SARIMAX={n_sarimax}stn"
            print(f"   {t:>35s}:  R2={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  "
                  f"log={m['log_transform']}{sarimax_tag}")

    return performance_report
