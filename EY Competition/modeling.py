import numpy as np
import os
import pandas as pd
import traceback
import xgboost as xgb
import warnings

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    warnings.warn("LightGBM not installed - ensemble will skip it.")
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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNetCV, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import BallTree
from scipy import stats as sp_stats
import joblib
import config

# Multi-target chaining order 
# EC first (strongest aux correlations), then TAL, then DRP (weakest, benefits
# from EC/TAL predictions as features).
CHAIN_ORDER = [
    'Electrical Conductance',
    'Total Alkalinity',
    'Dissolved Reactive Phosphorus',
]

# FEATURE ENGINEERING 
def engineer_features(df):
    df = df.copy()
    eps = 1e-8

    #  Temporal 
    if 'Sample Date' in df.columns:
        dt = pd.to_datetime(df['Sample Date'], dayfirst=True)
        df['Month']     = dt.dt.month
        df['DayOfYear'] = dt.dt.dayofyear
        df['Year']      = dt.dt.year
        df['Month_sin']     = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos']     = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

    #  Spectral indices 
    if 'nir08' in df.columns and 'red' in df.columns:
        df['NDVI'] = (df['nir08'] - df['red']) / (df['nir08'] + df['red'] + eps)
    if 'green' in df.columns and 'nir08' in df.columns:
        df['NDWI'] = (df['green'] - df['nir08']) / (df['green'] + df['nir08'] + eps)
    if 'nir08' in df.columns and 'swir16' in df.columns:
        df['NDMI_calc'] = (df['nir08'] - df['swir16']) / (df['nir08'] + df['swir16'] + eps)
    if 'red' in df.columns and 'green' in df.columns:
        df['Turbidity_proxy'] = df['red'] / (df['green'] + eps)
    if 'nir08' in df.columns and 'red' in df.columns:
        df['Chl_proxy'] = df['nir08'] / (df['red'] + eps)
    if all(c in df.columns for c in ['swir16', 'red', 'nir08', 'green']):
        df['BSI'] = (((df['swir16'] + df['red']) - (df['nir08'] + df['green'])) /
                     ((df['swir16'] + df['red']) + (df['nir08'] + df['green']) + eps))

    #  Elevation / slope 
    if 'elevation_mean' in df.columns:
        df['log_Elev'] = np.log1p(np.clip(df['elevation_mean'], 0, None))
        if 'slope_mean' in df.columns:
            df['Elev_Slope'] = df['elevation_mean'] * df['slope_mean']

    #  Rainfall 
    if 'rain_7d_sum' in df.columns and 'rain_30d_sum' in df.columns:
        df['Rain_7d_frac'] = df['rain_7d_sum'] / (df['rain_30d_sum'] + eps)
    if 'elevation_mean' in df.columns and 'rain_30d_sum' in df.columns:
        df['Elev_Rain'] = df['elevation_mean'] * df['rain_30d_sum']

    #  Land cover interactions 
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

    #  Geology 
    if 'geo_is_karst' in df.columns and 'rain_30d_sum' in df.columns:
        df['Karst_Rain'] = df['geo_is_karst'] * df['rain_30d_sum']

    #  Dam proximity 
    if 'nearest_dam_dist_m' in df.columns:
        df['log_Dam_dist'] = np.log1p(np.clip(df['nearest_dam_dist_m'], 0, None))

    return df


# KNN SPATIAL FEATURES (chemistry-based, no target leakage)
_KNN_AUX_COLS = [
    'dws_aux_pH_Diss_Water', 'dws_aux_Ca_Diss_Water',
    'dws_aux_Mg_Diss_Water', 'dws_aux_Na_Diss_Water',
    'dws_aux_Cl_Diss_Water', 'dws_aux_SO4_Diss_Water',
    'dws_aux_F_Diss_Water',  'dws_aux_Si_Diss_Water',
]

class SpatialKNNEncoder:
    def __init__(self, targets, k_values=(5, 10)):
        self.targets  = targets
        self.k_values = k_values
        self.tree_    = None

    def fit(self, df):
        coords = np.deg2rad(df[['Latitude', 'Longitude']].values)
        self.tree_ = BallTree(coords, metric='haversine')
        self.train_coords_ = coords
        self.train_aux_ = {}
        for col in _KNN_AUX_COLS:
            if col in df.columns:
                self.train_aux_[col] = df[col].values.copy()
        return self

    def transform(self, df, is_train=False):
        df = df.copy()
        coords = np.deg2rad(df[['Latitude', 'Longitude']].values)
        max_k = max(self.k_values)
        k_q = max_k + 1 if is_train else max_k
        k_q = min(k_q, self.tree_.data.shape[0])
        if k_q < 2:
            return df
        dists_all, inds_all = self.tree_.query(coords, k=k_q)
        if is_train:
            inds_all  = inds_all[:, 1:]
            dists_all = dists_all[:, 1:]
        for col in self.train_aux_:
            short = col.replace('dws_aux_', '').replace('_Diss_Water', '')\
                       .replace('_Tot_Water', '')
            y_all = self.train_aux_[col]
            for k in self.k_values:
                if k > inds_all.shape[1]:
                    continue
                inds = inds_all[:, :k]
                dists = dists_all[:, :k]
                vals = y_all[inds]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[f'knn{k}_{short}_mean'] = np.nanmean(vals, axis=1)
                # IDW weighted mean
                w = 1.0 / (dists * 6371.0 + 1.0)
                valid = ~np.isnan(vals)
                wv = np.where(valid, vals * w, 0.0)
                ws = np.where(valid, w, 0.0)
                df[f'knn{k}_{short}_wmean'] = wv.sum(axis=1) / (ws.sum(axis=1) + 1e-8)

        # Nearest-neighbour distance (useful regardless)
        df['nn_dist_km'] = dists_all[:, 0] * 6371.0
        return df


# STATION LSTM - temporal model for per-station time series
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

class _LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, lengths):
        # x: (B, T, F), lengths: (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out = self.head(h_n[-1])  # last layer hidden state
        return out.squeeze(-1)


# Sequence features: DWS columns that are available at each timestep
_LSTM_AUX_COLS = [
    'Ca_Diss_Water', 'Cl_Diss_Water', 'DMS_Tot_Water',
    'F_Diss_Water', 'K_Diss_Water', 'Mg_Diss_Water',
    'Na_Diss_Water', 'NH4_N_Diss_Water', 'NO3_NO2_N_Diss_Water',
    'P_Tot_Water', 'pH_Diss_Water', 'Si_Diss_Water', 'SO4_Diss_Water',
]


class StationLSTMModel:
    """
    Per-station LSTM that learns temporal dynamics from DWS time series.

    For each prediction row at station S on date D:
      1. Retrieve the DWS time series at S before D
      2. Build a sequence of [target, aux_chemistry, month_sin, month_cos]
      3. Feed through LSTM → predict target value

    This captures temporal dynamics (trends, seasonality, regime changes)
    without memorising station identity.
    """

    def __init__(self, target_dws_col, target_name, use_log=True,
                 seq_len=24, hidden=64, n_layers=2, epochs=30,
                 lr=1e-3, batch_size=256):
        self.target_dws_col = target_dws_col
        self.target_name = target_name
        self.use_log = use_log
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model_ = None
        self.feat_means_ = None
        self.feat_stds_ = None
        self.n_features_ = None

    def _build_sequences(self, all_dws, rows_df, exclude_dates=None):
        """Build (sequence, target, length) triples for each row.

        rows_df must have '_station', 'date' (or 'Sample Date') columns.
        For training rows: use the DWS time series BEFORE each row's date.
        """
        import dws_data as dws_mod
        DWS_UNIT_CONVERSIONS = dws_mod.DWS_UNIT_CONVERSIONS

        if exclude_dates is None:
            exclude_dates = {}

        # Parse dates (handle mixed formats: dd-mm-yyyy strings + ISO timestamps)
        if 'date' not in rows_df.columns or rows_df['date'].dtype == object:
            if 'Sample Date' in rows_df.columns:
                dates = pd.to_datetime(
                    rows_df['Sample Date'], format='mixed', dayfirst=True)
            else:
                return [], [], []
        else:
            dates = pd.to_datetime(rows_df['date'])

        stations = rows_df['_station'].values
        conversion = DWS_UNIT_CONVERSIONS.get(self.target_dws_col, 1.0)

        # Feature columns: target + aux + temporal
        feat_cols = [self.target_dws_col] + _LSTM_AUX_COLS

        sequences, targets, lengths = [], [], []

        for i in range(len(rows_df)):
            stn = stations[i]
            dt = dates.iloc[i] if hasattr(dates, 'iloc') else dates[i]
            if pd.isna(stn) or pd.isna(dt) or stn not in all_dws:
                sequences.append(None)
                targets.append(np.nan)
                lengths.append(0)
                continue

            sdf = all_dws[stn]
            stn_excluded = exclude_dates.get(stn, set())

            # Get history before this date
            mask = sdf['date'] < dt
            if stn_excluded:
                mask = mask & ~sdf['date'].dt.date.isin(stn_excluded)
            hist = sdf[mask].sort_values('date').tail(self.seq_len)

            if len(hist) < 2:
                sequences.append(None)
                targets.append(np.nan)
                lengths.append(0)
                continue

            # Build feature matrix: [target, aux, month_sin, month_cos]
            row_feats = []
            for col in feat_cols:
                if col in hist.columns:
                    v = pd.to_numeric(hist[col], errors='coerce').values
                else:
                    v = np.full(len(hist), np.nan)
                row_feats.append(v)

            # Temporal features
            m = hist['date'].dt.month.values
            row_feats.append(np.sin(2 * np.pi * m / 12))
            row_feats.append(np.cos(2 * np.pi * m / 12))
            doy = hist['date'].dt.dayofyear.values
            row_feats.append(np.sin(2 * np.pi * doy / 365.25))
            row_feats.append(np.cos(2 * np.pi * doy / 365.25))

            seq = np.column_stack(row_feats)  # (T, F)
            sequences.append(seq)
            lengths.append(len(hist))

            # Target: the actual target value for THIS row
            idx = rows_df.index[i]
            if self.target_name in rows_df.columns:
                targets.append(rows_df.at[idx, self.target_name])
            else:
                targets.append(np.nan)

        return sequences, targets, lengths

    def _get_row_dates(self, rows_df):
        """Extract parsed dates for each row (for temporal splitting)."""
        if 'date' not in rows_df.columns or rows_df['date'].dtype == object:
            if 'Sample Date' in rows_df.columns:
                return pd.to_datetime(
                    rows_df['Sample Date'], format='mixed',
                    dayfirst=True).values
        else:
            return pd.to_datetime(rows_df['date']).values
        return None

    def _normalise_sequences(self, sequences, fit=False):
        """Z-score normalise each feature across all timesteps."""
        # Collect all valid values to compute mean/std
        if fit:
            all_vals = []
            for seq in sequences:
                if seq is not None:
                    all_vals.append(seq)
            if not all_vals:
                return sequences
            big = np.concatenate(all_vals, axis=0)
            self.feat_means_ = np.nanmean(big, axis=0)
            self.feat_stds_ = np.nanstd(big, axis=0)
            self.feat_stds_[self.feat_stds_ < 1e-8] = 1.0

        normed = []
        for seq in sequences:
            if seq is None:
                normed.append(None)
            else:
                s = (seq - self.feat_means_) / self.feat_stds_
                s = np.nan_to_num(s, nan=0.0)
                normed.append(s)
        return normed

    def _pad_sequences(self, sequences, lengths):
        """Pad sequences to fixed length, return tensor + lengths."""
        n_feat = self.n_features_
        padded = np.zeros((len(sequences), self.seq_len, n_feat), dtype=np.float32)
        valid_lengths = np.zeros(len(sequences), dtype=np.int64)

        for i, (seq, l) in enumerate(zip(sequences, lengths)):
            if seq is not None and l > 0:
                T = min(l, self.seq_len)
                padded[i, :T, :] = seq[-T:]
                valid_lengths[i] = T
            else:
                valid_lengths[i] = 1  # at least 1 to avoid pack_padded error
        return torch.tensor(padded), torch.tensor(valid_lengths)

    def fit(self, all_dws, rows_df, exclude_dates=None):
        """Train the LSTM on DWS station time series."""
        print(f"      LSTM: building sequences for {self.target_name}...")
        sequences, targets, lengths = self._build_sequences(
            all_dws, rows_df, exclude_dates)

        # Filter to valid rows
        valid_mask = [(seq is not None and l >= 2 and np.isfinite(t))
                      for seq, t, l in zip(sequences, targets, lengths)]
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        sequences = [sequences[i] for i in valid_indices]
        targets = [targets[i] for i in valid_indices]
        lengths = [lengths[i] for i in valid_indices]

        if len(sequences) < 100:
            print(f"      LSTM: only {len(sequences)} valid sequences, skipping")
            return self

        self.n_features_ = sequences[0].shape[1]
        sequences = self._normalise_sequences(sequences, fit=True)

        y = np.array(targets, dtype=np.float32)
        if self.use_log:
            y = np.log1p(np.clip(y, 0, None))

        X_pad, L = self._pad_sequences(sequences, lengths)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Train/val split 
        n = len(y)
        row_dates = self._get_row_dates(rows_df)
        if row_dates is not None:
            valid_dates = row_dates[np.array(valid_indices)]
            date_order = np.argsort(valid_dates)
            n_train = int(0.8 * n)
            tr_idx = date_order[:n_train]
            va_idx = date_order[n_train:]
        else:
            # Fallback to random if no date info
            perm = np.random.RandomState(42).permutation(n)
            n_train = int(0.8 * n)
            tr_idx, va_idx = perm[:n_train], perm[n_train:]

        self.model_ = _LSTMNet(self.n_features_, self.hidden,
                                self.n_layers, dropout=0.2)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        print(f"      LSTM: training on {n_train} sequences "
              f"(val={n - n_train}), {self.n_features_} features, "
              f"seq_len={self.seq_len}...")

        for epoch in range(self.epochs):
            self.model_.train()
            # Shuffle training indices
            np.random.shuffle(tr_idx)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(tr_idx), self.batch_size):
                batch_idx = tr_idx[start:start + self.batch_size]
                xb = X_pad[batch_idx]
                lb = L[batch_idx]
                yb = y_t[batch_idx]

                pred = self.model_(xb, lb)
                loss = nn.functional.huber_loss(pred, yb, delta=1.0)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            self.model_.eval()
            with torch.no_grad():
                va_pred = self.model_(X_pad[va_idx], L[va_idx])
                va_loss = nn.functional.mse_loss(va_pred, y_t[va_idx]).item()
            scheduler.step(va_loss)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state = {k: v.clone() for k, v in
                              self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                print(f"      LSTM: early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        with torch.no_grad():
            va_pred = self.model_(X_pad[va_idx], L[va_idx]).numpy()
            va_true = y[va_idx]
            if self.use_log:
                va_pred_real = np.expm1(va_pred)
                va_true_real = np.expm1(va_true)
            else:
                va_pred_real, va_true_real = va_pred, va_true
            va_pred_real = np.clip(va_pred_real, 0, None)
            va_r2 = r2_score(va_true_real, va_pred_real)
        print(f"      LSTM: val R2={va_r2:.4f} (best_loss={best_val_loss:.4f})")
        return self

    def predict(self, all_dws, rows_df, exclude_dates=None):
        """Generate LSTM predictions for each row."""
        if self.model_ is None or self.n_features_ is None:
            return np.full(len(rows_df), np.nan)

        sequences, _, lengths = self._build_sequences(
            all_dws, rows_df, exclude_dates)

        # Handle rows with no valid sequence
        valid_mask = [(seq is not None and l >= 2)
                      for seq, l in zip(sequences, lengths)]

        if not any(valid_mask):
            return np.full(len(rows_df), np.nan)

        # Normalise and pad valid sequences only
        valid_seqs = [s for s, v in zip(sequences, valid_mask) if v]
        valid_lens = [l for l, v in zip(lengths, valid_mask) if v]
        valid_seqs = self._normalise_sequences(valid_seqs, fit=False)

        X_pad, L = self._pad_sequences(valid_seqs, valid_lens)

        self.model_.eval()
        preds = np.full(len(rows_df), np.nan)
        with torch.no_grad():
            # Batch predict
            all_pred = []
            for start in range(0, len(valid_seqs), self.batch_size):
                end = min(start + self.batch_size, len(valid_seqs))
                p = self.model_(X_pad[start:end], L[start:end]).numpy()
                all_pred.append(p)
            all_pred = np.concatenate(all_pred)

            if self.use_log:
                all_pred = np.expm1(all_pred)
            all_pred = np.clip(all_pred, 0, None)

        # Map back to original indices
        vi = 0
        for i, v in enumerate(valid_mask):
            if v:
                preds[i] = all_pred[vi]
                vi += 1

        return preds



# DWS FEATURE BUILDER
def build_dws_training_set(all_dws, targets, exclude_dates=None):
    import dws_data as dws_mod

    DWS_COL_MAP = dws_mod.DWS_COL_MAP
    DWS_AUX_COLS = dws_mod.DWS_AUX_COLS
    full_registry = dws_mod._FULL_REGISTRY or dws_mod.STATION_REGISTRY

    if exclude_dates is None:
        exclude_dates = {}

    station_frames = []
    n_excluded = 0

    for stn, sdf in all_dws.items():
        if stn not in full_registry:
            continue
        lat, lon = full_registry[stn][0], full_registry[stn][1]
        stn_excluded = exclude_dates.get(stn, set())

        sf = sdf.sort_values('date').reset_index(drop=True).copy()

        #  Exclude test dates 
        if stn_excluded:
            keep = ~sf['date'].dt.date.isin(stn_excluded)
            n_excluded += (~keep).sum()
            sf = sf[keep].reset_index(drop=True)

        if len(sf) == 0:
            continue

        #  Targets 
        for dws_col, target in DWS_COL_MAP.items():
            sf[target] = pd.to_numeric(sf.get(dws_col), errors='coerce')
        # Keep rows where at least one target is non-NaN
        tgt_cols = list(DWS_COL_MAP.values())
        sf = sf[sf[tgt_cols].notna().any(axis=1)].reset_index(drop=True)
        if len(sf) == 0:
            continue

        #  Metadata 
        sf['Latitude'] = lat
        sf['Longitude'] = lon
        sf['Sample Date'] = sf['date'].dt.strftime('%d-%m-%Y')
        sf['_station'] = stn
        sf['_is_dws'] = 1
        sf['_has_dws_aux'] = 1
        sf['_has_dws_lag'] = 1
        sf['_has_stn_hist'] = 1
        sf['_has_enrichment'] = 0  # set to 1 later by _merge_enrichment_to_dws

        #  Same-day auxiliary chemistry 
        for aux in DWS_AUX_COLS:
            sf[f'dws_aux_{aux}'] = pd.to_numeric(sf.get(aux), errors='coerce')

        #  Temporal 
        dt = sf['date']
        sf['Month'] = dt.dt.month
        sf['Month_sin'] = np.sin(2 * np.pi * sf['Month'] / 12)
        sf['Month_cos'] = np.cos(2 * np.pi * sf['Month'] / 12)
        sf['DayOfYear'] = dt.dt.dayofyear
        sf['DayOfYear_sin'] = np.sin(2 * np.pi * sf['DayOfYear'] / 365.25)
        sf['DayOfYear_cos'] = np.cos(2 * np.pi * sf['DayOfYear'] / 365.25)
        sf['Year'] = dt.dt.year

        #  Lag features (vectorised via shift / rolling) 
        for dws_col, target in DWS_COL_MAP.items():
            pfx = target[:3].upper()
            vals = pd.to_numeric(sf.get(dws_col), errors='coerce')

            # Forward-fill to get "last non-NaN before me"
            last_valid = vals.ffill().shift(1)  # shift(1) = previous row
            sf[f'lag_{pfx}_val'] = last_valid

            # Days since last measurement: date - last_date_with_value
            valid_date = sf['date'].where(vals.notna()).ffill().shift(1)
            sf[f'lag_{pfx}_days'] = (sf['date'] - valid_date).dt.days

            # Expanding-window rolling: use only past non-NaN values
            # rolling(min_periods=1).mean() on an expanding window
            exp = vals.expanding(min_periods=1)
            # For roll-N we need the last N non-NaN values.
            # We use a trick: build a series of only non-NaN values,
            # compute rolling, then map back.
            non_nan_vals = vals.dropna()
            roll3_src  = non_nan_vals.rolling(3,  min_periods=1).mean()
            roll5_src  = non_nan_vals.rolling(5,  min_periods=1).mean()
            roll10_src = non_nan_vals.rolling(10, min_periods=1).mean()

            # Map back: for each row, find the rolling mean of the
            # last K non-NaN values BEFORE this row.
            # "Before" = shift the non-NaN series by 1, then reindex.
            roll3_shifted  = roll3_src.shift(1).reindex(sf.index)
            roll5_shifted  = roll5_src.shift(1).reindex(sf.index)
            roll10_shifted = roll10_src.shift(1).reindex(sf.index)

            # Forward-fill to fill NaN rows between valid measurements
            sf[f'roll3_{pfx}']  = roll3_shifted.ffill()
            sf[f'roll5_{pfx}']  = roll5_shifted.ffill()
            sf[f'roll10_{pfx}'] = roll10_shifted.ffill()

            # First row has no history → NaN (shift already handles this)

        #  Lag for key auxiliary chemistry (vectorised) 
        for aux in ['pH_Diss_Water', 'Ca_Diss_Water', 'Mg_Diss_Water',
                    'Na_Diss_Water', 'Cl_Diss_Water', 'SO4_Diss_Water']:
            if aux in sf.columns:
                aux_vals = pd.to_numeric(sf[aux], errors='coerce')
                sf[f'lag_aux_{aux}'] = aux_vals.ffill().shift(1)
            else:
                sf[f'lag_aux_{aux}'] = np.nan

        station_frames.append(sf)

    #  Assemble 
    # Select only the columns we need (drop raw DWS columns)
    keep_prefixes = ('Latitude', 'Longitude', 'Sample Date', '_station',
                     '_is_dws', '_has_dws', '_has_stn', '_has_enrichment',
                     'Total', 'Electrical', 'Dissolved',
                     'dws_aux_', 'Month', 'DayOfYear', 'Year',
                     'lag_', 'roll3_', 'roll5_', 'roll10_',
                     'Month_sin', 'Month_cos', 'DayOfYear_sin',
                     'DayOfYear_cos')

    all_cols = set()
    for sf in station_frames:
        for c in sf.columns:
            if any(c.startswith(p) or c == p for p in keep_prefixes):
                all_cols.add(c)
    all_cols = sorted(all_cols)

    dws_df = pd.concat(
        [sf[[c for c in all_cols if c in sf.columns]] for sf in station_frames],
        ignore_index=True,
    )
    print(f"   Built DWS training set: {len(dws_df)} rows from "
          f"{dws_df['_station'].nunique()} stations"
          f" ({n_excluded} test-date rows excluded)")

    return dws_df


def add_station_historical_features(df, all_dws):
    import dws_data as dws_mod
    full_registry = dws_mod._FULL_REGISTRY or dws_mod.STATION_REGISTRY

    stn_stats = {}
    for stn, sdf in all_dws.items():
        if stn not in full_registry:
            continue
        feats = {}

        # Auxiliary chemistry stats ONLY (no target stats!)
        for aux in ['pH_Diss_Water', 'Ca_Diss_Water', 'Mg_Diss_Water',
                    'Na_Diss_Water', 'Cl_Diss_Water', 'SO4_Diss_Water',
                    'F_Diss_Water', 'Si_Diss_Water', 'K_Diss_Water',
                    'NH4_N_Diss_Water', 'NO3_NO2_N_Diss_Water',
                    'DMS_Tot_Water']:
            vals = pd.to_numeric(
                sdf.get(aux, pd.Series(dtype=float)), errors='coerce'
            ).dropna()
            short = aux.split('_')[0]
            if len(vals) >= 3:
                feats[f'stn_aux_{short}_mean'] = vals.mean()
                feats[f'stn_aux_{short}_std'] = vals.std()
                feats[f'stn_aux_{short}_median'] = vals.median()
                feats[f'stn_aux_{short}_cv'] = vals.std() / (vals.mean() + 1e-8)
                # Seasonal amplitude
                dates = sdf.loc[vals.index, 'date']
                ms = pd.Series(vals.values, index=pd.DatetimeIndex(dates.values))
                monthly = ms.groupby(ms.index.month).mean()
                feats[f'stn_aux_{short}_season'] = monthly.max() - monthly.min()
            else:
                feats[f'stn_aux_{short}_mean'] = vals.mean() if len(vals) else np.nan
                feats[f'stn_aux_{short}_std'] = np.nan

        # Station metadata: number of historical observations (useful proxy
        # for data quality, not station identity)
        feats['stn_n_obs'] = len(sdf)
        feats['stn_date_span_years'] = (
            (sdf['date'].max() - sdf['date'].min()).days / 365.25
            if len(sdf) > 1 else 0.0
        )

        stn_stats[stn] = feats

    stats_df = pd.DataFrame(stn_stats).T
    stats_df.index.name = '_station'
    stats_df = stats_df.reset_index()

    df = df.merge(stats_df, on='_station', how='left')
    print(f"   Station historical features: {len(stats_df.columns)-1} features "
          f"for {len(stats_df)} stations")
    return df


# TIME-WEIGHTED STATION FEATURES (exponential decay)
def add_time_weighted_station_features(df, all_dws, halflife_days=730):
    import dws_data as dws_mod
    full_registry = dws_mod._FULL_REGISTRY or dws_mod.STATION_REGISTRY
    decay = np.log(2) / halflife_days

    chem_cols = ['pH_Diss_Water', 'Ca_Diss_Water', 'Mg_Diss_Water',
                 'Na_Diss_Water', 'Cl_Diss_Water', 'SO4_Diss_Water',
                 'F_Diss_Water', 'Si_Diss_Water', 'K_Diss_Water',
                 'NH4_N_Diss_Water', 'NO3_NO2_N_Diss_Water',
                 'DMS_Tot_Water']

    stn_tw = {}
    for stn, sdf in all_dws.items():
        if stn not in full_registry:
            continue
        feats = {}
        if 'date' not in sdf.columns or len(sdf) < 3:
            stn_tw[stn] = feats
            continue

        ref_date = sdf['date'].max()  # most recent measurement
        days_ago = (ref_date - sdf['date']).dt.days.values
        weights = np.exp(-decay * days_ago)

        for aux in chem_cols:
            vals = pd.to_numeric(
                sdf.get(aux, pd.Series(dtype=float)), errors='coerce')
            valid = vals.notna().values
            if valid.sum() < 3:
                continue
            short = aux.split('_')[0]
            v = vals.values[valid]
            w = weights[valid]
            w_sum = w.sum()
            if w_sum < 1e-10:
                continue

            # Exponentially-weighted mean
            ew_mean = np.average(v, weights=w)
            # Exponentially-weighted std
            ew_var = np.average((v - ew_mean) ** 2, weights=w)
            ew_std = np.sqrt(ew_var)

            feats[f'stn_tw_{short}_mean'] = ew_mean
            feats[f'stn_tw_{short}_std'] = ew_std
            # Trend: difference between recent EW mean and all-time mean
            all_mean = v.mean()
            feats[f'stn_tw_{short}_trend'] = ew_mean - all_mean

        stn_tw[stn] = feats

    tw_df = pd.DataFrame(stn_tw).T
    tw_df.index.name = '_station'
    tw_df = tw_df.reset_index()

    n_before = len(df.columns)
    df = df.merge(tw_df, on='_station', how='left')
    n_new = len(df.columns) - n_before
    print(f"   Time-weighted station features: {n_new} features "
          f"for {len(tw_df)} stations (halflife={halflife_days}d)")
    return df

def adversarial_validation(train_df, test_df, features, top_k=5):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    feats = [f for f in features if f in train_df.columns and f in test_df.columns]
    if len(feats) < 5:
        print("   Adversarial validation: not enough shared features")
        return {}, 0.5, []

    X_train = train_df[feats].values.astype(np.float64)
    X_test  = test_df[feats].values.astype(np.float64)
    X = np.vstack([X_train, X_test])
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    # Fix inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.7, colsample_bytree=0.5, n_jobs=-1,
        random_state=42, verbose=-1,
    ) if HAS_LGB else xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        n_jobs=-1, random_state=42,
    )

    # 3-fold CV to get OOF probabilities
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_probs = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, oof_probs)

    # Retrain on full data for importances
    clf.fit(X, y)
    importances = clf.feature_importances_ if hasattr(clf, 'feature_importances_') else np.zeros(len(feats))
    imp_dict = dict(sorted(zip(feats, importances),
                           key=lambda x: x[1], reverse=True))
    shift_features = list(imp_dict.keys())[:top_k]

    print(f"   Adversarial validation: AUC={auc:.3f}")
    if auc > 0.7:
        print(f"   Significant distribution shift detected!")
        print(f"   Top shift features: {shift_features}")
    else:
        print(f"   Low shift (AUC={auc:.3f}) - train/test distributions similar")

    return imp_dict, auc, shift_features


# STATION-AWARE SAMPLE WEIGHTING
def compute_station_weights(df, test_stations, base_weight=1.0,
                            test_station_weight=4.0):
    stations = df.get('_station')
    if stations is None:
        return np.ones(len(df))
    weights = np.full(len(df), base_weight)
    test_mask = stations.isin(test_stations).values
    weights[test_mask] = test_station_weight
    n_test = test_mask.sum()
    print(f"   Sample weights: {n_test}/{len(df)} rows from test stations "
          f"(weight={test_station_weight}x)")
    return weights


# TARGET-SPECIFIC TRANSFORMS + MODEL CONFIGS
class TargetTransformer:
    def __init__(self, method='log1p'):
        self.method = method   # 'log1p' | 'yeojohnson' | 'shifted_log'
        self._yj = None
        self._shift = 1.0  # additive shift for shifted_log

    def fit_transform(self, y):
        y = np.array(y, dtype=np.float64)
        y = np.clip(y, 0, None)  # safety
        if self.method == 'log1p':
            return np.log1p(y)
        elif self.method == 'yeojohnson':
            self._yj = PowerTransformer(method='yeo-johnson', standardize=False)
            return self._yj.fit_transform(y.reshape(-1, 1)).ravel()
        elif self.method == 'shifted_log':
            # Shift: log1p(y + 1) - extra shift for near-zero DRP
            return np.log1p(y + self._shift)
        return y

    def transform(self, y):
        y = np.array(y, dtype=np.float64)
        y = np.clip(y, 0, None)
        if self.method == 'log1p':
            return np.log1p(y)
        elif self.method == 'yeojohnson':
            if self._yj is not None:
                return self._yj.transform(y.reshape(-1, 1)).ravel()
            return np.log1p(y)  # fallback
        elif self.method == 'shifted_log':
            return np.log1p(y + self._shift)
        return y

    def inverse_transform(self, y_t):
        y_t = np.array(y_t, dtype=np.float64)
        if self.method == 'log1p':
            return np.expm1(y_t)
        elif self.method == 'yeojohnson':
            if self._yj is not None:
                return self._yj.inverse_transform(y_t.reshape(-1, 1)).ravel()
            return np.expm1(y_t)  # fallback
        elif self.method == 'shifted_log':
            return np.expm1(y_t) - self._shift
        return y_t


def _get_target_transform(target_name):
    if 'Phosphorus' in target_name:
        return TargetTransformer('shifted_log')   # heavy right skew, many near-zero
    elif 'Conductance' in target_name:
        return TargetTransformer('log1p')         # roughly log-normal
    else:  # Alkalinity
        return TargetTransformer('yeojohnson')    # moderate skew


def _get_models_for_target(target_name):
    target_tf = _get_target_transform(target_name)

    if 'Phosphorus' in target_name:
        xgb_p = dict(
            objective='reg:pseudohubererror',
            n_estimators=1000, learning_rate=0.025, max_depth=3,
            min_child_weight=80, subsample=0.65, colsample_bytree=0.3,
            colsample_bylevel=0.5, reg_alpha=10.0, reg_lambda=30.0,
            gamma=3.0, max_delta_step=5.0,
            n_jobs=-1, tree_method='hist', random_state=42,
        )
        lgb_p = dict(
            objective='huber', alpha=1.0,  # Huber delta
            n_estimators=1000, learning_rate=0.025, num_leaves=12,
            max_depth=4, min_child_samples=80, subsample=0.65,
            colsample_bytree=0.3, reg_alpha=10.0, reg_lambda=30.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )
    elif 'Conductance' in target_name:
        xgb_p = dict(
            n_estimators=1200, learning_rate=0.025, max_depth=4,
            min_child_weight=60, subsample=0.65, colsample_bytree=0.35,
            colsample_bylevel=0.5, reg_alpha=8.0, reg_lambda=25.0,
            gamma=2.0, max_delta_step=3.0,
            n_jobs=-1, tree_method='hist', random_state=42,
        )
        lgb_p = dict(
            n_estimators=1200, learning_rate=0.025, num_leaves=15,
            max_depth=4, min_child_samples=60, subsample=0.65,
            colsample_bytree=0.35, reg_alpha=8.0, reg_lambda=25.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )
    else:  # Alkalinity
        xgb_p = dict(
            n_estimators=1200, learning_rate=0.025, max_depth=4,
            min_child_weight=60, subsample=0.65, colsample_bytree=0.35,
            colsample_bylevel=0.5, reg_alpha=8.0, reg_lambda=25.0,
            gamma=2.0, max_delta_step=3.0,
            n_jobs=-1, tree_method='hist', random_state=42,
        )
        lgb_p = dict(
            n_estimators=1200, learning_rate=0.025, num_leaves=15,
            max_depth=4, min_child_samples=60, subsample=0.65,
            colsample_bytree=0.35, reg_alpha=8.0, reg_lambda=25.0,
            n_jobs=-1, random_state=42, verbose=-1,
        )

    estimators = [('xgb', xgb.XGBRegressor(**xgb_p))]

    if HAS_LGB:
        estimators.append(('lgb', lgb.LGBMRegressor(**lgb_p)))

    if HAS_CB:
        cb_p = dict(
            iterations=1000, learning_rate=0.025, depth=4,
            l2_leaf_reg=30.0, random_seed=42, verbose=0,
            subsample=0.65, colsample_bylevel=0.35,
            min_data_in_leaf=60,
        )
        if 'Phosphorus' in target_name:
            cb_p['loss_function'] = 'Huber:delta=1.0'
        estimators.append(('cb', CatBoostRegressor(**cb_p)))

    et_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('et', ExtraTreesRegressor(
            n_estimators=1000, max_depth=15, min_samples_leaf=20,
            max_features=0.5, n_jobs=-1, random_state=42,
        )),
    ])
    estimators.append(('et', et_pipe))

    hgb = HistGradientBoostingRegressor(
        max_iter=1000, learning_rate=0.025, max_depth=4,
        min_samples_leaf=60, max_leaf_nodes=15,
        l2_regularization=25.0,
        random_state=42,
    )
    estimators.append(('hgb', hgb))

    return estimators, target_tf


# PER-STATION CALIBRATION
class StationCalibrator:
    """
    Post-prediction calibration using each station's historical distribution.

    For every test row at station S, the raw model prediction is z-scored
    against the global training distribution, then mapped onto station S's
    historical distribution.  This corrects systematic per-station bias
    (e.g., always-high-EC stations) without any model change.

    blend_weight: fraction of calibrated prediction to use (0=raw, 1=full).
    A moderate value (0.4-0.6) hedges against sparse station histories.
    """

    def __init__(self, blend_weight=0.5):
        self.blend_weight = blend_weight
        self.global_mean_ = None
        self.global_std_ = None
        self.station_stats_ = {}  # {station: (mean, std, median, q25, q75)}

    def fit(self, all_dws, target, dws_col, exclude_dates=None):
        """Compute per-station and global statistics for target.

        NOTE: all_dws data is already unit-converted by
        dws_data.load_all_station_data(), so we do NOT apply
        DWS_UNIT_CONVERSIONS here.
        """
        if exclude_dates is None:
            exclude_dates = {}

        all_vals = []
        for stn, sdf in all_dws.items():
            vals = pd.to_numeric(sdf.get(dws_col, pd.Series(dtype=float)),
                                errors='coerce')
            # NO unit conversion - already done in load_all_station_data
            # Exclude test dates
            if stn in exclude_dates:
                keep = ~sdf['date'].dt.date.isin(exclude_dates[stn])
                vals = vals[keep]
            vals = vals.dropna()
            if len(vals) >= 5:
                self.station_stats_[stn] = {
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'median': float(vals.median()),
                    'q25': float(vals.quantile(0.25)),
                    'q75': float(vals.quantile(0.75)),
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                    'n': len(vals),
                }
                all_vals.extend(vals.values)

        all_vals = np.array(all_vals)
        self.global_mean_ = float(np.mean(all_vals)) if len(all_vals) else 0.0
        self.global_std_ = float(np.std(all_vals)) if len(all_vals) else 1.0
        if self.global_std_ < 1e-8:
            self.global_std_ = 1.0

        print(f"      Calibrator: {len(self.station_stats_)} stations, "
              f"global mean={self.global_mean_:.2f}, std={self.global_std_:.2f}")
        return self

    def calibrate(self, preds, stations):
        """Apply per-station calibration to predictions.

        preds: array of shape (N,) - raw model predictions
        stations: array of shape (N,) - station codes (can have NaN)

        Returns calibrated predictions.
        """
        calibrated = preds.copy()
        w = self.blend_weight

        for i in range(len(preds)):
            stn = stations[i] if not pd.isna(stations[i]) else None
            if stn is None or stn not in self.station_stats_:
                continue

            stats = self.station_stats_[stn]
            stn_mean = stats['mean']
            stn_std = stats['std']
            if stn_std < 1e-8:
                stn_std = self.global_std_

            # Z-score the prediction against global distribution
            z = (preds[i] - self.global_mean_) / self.global_std_

            # Map to station distribution
            cal_pred = stn_mean + z * stn_std

            # Clip to station historical range (with margin)
            cal_pred = np.clip(cal_pred,
                               stats['min'] * 0.5,
                               stats['max'] * 1.5)

            # Blend: weighted average of raw and calibrated
            calibrated[i] = (1 - w) * preds[i] + w * cal_pred

        calibrated = np.clip(calibrated, 0, None)
        return calibrated


# STACKING ENSEMBLE
class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    2-layer stacking ensemble

    L1:  Base models (XGB, LGB, CB, HGB, ET, Ridge) trained with K-fold CV.
         Out-of-fold (OOF) predictions are collected.
         
    L2:  RidgeCV meta-learner learns optimal combination weights from OOF.
    Predict:  Bagged L1 predictions (averaged across K fold-models) → L2.

    AutoGluon (Erickson et al., 2020) showed multi-layer stacking
    consistently outperforms single-model or simple-average ensembles,
    beating 99% of Kaggle participants with just 4h of training.
    """

    def __init__(self, estimators, n_folds=5):
        self.estimators = estimators
        self.n_folds = n_folds

    # Models that handle NaN natively (tree-based)
    _NATIVE_NAN_MODELS = {'xgb', 'lgb', 'cb', 'hgb'}

    @staticmethod
    def _prepare_X(X, model_name=None):
        """Prepare feature matrix: keep NaN for tree models, fill for others."""
        X_arr = np.array(X, dtype=np.float64)
        X_arr = np.where(np.isposinf(X_arr), 0.0, X_arr)
        X_arr = np.where(np.isneginf(X_arr), 0.0, X_arr)
        if model_name and model_name not in StackingEnsemble._NATIVE_NAN_MODELS:
            X_arr = np.nan_to_num(X_arr, nan=0.0)
        return X_arr

    @staticmethod
    def _fit_with_weight(model, X, y, sample_weight=None):
        """Fit a model, passing sample_weight correctly for Pipelines."""
        if sample_weight is None:
            model.fit(X, y)
            return
        try:
            model.fit(X, y, sample_weight=sample_weight)
        except (TypeError, ValueError):
            if hasattr(model, 'named_steps'):
                last_step = list(model.named_steps.keys())[-1]
                try:
                    model.fit(X, y,
                              **{f'{last_step}__sample_weight': sample_weight})
                except (TypeError, ValueError):
                    model.fit(X, y)
            else:
                model.fit(X, y)

    def fit(self, X, y, groups=None, sample_weight=None):
        X_raw = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)
        w_arr = (np.array(sample_weight, dtype=np.float64)
                 if sample_weight is not None else None)
        n = len(y_arr)

        #  Set up CV splits 
        n_splits = self.n_folds
        if groups is not None:
            n_unique = len(set(groups))
            n_splits = min(n_splits, n_unique)
            gkf = GroupKFold(n_splits=n_splits)
            splits = list(gkf.split(X_raw, y_arr, groups=groups))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kf.split(X_raw, y_arr))

        n_models = len(self.estimators)
        oof_preds = np.full((n, n_models), np.nan)
        fold_models = {name: [] for name, _ in self.estimators}

        # Generate OOF predictions via K-fold CV 
        for fi, (tr_idx, va_idx) in enumerate(splits):
            for mi, (name, est) in enumerate(self.estimators):
                try:
                    m = clone(est)
                    X_tr = self._prepare_X(X_raw[tr_idx], name)
                    X_va = self._prepare_X(X_raw[va_idx], name)
                    self._fit_with_weight(
                        m, X_tr, y_arr[tr_idx],
                        w_arr[tr_idx] if w_arr is not None else None)
                    preds = m.predict(X_va)
                    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0,
                                          neginf=0.0)
                    oof_preds[va_idx, mi] = preds
                    fold_models[name].append(m)
                except Exception as e:
                    warnings.warn(f"   {name} failed in fold {fi}: {e}")
                    oof_preds[va_idx, mi] = np.nanmean(y_arr)

        # Report OOF R2
        for mi, (name, _) in enumerate(self.estimators):
            valid = np.isfinite(oof_preds[:, mi])
            if valid.sum() > 10:
                r2 = max(r2_score(y_arr[valid], oof_preds[valid, mi]), 0.0)
            else:
                r2 = 0.0
            print(f"      {name:>8s}: OOF R2={r2:.4f}")

        # Train L2 meta-learner on OOF predictions 
        valid = np.all(np.isfinite(oof_preds), axis=1)
        if valid.sum() >= 20:
            self.meta_learner_ = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', RidgeCV(
                    alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                    fit_intercept=True)),
            ])
            self.meta_learner_.fit(oof_preds[valid], y_arr[valid])
            meta_preds = self.meta_learner_.predict(oof_preds[valid])
            meta_r2 = r2_score(y_arr[valid], meta_preds)
            print(f"      {'L2 stack':>8s}: OOF R2={meta_r2:.4f}")

            # Report effective weights
            ridge_m = self.meta_learner_.named_steps['ridge']
            scaler_m = self.meta_learner_.named_steps['scaler']
            coefs = ridge_m.coef_ / (scaler_m.scale_ + 1e-10)
            # Normalise for display (actual combination uses Ridge directly)
            abs_sum = np.abs(coefs).sum() + 1e-10
            rel_w = coefs / abs_sum
            print(f"      Effective weights: "
                  f"{dict(zip([n for n,_ in self.estimators], [f'{w:.3f}' for w in rel_w]))}")
        else:
            print("      Warning: insufficient OOF data, using equal weights")
            self.meta_learner_ = None

        # Retrain all L1 models on full data 
        self.fitted_models_ = []
        self.model_names_ = []
        for name, est in self.estimators:
            m = clone(est)
            X_fit = self._prepare_X(X_raw, name)
            self._fit_with_weight(
                m, X_fit, y_arr,
                w_arr if w_arr is not None else None)
            self.fitted_models_.append(m)
            self.model_names_.append(name)

        # Store fold models for bagged prediction (more robust)
        self.fold_models_ = fold_models
        return self

    def predict(self, X):
        X_raw = np.array(X, dtype=np.float64)
        n = len(X_raw)
        n_models = len(self.fitted_models_)

        #  L1 predictions: bagged (averaged across fold models) 
        l1_preds = np.zeros((n, n_models))
        for mi, name in enumerate(self.model_names_):
            fold_ms = self.fold_models_.get(name, [])
            if len(fold_ms) >= 2:
                # Average predictions from all fold models (bagging)
                X_m = self._prepare_X(X_raw, name)
                bag = []
                for m in fold_ms:
                    p = np.nan_to_num(m.predict(X_m),
                                      nan=0.0, posinf=0.0, neginf=0.0)
                    bag.append(p)
                l1_preds[:, mi] = np.mean(bag, axis=0)
            else:
                # Fallback to full-data model
                X_m = self._prepare_X(X_raw, name)
                l1_preds[:, mi] = np.nan_to_num(
                    self.fitted_models_[mi].predict(X_m),
                    nan=0.0, posinf=0.0, neginf=0.0)

        if self.meta_learner_ is not None:
            return self.meta_learner_.predict(l1_preds)
        else:
            return l1_preds.mean(axis=1)


class MultiSeedEnsemble(BaseEstimator, RegressorMixin):
    """
    Train N StackingEnsembles with different random seeds and average.

    This reduces variance by ~1/sqrt(N) - standard in every top Kaggle
    solution.  Each seed perturbs:
      - The K-fold split (different random KFold)
      - The random_state of all L1 tree models
      - The SubSample / colsample randomness within trees

    The result is a more robust prediction that smooths out
    individual-seed noise.
    """

    def __init__(self, base_estimators, n_seeds=3, n_folds=5):
        self.base_estimators = base_estimators
        self.n_seeds = n_seeds
        self.n_folds = n_folds
        self.seed_ensembles_ = []

    @staticmethod
    def _reseeded_estimators(estimators, seed):
        """Clone estimators with a different random_state."""
        reseeded = []
        for name, est in estimators:
            m = clone(est)
            # Set random_state if the estimator supports it
            if hasattr(m, 'random_state'):
                m.set_params(random_state=seed)
            elif hasattr(m, 'random_seed'):
                m.set_params(random_seed=seed)
            # For pipelines, set the inner estimator's random_state
            if hasattr(m, 'named_steps'):
                for step_name, step in m.named_steps.items():
                    if hasattr(step, 'random_state'):
                        step.set_params(random_state=seed)
            reseeded.append((name, m))
        return reseeded

    def fit(self, X, y, groups=None, sample_weight=None):
        self.seed_ensembles_ = []
        seeds = [42 + i * 1000 for i in range(self.n_seeds)]

        for si, seed in enumerate(seeds):
            print(f"\n    Multi-seed round {si+1}/{self.n_seeds} "
                  f"(seed={seed}) ")
            reseeded = self._reseeded_estimators(
                self.base_estimators, seed)
            ens = StackingEnsemble(reseeded, n_folds=self.n_folds)
            ens.fit(X, y, groups=groups, sample_weight=sample_weight)
            self.seed_ensembles_.append(ens)

        return self

    def predict(self, X):
        """Average predictions across all seed ensembles."""
        all_preds = []
        for ens in self.seed_ensembles_:
            all_preds.append(ens.predict(X))
        return np.mean(all_preds, axis=0)


# EVALUATION  (leave-station-out CV with per-fold KNN)
def evaluate_model(df_full, features, target, groups, estimators, use_log,
                   knn_targets, knn_k=(5, 10)):
    """Spatial GroupKFold CV with KNN recomputed per fold."""
    print(f"\n   Evaluating {target} (leave-station-out, log={use_log})...")

    y_arr = df_full[target].values.astype(np.float64)
    g_arr = np.array(groups)
    n_groups = len(set(g_arr))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    fold_r2, fold_rmse, fold_mae = [], [], []
    fold_train_r2 = []

    for fi, (tri, tei) in enumerate(gkf.split(np.zeros(len(y_arr)),
                                               y_arr, g_arr)):
        df_tr = df_full.iloc[tri].copy()
        df_te = df_full.iloc[tei].copy()
        y_tr  = y_arr[tri]
        y_te  = y_arr[tei]
        g_tr  = g_arr[tri]

        # Per-fold KNN
        fold_knn = SpatialKNNEncoder(knn_targets, k_values=knn_k)
        fold_knn.fit(df_tr)
        df_tr = fold_knn.transform(df_tr, is_train=True)
        df_te = fold_knn.transform(df_te, is_train=False)

        all_feats = [f for f in df_tr.columns
                     if f in features
                     or f.startswith('knn') or f == 'nn_dist_km']
        all_feats = [f for f in all_feats if f not in config.TARGETS]
        all_feats = list(dict.fromkeys(all_feats))

        X_tr = np.array(df_tr[all_feats], dtype=np.float64)
        X_te = np.array(df_te[all_feats], dtype=np.float64)

        y_fit = np.log1p(np.clip(y_tr, 0, None)) if use_log else y_tr

        print(f"\n    Fold {fi+1}/{n_splits} "
              f"(train={len(tri)}, test={len(tei)}, "
              f"train_locs={len(set(g_tr))}) ")

        # Train a StackingEnsemble on this fold (matches submission model)
        fold_stack = StackingEnsemble(estimators, n_folds=min(3, len(set(g_tr))))
        X_tr_fix = np.where(np.isinf(X_tr), 0.0, X_tr)
        X_te_fix = np.where(np.isinf(X_te), 0.0, X_te)
        try:
            fold_stack.fit(X_tr_fix, y_fit, groups=g_tr)
            preds_raw = fold_stack.predict(X_te_fix)
        except Exception as e:
            warnings.warn(f"   StackingEnsemble failed in fold {fi+1}: {e}")
            # Fallback: simple XGB
            m_fb = clone(estimators[0][1])
            m_fb.fit(np.nan_to_num(X_tr_fix, nan=0.0), y_fit)
            preds_raw = m_fb.predict(np.nan_to_num(X_te_fix, nan=0.0))

        preds = np.expm1(preds_raw) if use_log else preds_raw
        y_cap = float(np.max(y_tr)) * 2
        preds = np.clip(preds, 0, y_cap)
        preds = np.nan_to_num(preds, nan=0.0, posinf=y_cap, neginf=0.0)

        r2  = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mae  = mean_absolute_error(y_te, preds)
        fold_r2.append(r2); fold_rmse.append(rmse); fold_mae.append(mae)

        # Train diagnostic
        tr_preds_raw = fold_stack.predict(X_tr_fix) if hasattr(fold_stack, 'predict') else preds_raw[:len(tri)]
        tr_p = np.expm1(tr_preds_raw) if use_log else tr_preds_raw
        tr_p = np.nan_to_num(np.clip(tr_p, 0, y_cap),
                             nan=0.0, posinf=y_cap, neginf=0.0)
        tr_r2 = r2_score(y_tr, tr_p)
        fold_train_r2.append(tr_r2)
        gap = tr_r2 - r2
        print(f"      Fold {fi+1}: Train R2={tr_r2:.4f}  Test R2={r2:.4f}  "
              f"Gap={gap:.4f}  RMSE={rmse:.1f}")

    mr2 = np.mean(fold_r2)
    mg  = np.mean(fold_train_r2) - mr2
    print(f"\n   CV ({n_splits}-fold):  R2={mr2:.4f} "
          f"(±{np.std(fold_r2):.3f})  "
          f"RMSE={np.mean(fold_rmse):.1f}  MAE={np.mean(fold_mae):.1f}")
    print(f"   Avg Train R2={np.mean(fold_train_r2):.4f}  Avg Gap={mg:.4f}")
    return mr2, np.mean(fold_rmse)


# FEATURE IMPORTANCE PRUNING
def _drop_station_constant_features(df, features, groups, threshold=0.03):
    g_arr = np.array(groups)
    kept, dropped = [], []
    for f in features:
        if f not in df.columns:
            continue
        vals = pd.to_numeric(df[f], errors='coerce')
        total_std = vals.std()
        if total_std < 1e-8:
            dropped.append(f)
            continue
        # Compute mean within-group std
        within_std = vals.groupby(g_arr).transform('std').mean()
        ratio = within_std / total_std
        if ratio < threshold:
            dropped.append(f)
        else:
            kept.append(f)
    if dropped:
        print(f"   Dropped {len(dropped)} station-constant features "
              f"(within/total std ratio < {threshold})")
        if len(dropped) <= 20:
            print(f"      Dropped: {dropped}")
    return kept


def prune_features(X, y, features, use_log, groups=None, min_features=30):
    if groups is not None and isinstance(X, pd.DataFrame):
        features = _drop_station_constant_features(X, features, groups)

    y_fit = np.log1p(np.clip(y, 0, None)) if use_log else y
    m = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        min_child_weight=50, subsample=0.6, colsample_bytree=0.3,
        tree_method='hist', random_state=42, n_jobs=-1,
    )
    X_arr = np.array(X[features] if isinstance(X, pd.DataFrame) else X,
                     dtype=np.float64)
    # Fix inf only; XGB handles NaN natively
    X_arr = np.where(np.isinf(X_arr), 0.0, X_arr)
    m.fit(X_arr, np.array(y_fit, dtype=np.float64))
    imps = m.feature_importances_

    feat_imp = sorted(zip(features, imps), key=lambda x: x[1], reverse=True)
    keep = [f for f, i in feat_imp if i > 0]
    dropped = [f for f, i in feat_imp if i == 0]

    if len(keep) < min_features and len(features) >= min_features:
        need = min_features - len(keep)
        keep.extend(dropped[:need])
        dropped = dropped[need:]

    if dropped:
        print(f"   Pruned {len(dropped)} zero-gain features")
    print(f"   Keeping {len(keep)} features")
    return keep


# OPTUNA HYPERPARAMETER SEARCH
def _optuna_search(X, y, groups, use_log, target_name, n_trials=80):
    if not HAS_OPTUNA:
        return None

    print(f"   Running Optuna ({n_trials} trials)...")
    y_fit = np.log1p(np.clip(y, 0, None)) if use_log else y
    X_arr = np.array(X, dtype=np.float64)
    X_arr = np.where(np.isinf(X_arr), 0.0, X_arr)  # fix inf, keep NaN
    y_arr = np.array(y_fit, dtype=np.float64)
    g_arr = np.array(groups)

    # Subsample for speed (40K rows max, preserving group structure)
    MAX_OPTUNA_ROWS = 40000
    if len(y_arr) > MAX_OPTUNA_ROWS:
        rng = np.random.RandomState(42)
        unique_groups = list(set(g_arr))
        rng.shuffle(unique_groups)
        keep_mask = np.zeros(len(y_arr), dtype=bool)
        for g in unique_groups:
            keep_mask |= (g_arr == g)
            if keep_mask.sum() >= MAX_OPTUNA_ROWS:
                break
        X_arr = X_arr[keep_mask]
        y_arr = y_arr[keep_mask]
        g_arr = g_arr[keep_mask]
        print(f"   Optuna subsampled to {len(y_arr)} rows "
              f"({len(set(g_arr))} groups)")

    n_groups = len(set(g_arr))
    n_splits = min(3, n_groups)  # 3-fold for speed
    gkf = GroupKFold(n_splits=n_splits)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_est', 500, 1500),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'min_child_weight': trial.suggest_int('mcw', 40, 200),
            'subsample': trial.suggest_float('subsample', 0.55, 0.8),
            'colsample_bytree': trial.suggest_float('colsample', 0.2, 0.45),
            'reg_alpha': trial.suggest_float('alpha', 3.0, 60.0, log=True),
            'reg_lambda': trial.suggest_float('lambda', 5.0, 100.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.5, 10.0),
            'max_delta_step': trial.suggest_float('max_delta_step', 1.0, 8.0),
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
            y_cap = float(np.max(y_real)) * 2
            preds_real = np.nan_to_num(np.clip(preds_real, 0, y_cap),
                                       nan=0.0, posinf=y_cap, neginf=0.0)
            scores.append(r2_score(y_real, preds_real))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   n_jobs=1)  # XGB already uses all cores

    best = study.best_params
    best['learning_rate'] = best.pop('lr')
    best['min_child_weight'] = best.pop('mcw')
    best['colsample_bytree'] = best.pop('colsample')
    best['reg_alpha'] = best.pop('alpha')
    best['reg_lambda'] = best.pop('lambda')
    best['n_estimators'] = best.pop('n_est')
    print(f"   Best Optuna R2: {study.best_value:.4f}")
    print(f"   Best params: {best}")
    return best


def _optuna_search_lgb(X, y, groups, use_log, target_name, n_trials=50):
    if not HAS_OPTUNA or not HAS_LGB:
        return None

    print(f"   Running Optuna-LGB ({n_trials} trials)...")
    y_fit = np.log1p(np.clip(y, 0, None)) if use_log else y
    X_arr = np.array(X, dtype=np.float64)
    X_arr = np.where(np.isinf(X_arr), 0.0, X_arr)
    y_arr = np.array(y_fit, dtype=np.float64)
    g_arr = np.array(groups)

    # Subsample for speed
    MAX_ROWS = 40000
    if len(y_arr) > MAX_ROWS:
        rng = np.random.RandomState(42)
        unique_groups = list(set(g_arr))
        rng.shuffle(unique_groups)
        keep_mask = np.zeros(len(y_arr), dtype=bool)
        for g in unique_groups:
            keep_mask |= (g_arr == g)
            if keep_mask.sum() >= MAX_ROWS:
                break
        X_arr, y_arr, g_arr = X_arr[keep_mask], y_arr[keep_mask], g_arr[keep_mask]
        print(f"   Optuna-LGB subsampled to {len(y_arr)} rows")

    n_groups = len(set(g_arr))
    n_splits = min(3, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    is_drp = 'Phosphorus' in target_name

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_est', 500, 1500),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.08, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 31),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'min_child_samples': trial.suggest_int('mcs', 40, 200),
            'subsample': trial.suggest_float('subsample', 0.55, 0.8),
            'colsample_bytree': trial.suggest_float('colsample', 0.2, 0.45),
            'reg_alpha': trial.suggest_float('alpha', 3.0, 60.0, log=True),
            'reg_lambda': trial.suggest_float('lambda', 5.0, 100.0, log=True),
            'n_jobs': -1, 'random_state': 42, 'verbose': -1,
        }
        if is_drp:
            params['objective'] = 'huber'
            params['alpha'] = trial.suggest_float('huber_delta', 0.5, 5.0)

        scores = []
        for tr, va in gkf.split(X_arr, y_arr, g_arr):
            m = lgb.LGBMRegressor(**params)
            m.fit(X_arr[tr], y_arr[tr],
                  eval_set=[(X_arr[va], y_arr[va])],
                  callbacks=[lgb.log_evaluation(-1)])
            preds = m.predict(X_arr[va])
            if use_log:
                preds_real = np.expm1(preds)
                y_real = np.expm1(y_arr[va])
            else:
                preds_real, y_real = preds, y_arr[va]
            y_cap = float(np.max(y_real)) * 2
            preds_real = np.nan_to_num(np.clip(preds_real, 0, y_cap),
                                       nan=0.0, posinf=y_cap, neginf=0.0)
            scores.append(r2_score(y_real, preds_real))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=43))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   n_jobs=1)

    best = study.best_params
    best['learning_rate'] = best.pop('lr')
    best['min_child_samples'] = best.pop('mcs')
    best['colsample_bytree'] = best.pop('colsample')
    best['reg_alpha'] = best.pop('alpha', best.get('reg_alpha', 10.0))
    best['reg_lambda'] = best.pop('lambda')
    best['n_estimators'] = best.pop('n_est')
    if 'huber_delta' in best:
        best['alpha'] = best.pop('huber_delta')
    print(f"   Best Optuna-LGB R2: {study.best_value:.4f}")
    print(f"   Best LGB params: {best}")
    return best


def _optuna_search_cb(X, y, groups, use_log, target_name, n_trials=40):
    """Optuna search over CatBoost hyperparameters."""
    if not HAS_OPTUNA or not HAS_CB:
        return None

    print(f"   Running Optuna-CB ({n_trials} trials)...")
    y_fit = np.log1p(np.clip(y, 0, None)) if use_log else y
    X_arr = np.array(X, dtype=np.float64)
    X_arr = np.where(np.isinf(X_arr), 0.0, X_arr)
    # CatBoost cannot handle NaN in float64 arrays - fill them
    X_arr = np.nan_to_num(X_arr, nan=0.0)
    y_arr = np.array(y_fit, dtype=np.float64)
    g_arr = np.array(groups)

    MAX_ROWS = 40000
    if len(y_arr) > MAX_ROWS:
        rng = np.random.RandomState(42)
        unique_groups = list(set(g_arr))
        rng.shuffle(unique_groups)
        keep_mask = np.zeros(len(y_arr), dtype=bool)
        for g in unique_groups:
            keep_mask |= (g_arr == g)
            if keep_mask.sum() >= MAX_ROWS:
                break
        X_arr, y_arr, g_arr = X_arr[keep_mask], y_arr[keep_mask], g_arr[keep_mask]
        print(f"   Optuna-CB subsampled to {len(y_arr)} rows")

    n_groups = len(set(g_arr))
    n_splits = min(3, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    is_drp = 'Phosphorus' in target_name

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iters', 500, 1500),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.08, log=True),
            'depth': trial.suggest_int('depth', 3, 6),
            'l2_leaf_reg': trial.suggest_float('l2', 5.0, 100.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.55, 0.8),
            'colsample_bylevel': trial.suggest_float('colsample', 0.2, 0.45),
            'min_data_in_leaf': trial.suggest_int('mdil', 40, 200),
            'random_seed': 42, 'verbose': 0,
        }
        if is_drp:
            delta = trial.suggest_float('huber_delta', 0.5, 5.0)
            params['loss_function'] = f'Huber:delta={delta}'

        scores = []
        for tr, va in gkf.split(X_arr, y_arr, g_arr):
            m = CatBoostRegressor(**params)
            m.fit(X_arr[tr], y_arr[tr],
                  eval_set=(X_arr[va], y_arr[va]),
                  verbose=0)
            preds = m.predict(X_arr[va])
            if use_log:
                preds_real = np.expm1(preds)
                y_real = np.expm1(y_arr[va])
            else:
                preds_real, y_real = preds, y_arr[va]
            y_cap = float(np.max(y_real)) * 2
            preds_real = np.nan_to_num(np.clip(preds_real, 0, y_cap),
                                       nan=0.0, posinf=y_cap, neginf=0.0)
            scores.append(r2_score(y_real, preds_real))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=44))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   n_jobs=1)

    best = study.best_params
    best['learning_rate'] = best.pop('lr')
    best['l2_leaf_reg'] = best.pop('l2')
    best['colsample_bylevel'] = best.pop('colsample')
    best['iterations'] = best.pop('iters')
    best['min_data_in_leaf'] = best.pop('mdil')
    if is_drp and 'huber_delta' in best:
        delta = best.pop('huber_delta')
        best['loss_function'] = f'Huber:delta={delta}'
    print(f"   Best Optuna-CB R2: {study.best_value:.4f}")
    print(f"   Best CB params: {best}")
    return best


# MAIN ENTRY POINT
def train_models(df, log_transform=None, use_optuna=True, dws_context=None):

    if dws_context is None:
        print("   No DWS context - falling back to competition data only")
        return _train_models_basic(df, log_transform, use_optuna)

    all_dws = dws_context['all_dws']
    exclude_dates = dws_context.get('test_dates_by_stn', {})
    
    COMBINED_CACHE = "combined_training_cache.parquet"

    if os.path.exists(COMBINED_CACHE):
        print(f"\n Loading cached combined training set from {COMBINED_CACHE} ")
        combined_df = pd.read_parquet(COMBINED_CACHE)
        print(f"   {len(combined_df)} rows, {len(combined_df.columns)} columns")
    else:
        print("\n Building DWS training dataset ")
        dws_df = build_dws_training_set(all_dws, config.TARGETS,
                                        exclude_dates=exclude_dates)

        print("\n Station historical features ")
        dws_df = add_station_historical_features(dws_df, all_dws)

        print("\n Time-weighted station features ")
        dws_df = add_time_weighted_station_features(dws_df, all_dws,
                                                     halflife_days=730)

        print("\n Neighbor/upstream features ")
        import dws_data as _dws_mod
        dws_df = _dws_mod.build_neighbor_features(dws_df, all_dws)

        print("\n Merging enrichment features ")
        dws_df = _merge_enrichment_to_dws(dws_df, df)

        print("\n eature engineering ")
        dws_df = engineer_features(dws_df)

        print("\n Combining with competition training data ")
        comp_df = _prepare_competition_data(df, all_dws, dws_df)

        if comp_df is not None and len(comp_df) > 0:
            combined_df = pd.concat([dws_df, comp_df], ignore_index=True)
            print(f"   Combined: {len(dws_df)} DWS + {len(comp_df)} competition "
                  f"= {len(combined_df)} total rows")
        else:
            combined_df = dws_df
            print(f"   Using DWS data only: {len(combined_df)} rows")

        # Cache to disk
        # Ensure 'Sample Date' is string type for parquet serialisation
        if 'Sample Date' in combined_df.columns:
            combined_df['Sample Date'] = combined_df['Sample Date'].astype(str)
        combined_df.to_parquet(COMBINED_CACHE, index=False)
        print(f"   Cached combined training set to {COMBINED_CACHE} "
              f"({os.path.getsize(COMBINED_CACHE)/1e6:.1f} MB)")

    # KNN spatial features (chemistry-based, fit on full combined data) 
    print("\n Step 6: KNN spatial features (chemistry-based) ")
    knn_enc = SpatialKNNEncoder(config.TARGETS, k_values=(5, 10))
    knn_enc.fit(combined_df)
    combined_knn = knn_enc.transform(combined_df, is_train=True)
    knn_cols = [c for c in combined_knn.columns
                if c.startswith('knn') or c == 'nn_dist_km']
    print(f"   Added {len(knn_cols)} KNN features")

    #  Feature selection 
    always_drop = ['Sample Date', '_station', '_dt', '_loc_id', '_enc_loc_id',
                   '_is_missing', '_geo_key', 'Latitude', 'Longitude',
                   '_dws_station', 'date', 'station', 'key']
    drop_cols = config.TARGETS + always_drop
    base_features = [c for c in combined_df.columns
                     if c not in drop_cols
                     and pd.api.types.is_numeric_dtype(combined_df[c])]
    print(f"   Base features: {len(base_features)}")

    import dws_data as dws_mod
    DWS_COL_MAP_INV = {v: k for k, v in dws_mod.DWS_COL_MAP.items()}

    # Determine test station set for sample weighting
    test_stations = set(dws_context.get('test_dates_by_stn', {}).keys())

    performance_report = {
        '_knn_encoder': knn_enc,
        '_xt_featurizer': None,
        '_base_features': base_features,
        '_dws_context': dws_context,
        '_lstm_models': {},
        '_chain_order': CHAIN_ORDER,
    }

    # Store OOF chain predictions to use as features for downstream targets
    chain_oof_preds = {}  # {target: pd.Series of OOF predictions}

    for target in CHAIN_ORDER:
        if target not in combined_df.columns:
            continue

        print(f"\n{'='*65}")
        print(f"   TARGET: {target}")
        print(f"{'='*65}")

        estimators, target_tf = _get_models_for_target(target)
        # log_transform override: None means use the target-specific transform
        use_log = None
        tf_method = target_tf.method

        tdf = combined_df.dropna(subset=[target]).copy()

        #  Multi-target chaining: add upstream predictions as features 
        for prev_target, prev_preds in chain_oof_preds.items():
            chain_col = f'_chain_{prev_target[:3].upper()}'
            tdf[chain_col] = prev_preds.reindex(tdf.index).values
            n_valid_chain = tdf[chain_col].notna().sum()
            print(f"   Chained feature {chain_col}: "
                  f"{n_valid_chain}/{len(tdf)} valid")

        # Groups = station (for DWS rows) or lat_lon (for competition rows)
        groups = tdf['_station'].fillna(
            tdf['Latitude'].round(2).astype(str) + "_" +
            tdf['Longitude'].round(2).astype(str)
        )

        y = tdf[target].copy()

        # Winsorise at 1st/99th percentile
        lo, hi = y.quantile(0.01), y.quantile(0.99)
        y = y.clip(lower=lo, upper=hi)
        tdf[target] = y

        #  Station-aware sample weights 
        sample_weights = compute_station_weights(
            tdf, test_stations,
            base_weight=1.0, test_station_weight=4.0)

        print(f"   Samples: {len(tdf)},  Groups: {groups.nunique()}")
        print(f"   y: mean={y.mean():.2f}  std={y.std():.2f}  "
              f"skew={y.skew():.2f}  transform={tf_method}")

        target_dws_col = DWS_COL_MAP_INV.get(target)
        lstm_model = None
        if target_dws_col is not None:
            print(f"\n    Step 7a: Training LSTM for {target} ")
            lstm_model = StationLSTMModel(
                target_dws_col=target_dws_col,
                target_name=target,
                use_log=use_log,
                seq_len=36,
                hidden=96,
                n_layers=2,
                epochs=70,
                lr=1e-3,
                batch_size=512,
            )
            try:
                lstm_model.fit(all_dws, tdf, exclude_dates=exclude_dates)
                lstm_preds = lstm_model.predict(
                    all_dws, tdf, exclude_dates=exclude_dates)
                tdf['_lstm_pred'] = lstm_preds
                n_valid = np.isfinite(lstm_preds).sum()
                print(f"      LSTM predictions: {n_valid}/{len(tdf)} valid")
                performance_report['_lstm_models'][target] = lstm_model
            except Exception as e:
                print(f"      LSTM failed (non-fatal): {e}")
                traceback.print_exc()
                tdf['_lstm_pred'] = np.nan
                lstm_model = None

        # Build feature list: base + chain + LSTM
        tgt_base = list(base_features)
        chain_cols = [c for c in tdf.columns if c.startswith('_chain_')]
        tgt_base.extend(chain_cols)

        # Prune - pass groups to also drop station-constant features
        X_base = tdf[tgt_base]
        tgt_feats = prune_features(X_base, y, tgt_base, use_log, groups=groups)
        # Add LSTM prediction as a feature if available
        if '_lstm_pred' in tdf.columns:
            tgt_feats.append('_lstm_pred')
        # Ensure chain features survive pruning
        for cc in chain_cols:
            if cc not in tgt_feats:
                tgt_feats.append(cc)
        print(f"   Features after pruning: {len(tgt_feats)}")

        # Use test_template rows to detect distribution shift
        try:
            test_tmpl = dws_context.get('test_template')
            if test_tmpl is not None and len(test_tmpl) > 0:
                adv_feats = [f for f in tgt_feats
                             if not f.startswith('_') and f in tdf.columns
                             and f in test_tmpl.columns]
                if len(adv_feats) >= 10:
                    adv_imp, adv_auc, shift_feats = adversarial_validation(
                        tdf, test_tmpl, adv_feats, top_k=5)
                    # If high shift, use adversarial score for sample weighting
                    # (NOT as feature - avoids leaking "is_test_like" into model)
                    if adv_auc > 0.7 and HAS_LGB:
                        print("   Using adversarial score for sample weighting...")
                        clf_adv = lgb.LGBMClassifier(
                            n_estimators=100, max_depth=3, n_jobs=-1,
                            random_state=42, verbose=-1)
                        X_adv = np.nan_to_num(tdf[adv_feats].values, nan=0.0)
                        X_test_adv = np.nan_to_num(test_tmpl[adv_feats].values, nan=0.0)
                        X_all_adv = np.vstack([X_adv, X_test_adv])
                        y_adv = np.concatenate([np.zeros(len(X_adv)),
                                                np.ones(len(X_test_adv))])
                        clf_adv.fit(X_all_adv, y_adv)
                        adv_scores = clf_adv.predict_proba(X_adv)[:, 1]
                        # Upweight training rows that look like test data
                        sample_weights = sample_weights * (1.0 + adv_scores)
                        print(f"   Adversarial weighting: mean boost={adv_scores.mean():.3f}")
                        performance_report[f'_adv_classifier_{target}'] = clf_adv
                        performance_report[f'_adv_features_{target}'] = adv_feats
        except Exception as e:
            print(f"   Adversarial validation failed (non-fatal): {e}")

        # Optuna - use only base features (no global KNN to avoid leakage)
        if use_optuna and HAS_OPTUNA:
            optuna_feats = [f for f in tgt_feats if f not in config.TARGETS]
            optuna_feats = list(dict.fromkeys(optuna_feats))
            best_xgb = _optuna_search(
                tdf[optuna_feats], y, groups, use_log, target, n_trials=50)
            if best_xgb is not None:
                best_xgb['tree_method'] = 'hist'
                best_xgb['n_jobs'] = -1
                best_xgb['random_state'] = 42
                # Preserve Huber loss for DRP
                if 'Phosphorus' in target:
                    best_xgb['objective'] = 'reg:pseudohubererror'
                estimators = [(n, e) if n != 'xgb' else
                              ('xgb', xgb.XGBRegressor(**best_xgb))
                              for n, e in estimators]

            # Optuna for LightGBM
            best_lgb = _optuna_search_lgb(
                tdf[optuna_feats], y, groups, use_log, target, n_trials=50)
            if best_lgb is not None:
                best_lgb['n_jobs'] = -1
                best_lgb['random_state'] = 42
                best_lgb['verbose'] = -1
                if 'Phosphorus' in target and 'objective' not in best_lgb:
                    best_lgb['objective'] = 'huber'
                estimators = [(n, e) if n != 'lgb' else
                              ('lgb', lgb.LGBMRegressor(**best_lgb))
                              for n, e in estimators]

            # Optuna for CatBoost
            best_cb = _optuna_search_cb(
                tdf[optuna_feats], y, groups, use_log, target, n_trials=40)
            if best_cb is not None:
                best_cb['random_seed'] = 42
                best_cb['verbose'] = 0
                estimators = [(n, e) if n != 'cb' else
                              ('cb', CatBoostRegressor(**best_cb))
                              for n, e in estimators]

        # Evaluate
        r2, rmse = evaluate_model(tdf, tgt_feats, target, groups,
                                  estimators, use_log, config.TARGETS,
                                  knn_k=(5, 10))

        # Final fit on ALL data
        print("   Training final model on ALL data...")
        tdf_final = knn_enc.transform(tdf, is_train=True)
        all_feats = [f for f in tdf_final.columns
                     if (f in tgt_feats
                         or f.startswith('knn')
                         or f == 'nn_dist_km')
                     and f not in config.TARGETS]
        all_feats = list(dict.fromkeys(all_feats))

        y_fit = target_tf.fit_transform(y)
        final = MultiSeedEnsemble(estimators, n_seeds=3, n_folds=5)
        X_final = np.array(tdf_final[all_feats], dtype=np.float64)
        # Fix inf only; StackingEnsemble._prepare_X handles NaN per-model
        X_final = np.where(np.isinf(X_final), 0.0, X_final)
        final.fit(
            X_final,
            np.array(y_fit, dtype=np.float64),
            groups=np.array(groups),
            sample_weight=sample_weights)

        print("   Generating OOF chain predictions...")
        try:
            n_splits_chain = min(5, len(set(groups)))
            gkf_chain = GroupKFold(n_splits=n_splits_chain)
            oof_chain = np.full(len(tdf), np.nan)
            for tri, tei in gkf_chain.split(X_final, y_fit, groups.values):
                m_chain = clone(xgb.XGBRegressor(
                    n_estimators=500, learning_rate=0.03, max_depth=4,
                    tree_method='hist', n_jobs=-1, random_state=42))
                m_chain.fit(X_final[tri], y_fit[tri])
                p = m_chain.predict(X_final[tei])
                # Inverse transform
                p_orig = target_tf.inverse_transform(p)
                p_orig = np.clip(np.nan_to_num(p_orig, nan=0.0), 0, None)
                oof_chain[tei] = p_orig
            chain_oof_preds[target] = pd.Series(oof_chain, index=tdf.index)
            print(f"   OOF chain: {np.isfinite(oof_chain).sum()}/{len(oof_chain)} valid, "
                  f"mean={np.nanmean(oof_chain):.2f}")
        except Exception as e:
            print(f"   OOF chain generation failed (non-fatal): {e}")

        #  Station Calibrator 
        calibrator = None
        if target_dws_col is not None:
            print(f"   Training station calibrator for {target}...")
            calibrator = StationCalibrator(blend_weight=0.5)
            calibrator.fit(all_dws, target, target_dws_col,
                           exclude_dates=exclude_dates)

        performance_report[target] = {
            'R2': r2, 'RMSE': rmse,
            'model': final,
            'features': all_feats,
            'log_transform': use_log,
            'target_transformer': target_tf,
            'global_mean': float(y.mean()),
            'calibrator': calibrator,
        }

        safe = target.replace(' ', '_')
        joblib.dump(final, f"model_{safe}.joblib")
        print(f"   Saved model_{safe}.joblib")

    joblib.dump(knn_enc, "knn_encoder.joblib")

    print(f"\n{'='*65}")
    print("   FINAL CV RESULTS")
    print(f"{'='*65}")
    for t in config.TARGETS:
        if t in performance_report:
            m = performance_report[t]
            print(f"   {t:>35s}:  R2={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  "
                  f"log={m['log_transform']}")

    return performance_report

def _merge_enrichment_to_dws(dws_df, enriched_df):
    enr_cols = [c for c in enriched_df.columns
                if c not in config.TARGETS + ['Sample Date', 'Latitude', 'Longitude']
                and pd.api.types.is_numeric_dtype(enriched_df[c])]

    if not enr_cols:
        print("   No enrichment features to merge.")
        return dws_df

    enriched_df = enriched_df.copy()
    loc_key = (enriched_df['Latitude'].round(4).astype(str) + '_' +
               enriched_df['Longitude'].round(4).astype(str))
    enriched_df['_loc'] = loc_key

    # All-time per-location medians (fallback)
    loc_medians = enriched_df.groupby('_loc')[enr_cols].median()

    # Season-aware per-location medians
    has_season = False
    if 'Sample Date' in enriched_df.columns:
        try:
            enr_dates = pd.to_datetime(
                enriched_df['Sample Date'], format='mixed', dayfirst=True)
            enr_month = enr_dates.dt.month
            enriched_df['_season'] = enr_month.map(
                {12: 'DJF', 1: 'DJF', 2: 'DJF',
                 3: 'MAM', 4: 'MAM', 5: 'MAM',
                 6: 'JJA', 7: 'JJA', 8: 'JJA',
                 9: 'SON', 10: 'SON', 11: 'SON'})
            loc_season_medians = enriched_df.groupby(
                ['_loc', '_season'])[enr_cols].median()
            has_season = True
            print("   Using season-aware enrichment merge")
        except Exception:
            has_season = False

    # DWS season if available
    dws_season = None
    if has_season:
        dws_date_col = None
        if 'date' in dws_df.columns:
            dws_date_col = 'date'
        elif 'Sample Date' in dws_df.columns:
            dws_date_col = 'Sample Date'
        if dws_date_col is not None:
            try:
                dws_dates = pd.to_datetime(
                    dws_df[dws_date_col], format='mixed', dayfirst=True)
                dws_month = dws_dates.dt.month
                dws_season = dws_month.map(
                    {12: 'DJF', 1: 'DJF', 2: 'DJF',
                     3: 'MAM', 4: 'MAM', 5: 'MAM',
                     6: 'JJA', 7: 'JJA', 8: 'JJA',
                     9: 'SON', 10: 'SON', 11: 'SON'})
            except Exception:
                dws_season = None

    # Build location tree
    comp_locs = enriched_df[['Latitude', 'Longitude']].drop_duplicates().values
    comp_loc_keys = (pd.Series(comp_locs[:, 0]).round(4).astype(str) + '_' +
                     pd.Series(comp_locs[:, 1]).round(4).astype(str)).values

    if len(comp_locs) == 0:
        return dws_df

    from sklearn.neighbors import BallTree as BT
    tree = BT(np.deg2rad(comp_locs), metric='haversine')

    for f in enr_cols:
        dws_df[f] = np.nan

    dws_loc_key = (dws_df['Latitude'].round(4).astype(str) + '_' +
                   dws_df['Longitude'].round(4).astype(str))

    # Missingness indicators for enrichment
    dws_df['_has_enrichment'] = 0
    dws_df['_enrichment_dist_km'] = np.nan

    # For efficiency, work per-(location, season) group
    if has_season and dws_season is not None:
        dws_df['_tmp_loc'] = dws_loc_key
        dws_df['_tmp_season'] = dws_season
        unique_groups = dws_df.groupby(['_tmp_loc', '_tmp_season']).groups
    else:
        dws_df['_tmp_loc'] = dws_loc_key
        unique_groups = {(loc, None): idx
                         for loc, idx in dws_df.groupby('_tmp_loc').groups.items()}

    filled = 0
    for (uloc, season), row_idx in unique_groups.items():
        mask = dws_df.index.isin(row_idx)
        lat = dws_df.loc[mask, 'Latitude'].iloc[0]
        lon = dws_df.loc[mask, 'Longitude'].iloc[0]

        k = min(3, len(comp_locs))
        dists, inds = tree.query(np.deg2rad([[lat, lon]]), k=k)
        dists_km = dists[0] * 6371.0
        weights = 1.0 / (dists_km + 1.0)
        weights /= weights.sum()

        any_filled = False
        for f in enr_cols:
            vals, ws = [], []
            for j, idx in enumerate(inds[0]):
                lk = comp_loc_keys[idx]
                v = None
                # Try season-specific median first
                if has_season and season is not None:
                    try:
                        v = loc_season_medians.at[(lk, season), f]
                    except (KeyError, TypeError):
                        v = None
                # Fall back to all-time median
                if v is None or pd.isna(v):
                    if lk in loc_medians.index:
                        v = loc_medians.at[lk, f]
                if pd.notna(v):
                    vals.append(v)
                    ws.append(weights[j])
            if vals:
                dws_df.loc[mask, f] = np.average(vals, weights=ws)
                any_filled = True
        if any_filled:
            dws_df.loc[mask, '_has_enrichment'] = 1
            dws_df.loc[mask, '_enrichment_dist_km'] = float(dists_km[0])
        filled += 1

    # Clean up temp columns
    dws_df.drop(columns=['_tmp_loc'], inplace=True, errors='ignore')
    dws_df.drop(columns=['_tmp_season'], inplace=True, errors='ignore')

    print(f"   Merged enrichment features: {len(enr_cols)} features "
          f"for {filled} DWS location-season groups")
    return dws_df


def _prepare_competition_data(comp_df, all_dws, dws_df):
    """
    Prepare competition training data to be combined with DWS data.
    Add DWS aux features to competition rows where possible.
    """
    import dws_data as dws_mod

    comp = comp_df.copy()
    comp = engineer_features(comp)
    comp['_station'] = comp.apply(
        lambda r: dws_mod.coord_to_station(r['Latitude'], r['Longitude']),
        axis=1)

    # Source indicator
    comp['_is_dws'] = 0
    comp['_has_enrichment'] = 1   # competition data IS the enrichment source
    comp['_enrichment_dist_km'] = 0.0

    # Parse dates
    if 'date' not in comp.columns or comp['date'].dtype == object:
        if 'Sample Date' in comp.columns:
            comp['date'] = pd.to_datetime(comp['Sample Date'], dayfirst=True)

    # Add DWS aux features where stations match
    comp = dws_mod.add_sameday_aux_features(comp, all_dws)

    dws_aux_cols = [c for c in comp.columns if c.startswith('dws_aux_')]
    if dws_aux_cols:
        comp['_has_dws_aux'] = comp[dws_aux_cols].notna().any(axis=1).astype(int)
    else:
        comp['_has_dws_aux'] = 0

    # Add lag features
    comp = dws_mod.add_lag_features(comp, all_dws, config.TARGETS)

    lag_cols = [c for c in comp.columns if c.startswith('lag_') or c.startswith('roll')]
    if lag_cols:
        comp['_has_dws_lag'] = comp[lag_cols].notna().any(axis=1).astype(int)
    else:
        comp['_has_dws_lag'] = 0

    # Add station historical features
    comp = add_station_historical_features(comp, all_dws)

    # Add time-weighted station features
    comp = add_time_weighted_station_features(comp, all_dws, halflife_days=730)

    # Add neighbor/upstream features
    comp = dws_mod.build_neighbor_features(comp, all_dws)

    # Missingness indicator: does this comp row have station history?
    stn_cols = [c for c in comp.columns if c.startswith('stn_')]
    if stn_cols:
        comp['_has_stn_hist'] = comp[stn_cols].notna().any(axis=1).astype(int)
    else:
        comp['_has_stn_hist'] = 0

    # Drop helper columns
    comp.drop(columns=['date', 'station', '_dws_station'],
              errors='ignore', inplace=True)

    dws_cols = set(dws_df.columns)
    for c in dws_cols:
        if c not in comp.columns and c not in config.TARGETS:
            comp[c] = np.nan

    return comp


def _train_models_basic(df, log_transform, use_optuna):
    """Fallback: basic training without DWS context."""
    print("   Basic mode (no DWS data available)")
    df = engineer_features(df)

    knn_enc = SpatialKNNEncoder(config.TARGETS, k_values=(5, 10))
    knn_enc.fit(df)

    always_drop = ['Sample Date', '_station', 'Latitude', 'Longitude',
                   'date', 'station', 'key']
    drop_cols = config.TARGETS + always_drop
    base_features = [c for c in df.columns
                     if c not in drop_cols
                     and pd.api.types.is_numeric_dtype(df[c])]

    performance_report = {
        '_knn_encoder': knn_enc,
        '_xt_featurizer': None,
        '_base_features': base_features,
        '_dws_context': None,
        '_sarimax': None,
    }

    for target in config.TARGETS:
        if target not in df.columns:
            continue

        estimators, use_log_auto = _get_models_for_target(target)
        use_log = use_log_auto if log_transform is None else log_transform

        tdf = df.dropna(subset=[target]).copy()
        groups = (tdf['Latitude'].round(2).astype(str) + "_" +
                  tdf['Longitude'].round(2).astype(str))
        y = tdf[target].clip(lower=tdf[target].quantile(0.01),
                             upper=tdf[target].quantile(0.99))

        tdf_knn = knn_enc.transform(tdf, is_train=True)
        all_feats = [f for f in tdf_knn.columns
                     if (f in base_features or '_knn' in f or '_nn_dist' in f)
                     and f not in config.TARGETS]
        all_feats = list(dict.fromkeys(all_feats))

        y_fit = np.log1p(np.clip(y, 0, None)) if use_log else y.values
        final = StackingEnsemble(estimators, n_folds=5)
        X_basic = np.array(tdf_knn[all_feats], dtype=np.float64)
        X_basic = np.where(np.isinf(X_basic), 0.0, X_basic)
        final.fit(
            X_basic,
            np.array(y_fit, dtype=np.float64),
            groups=np.array(groups))

        performance_report[target] = {
            'R2': 0.0, 'RMSE': 999.0,
            'model': final,
            'features': all_feats,
            'log_transform': use_log,
            'global_mean': float(y.mean()),
        }

    return performance_report


_PS_AUX_COLS = [
    'pH_Diss_Water', 'Ca_Diss_Water', 'Mg_Diss_Water',
    'Na_Diss_Water', 'Cl_Diss_Water', 'SO4_Diss_Water',
    'F_Diss_Water', 'Si_Diss_Water', 'K_Diss_Water',
    'NH4_N_Diss_Water', 'NO3_NO2_N_Diss_Water',
    'DMS_Tot_Water', 'P_Tot_Water',
]

# Target-specific priority features for per-station regression
_PS_PRIORITY_FEATURES = {
    'Electrical Conductance': [
        'Na_Diss_Water', 'DMS_Tot_Water', 'Mg_Diss_Water',
        'Cl_Diss_Water', 'SO4_Diss_Water', 'Ca_Diss_Water',
        'K_Diss_Water', 'Si_Diss_Water',
    ],
    'Total Alkalinity': [
        'pH_Diss_Water', 'Ca_Diss_Water', 'Mg_Diss_Water',
        'DMS_Tot_Water', 'F_Diss_Water', 'Na_Diss_Water',
    ],
    'Dissolved Reactive Phosphorus': [
        'P_Tot_Water', 'NH4_N_Diss_Water', 'NO3_NO2_N_Diss_Water',
        'pH_Diss_Water', 'Si_Diss_Water', 'F_Diss_Water',
    ],
}


def _ps_temporal_interpolation(sdf, dws_col, target_date, k=7):
    """
    Predict target value from k nearest temporal neighbours using
    inverse-distance weighting in time.  Excludes the exact test date.

    Returns (prediction, confidence).
    """
    valid = sdf.copy()
    valid[dws_col] = pd.to_numeric(valid[dws_col], errors='coerce')
    valid = valid[(valid['date'].dt.date != target_date.date()) &
                  (valid[dws_col].notna())].sort_values('date')

    if len(valid) == 0:
        return None, 0.0

    vals = valid[dws_col].values.astype(np.float64)
    days_diff = np.abs(
        (valid['date'] - target_date).dt.total_seconds().values / 86400)

    k_actual = min(k, len(valid))
    idx = np.argpartition(days_diff, k_actual)[:k_actual]

    nearest_vals = vals[idx]
    nearest_days = days_diff[idx]

    weights = 1.0 / (nearest_days + 1.0)
    weights /= weights.sum()

    pred = float(np.average(nearest_vals, weights=weights))

    min_gap = nearest_days.min()
    if min_gap <= 7:
        conf = 0.95
    elif min_gap <= 30:
        conf = 0.85
    elif min_gap <= 90:
        conf = 0.60
    elif min_gap <= 180:
        conf = 0.35
    else:
        conf = 0.15

    return pred, conf


def _ps_regression(sdf, dws_col, target_date, aux_values, target_name=None):
    """
    Per-station ElasticNetCV regression: aux_chemistry + month → target.
    Trained on all data at this station EXCEPT the test date.
    Uses polynomial features for key interactions.

    Returns (prediction, confidence, model_r2).
    """
    train = sdf[sdf['date'].dt.date != target_date.date()].copy()
    train[dws_col] = pd.to_numeric(train[dws_col], errors='coerce')
    train = train[train[dws_col].notna()].copy()

    if len(train) < 5:
        return None, 0.0, 0.0

    # Select aux columns with enough coverage at this station
    # Prioritise target-specific features
    priority = _PS_PRIORITY_FEATURES.get(target_name, []) if target_name else []
    available = [c for c in _PS_AUX_COLS if c in train.columns]
    good_aux = []
    for c in priority + [c for c in available if c not in priority]:
        if c not in available:
            continue
        v = pd.to_numeric(train[c], errors='coerce')
        if v.notna().sum() >= max(5, len(train) * 0.15):
            good_aux.append(c)

    if len(good_aux) < 1:
        return None, 0.0, 0.0

    for c in good_aux:
        train[c] = pd.to_numeric(train[c], errors='coerce')

    # Seasonal features
    month = train['date'].dt.month
    train['_m_sin'] = np.sin(2 * np.pi * month / 12)
    train['_m_cos'] = np.cos(2 * np.pi * month / 12)
    train['_year_frac'] = (train['date'].dt.year +
                           train['date'].dt.dayofyear / 365.25)

    feature_cols = good_aux + ['_m_sin', '_m_cos', '_year_frac']
    subset = train.dropna(subset=good_aux).copy()

    if len(subset) < 5:
        return None, 0.0, 0.0

    X_train = subset[feature_cols].values.astype(np.float64)
    y_train = subset[dws_col].values.astype(np.float64)

    # Build test vector
    x_test = []
    for c in good_aux:
        v = aux_values.get(c, np.nan)
        if np.isnan(v):
            v = float(subset[c].median())
        x_test.append(v)
    x_test.append(np.sin(2 * np.pi * target_date.month / 12))
    x_test.append(np.cos(2 * np.pi * target_date.month / 12))
    x_test.append(target_date.year + target_date.timetuple().tm_yday / 365.25)
    X_test = np.array([x_test], dtype=np.float64)

    if np.any(np.isnan(X_test)):
        return None, 0.0, 0.0

    # Standardise
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Add polynomial features for top chemistry pairs (degree 2)
    # Only if enough samples to support the extra features
    n_poly_feats = X_train_s.shape[1]
    use_poly = len(subset) > 3 * (n_poly_feats * (n_poly_feats + 1) // 2)
    if use_poly and len(good_aux) >= 2:
        poly = PolynomialFeatures(degree=2, include_bias=False,
                                  interaction_only=False)
        X_train_s = poly.fit_transform(X_train_s)
        X_test_s = poly.transform(X_test_s)

    # ElasticNetCV with automatic regularisation + feature selection
    try:
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            alphas=np.logspace(-4, 2, 30),
            cv=min(5, len(subset)),
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_train_s, y_train)
        pred = float(model.predict(X_test_s)[0])

        # Confidence ≈ LOO-ish training R2
        train_pred = model.predict(X_train_s)
        r2 = max(0, r2_score(y_train, train_pred))

    except Exception:
        alpha = max(1.0, 300.0 / len(subset))
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)
        pred = float(model.predict(X_test_s)[0])
        train_pred = model.predict(X_train_s)
        r2 = max(0, r2_score(y_train, train_pred))

    # Clip to station's historical range with margin
    y_range = y_train.max() - y_train.min()
    margin = y_range * 0.4  
    pred = np.clip(pred,
                   max(0, y_train.min() - margin),
                   y_train.max() + margin)

    return pred, min(r2, 0.99), r2


def _ps_gradient_boosting(sdf, dws_col, target_date, aux_values,
                          target_name=None):
    train = sdf[sdf['date'].dt.date != target_date.date()].copy()
    train[dws_col] = pd.to_numeric(train[dws_col], errors='coerce')
    train = train[train[dws_col].notna()].copy()

    if len(train) < 30:
        return None, 0.0, 0.0

    # Select aux columns
    available = [c for c in _PS_AUX_COLS if c in train.columns]
    good_aux = []
    for c in available:
        v = pd.to_numeric(train[c], errors='coerce')
        if v.notna().sum() >= max(10, len(train) * 0.15):
            good_aux.append(c)

    if len(good_aux) < 2:
        return None, 0.0, 0.0

    for c in good_aux:
        train[c] = pd.to_numeric(train[c], errors='coerce')

    # Features
    month = train['date'].dt.month
    train['_m_sin'] = np.sin(2 * np.pi * month / 12)
    train['_m_cos'] = np.cos(2 * np.pi * month / 12)
    train['_year_frac'] = (train['date'].dt.year +
                           train['date'].dt.dayofyear / 365.25)

    feature_cols = good_aux + ['_m_sin', '_m_cos', '_year_frac']
    subset = train.dropna(subset=good_aux).copy()

    if len(subset) < 30:
        return None, 0.0, 0.0

    X_train = subset[feature_cols].values.astype(np.float64)
    y_train = subset[dws_col].values.astype(np.float64)

    # Build test vector
    x_test = []
    for c in good_aux:
        v = aux_values.get(c, np.nan)
        if np.isnan(v):
            v = float(subset[c].median())
        x_test.append(v)
    x_test.append(np.sin(2 * np.pi * target_date.month / 12))
    x_test.append(np.cos(2 * np.pi * target_date.month / 12))
    x_test.append(target_date.year + target_date.timetuple().tm_yday / 365.25)
    X_test = np.array([x_test], dtype=np.float64)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Regularised GBR - conservative to prevent per-station overfitting
    n_est = min(200, max(50, len(subset) // 2))
    model = GradientBoostingRegressor(
        n_estimators=n_est,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=max(5, len(subset) // 20),
        subsample=0.8,
        random_state=42,
    )

    try:
        model.fit(X_train, y_train)
        pred = float(model.predict(X_test)[0])

        # OOF-like R2 estimate via last 20% temporal split
        n_tr = int(0.8 * len(subset))
        if n_tr >= 10:
            dates_sorted = subset.sort_values('date').index
            tr_idx = dates_sorted[:n_tr]
            va_idx = dates_sorted[n_tr:]
            X_va = subset.loc[va_idx, feature_cols].values.astype(np.float64)
            X_va = np.nan_to_num(X_va, nan=0.0)
            y_va = subset.loc[va_idx, dws_col].values.astype(np.float64)
            X_tr_split = subset.loc[tr_idx, feature_cols].values.astype(np.float64)
            X_tr_split = np.nan_to_num(X_tr_split, nan=0.0)
            y_tr_split = subset.loc[tr_idx, dws_col].values.astype(np.float64)
            m_split = GradientBoostingRegressor(
                n_estimators=n_est, learning_rate=0.05, max_depth=3,
                min_samples_leaf=max(5, len(subset) // 20),
                subsample=0.8, random_state=42)
            m_split.fit(X_tr_split, y_tr_split)
            va_pred = m_split.predict(X_va)
            r2 = max(0, r2_score(y_va, va_pred))
        else:
            train_pred = model.predict(X_train)
            r2 = max(0, r2_score(y_train, train_pred)) * 0.7  # discount

    except Exception:
        return None, 0.0, 0.0

    # Clip to station range
    y_range = y_train.max() - y_train.min()
    margin = y_range * 0.4
    pred = np.clip(pred,
                   max(0, y_train.min() - margin),
                   y_train.max() + margin)

    return pred, min(r2, 0.99), r2


def per_station_predict(sub_df, all_dws):
    import dws_data as dws_mod

    sub = sub_df.copy()
    if 'date' not in sub.columns or sub['date'].dtype == object:
        if 'Sample Date' in sub.columns:
            sub['date'] = pd.to_datetime(sub['Sample Date'], dayfirst=True)

    if '_station' not in sub.columns:
        sub['_station'] = sub.apply(
            lambda r: dws_mod.coord_to_station(
                r['Latitude'], r['Longitude']),
            axis=1)

    results = {}
    confidences = {}

    for target in config.TARGETS:
        dws_col = dws_mod._COL_FOR_TARGET.get(target)
        if dws_col is None:
            continue

        print(f"      {target}:")
        preds = np.full(len(sub), np.nan)
        confs = np.full(len(sub), 0.0)
        n_reg, n_gb, n_interp, n_miss = 0, 0, 0, 0
        reg_r2s, gb_r2s = [], []

        for pos in range(len(sub)):
            row = sub.iloc[pos]
            station = row.get('_station')
            date = row['date']

            if pd.isna(station) or station not in all_dws:
                n_miss += 1
                continue

            sdf = all_dws[station]

            # Same-day auxiliary chemistry
            same_day = sdf[sdf['date'].dt.date == date.date()]
            aux_vals = {}
            if len(same_day) > 0:
                for c in _PS_AUX_COLS:
                    if c in same_day.columns:
                        v = pd.to_numeric(
                            same_day.iloc[0].get(c), errors='coerce')
                        if pd.notna(v):
                            aux_vals[c] = float(v)

            # Method A: Per-station ElasticNet regression
            pred_reg, conf_reg, r2_reg = _ps_regression(
                sdf, dws_col, date, aux_vals, target_name=target)
            if pred_reg is not None:
                n_reg += 1
                reg_r2s.append(r2_reg)

            # Method B: Per-station GradientBoosting
            pred_gb, conf_gb, r2_gb = _ps_gradient_boosting(
                sdf, dws_col, date, aux_vals, target_name=target)
            if pred_gb is not None:
                n_gb += 1
                gb_r2s.append(r2_gb)

            # Method C: Temporal interpolation
            pred_interp, conf_interp = _ps_temporal_interpolation(
                sdf, dws_col, date, k=10)
            if pred_interp is not None:
                n_interp += 1

            # Confidence-weighted blend of all available predictions
            candidates = []
            if pred_reg is not None:
                candidates.append((pred_reg, conf_reg))
            if pred_gb is not None:
                candidates.append((pred_gb, conf_gb))
            if pred_interp is not None:
                # Slightly downweight temporal interpolation vs regression
                candidates.append((pred_interp, conf_interp * 0.7))

            if candidates:
                total_conf = sum(c for _, c in candidates)
                if total_conf > 0:
                    blended = sum(p * c for p, c in candidates) / total_conf
                    preds[pos] = blended
                    confs[pos] = total_conf / len(candidates)
                else:
                    preds[pos] = np.mean([p for p, _ in candidates])
                    confs[pos] = 0.1
            else:
                n_miss += 1

        # Fill remaining NaN with station historical median
        for pos in range(len(sub)):
            if np.isnan(preds[pos]):
                station = sub.iloc[pos].get('_station')
                if pd.notna(station) and station in all_dws:
                    vals = pd.to_numeric(
                        all_dws[station].get(
                            dws_col, pd.Series(dtype=float)),
                        errors='coerce').dropna()
                    if len(vals) > 0:
                        preds[pos] = float(vals.median())
                        confs[pos] = 0.2  # low confidence for median fallback

        preds = np.clip(np.nan_to_num(preds, nan=0), 0, None)
        results[target] = preds
        confidences[target] = confs

        avg_reg_r2 = np.mean(reg_r2s) if reg_r2s else 0
        avg_gb_r2 = np.mean(gb_r2s) if gb_r2s else 0
        print(f"         reg={n_reg} (mean R2={avg_reg_r2:.3f})  "
              f"gb={n_gb} (mean R2={avg_gb_r2:.3f})  "
              f"interp={n_interp}  miss={n_miss}")
        print(f"         mean={preds.mean():.2f}  std={preds.std():.2f}  "
              f"min={preds.min():.2f}  max={preds.max():.2f}")
        print(f"         conf: mean={confs.mean():.3f}  "
              f"min={confs.min():.3f}  max={confs.max():.3f}")

    return results, confidences


# VARIANCE DECOMPRESSOR - fix global model prediction compression
class VarianceDecompressor:

    def __init__(self):
        self.station_stats_ = {}  # {station: (mean, std)}
        self.global_mean_ = 0.0
        self.global_std_ = 1.0

    def fit(self, all_dws, dws_col, exclude_dates=None):
        """Compute per-station target distribution stats."""
        if exclude_dates is None:
            exclude_dates = {}

        all_vals = []
        for stn, sdf in all_dws.items():
            vals = pd.to_numeric(sdf.get(dws_col, pd.Series(dtype=float)),
                                 errors='coerce')
            if stn in exclude_dates:
                keep = ~sdf['date'].dt.date.isin(exclude_dates[stn])
                vals = vals[keep]
            vals = vals.dropna()
            if len(vals) >= 5:
                self.station_stats_[stn] = (float(vals.mean()),
                                            float(vals.std()))
                all_vals.extend(vals.values)

        if all_vals:
            self.global_mean_ = float(np.mean(all_vals))
            self.global_std_ = max(float(np.std(all_vals)), 1e-8)
        return self

    def decompress(self, preds, stations):
        result = preds.copy()
        for stn in set(stations):
            if pd.isna(stn) or stn not in self.station_stats_:
                continue
            mask = np.array([s == stn for s in stations])
            if mask.sum() == 0:
                continue

            stn_mean, stn_std = self.station_stats_[stn]
            if stn_std < 1e-8:
                stn_std = self.global_std_

            stn_preds = preds[mask]
            pred_mean = stn_preds.mean()
            pred_std = max(stn_preds.std(), 1e-8)

            # Rescale to match station distribution
            scale = stn_std / pred_std
            # Cap the scale factor to avoid extreme amplification
            scale = min(scale, 5.0)
            result[mask] = stn_mean + (stn_preds - pred_mean) * scale

        return np.clip(result, 0, None)
