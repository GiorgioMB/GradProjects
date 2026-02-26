import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    print("   Warning: statsmodels not installed — SARIMAX models will be unavailable.")
    HAS_STATSMODELS = False

from dws_data import (
    DWS_COL_MAP, DWS_AUX_COLS,
    coord_to_station, _COL_FOR_TARGET
)


# EXOGENOUS VARIABLE CANDIDATES (auxiliary DWS measurements)
EXOG_CANDIDATES = {
    "Total Alkalinity": [
        "pH_Diss_Water", "Ca_Diss_Water", "Mg_Diss_Water",
        "EC_Phys_Water", "Cl_Diss_Water", "SO4_Diss_Water",
        "Na_Diss_Water", "Si_Diss_Water",
    ],
    "Electrical Conductance": [
        "pH_Diss_Water", "Ca_Diss_Water", "Mg_Diss_Water",
        "TAL_Diss_Water", "Cl_Diss_Water", "SO4_Diss_Water",
        "Na_Diss_Water", "K_Diss_Water",
    ],
    "Dissolved Reactive Phosphorus": [
        "pH_Diss_Water", "NH4_N_Diss_Water", "NO3_NO2_N_Diss_Water",
        "P_Tot_Water", "EC_Phys_Water", "DMS_Tot_Water",
    ],
}


def _prepare_station_series(sdf, target, cutoff_date, exog_cols=None):
    dws_col = _COL_FOR_TARGET.get(target)
    if dws_col is None or dws_col not in sdf.columns:
        return None, None, None

    df = sdf[sdf["date"] <= cutoff_date].copy()
    df = df.dropna(subset=[dws_col])
    if len(df) < 24:  # Need at least 2 years of data
        return None, None, None

    df = df.set_index("date").sort_index()

    # Resample to monthly (mean aggregation)
    y = df[dws_col].resample("MS").mean().dropna()
    if len(y) < 24:
        return None, None, None

    # Exogenous variables (also monthly-resampled)
    exog = None
    if exog_cols:
        available = [c for c in exog_cols if c in df.columns]
        if available:
            exog = df[available].resample("MS").mean()
            # Only keep columns with >60% non-NaN
            good = exog.columns[exog.notna().mean() > 0.6]
            if len(good) > 0:
                exog = exog[good].reindex(y.index)
                # Forward-fill then backward-fill small gaps
                exog = exog.ffill(limit=3).bfill(limit=3)
                # Drop any remaining columns with NaN (SARIMAX can't handle it)
                exog = exog.dropna(axis=1)
                if exog.shape[1] == 0:
                    exog = None
            else:
                exog = None

    return y, exog, y.index


def _fit_sarimax_single(y, exog, order, seasonal_order, max_iter=200):
    try:
        model = SARIMAX(
            y,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=max_iter,
                           method='lbfgs', low_memory=True)
        return result, result.aic
    except Exception:
        return None, np.inf


def _optuna_sarimax(y, exog, target, station, n_trials=30):
    n = len(y)
    train_end = int(n * 0.7)
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:]
    exog_train = exog.iloc[:train_end] if exog is not None else None
    exog_val = exog.iloc[train_end:] if exog is not None else None

    if len(y_val) < 3:
        # Not enough validation data, use default
        return (1, 1, 1), (1, 1, 0, 12), True

    best_result = {"aic": np.inf, "order": (1, 1, 1),
                   "seasonal": (1, 1, 0, 12), "use_exog": exog is not None}

    def objective(trial):
        p = trial.suggest_int("p", 0, 2)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 2)
        P = trial.suggest_int("P", 0, 1)
        D = trial.suggest_int("D", 0, 1)
        Q = trial.suggest_int("Q", 0, 1)
        use_exog = trial.suggest_categorical("use_exog",
                                              [True, False]) if exog is not None else False

        order = (p, d, q)
        seasonal = (P, D, Q, 12)
        ex = exog_train if use_exog else None
        ex_val = exog_val if use_exog else None

        try:
            model = SARIMAX(
                y_train, exog=ex,
                order=order, seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=150, method='lbfgs',
                            low_memory=True)

            # Forecast validation period
            fc = res.forecast(steps=len(y_val), exog=ex_val)
            fc = np.nan_to_num(np.array(fc, dtype=np.float64),
                               nan=0.0, posinf=0.0, neginf=0.0)
            fc = np.clip(fc, 0, None)
            y_v = y_val.values

            rmse = np.sqrt(np.mean((y_v - fc) ** 2))

            if np.isnan(rmse) or np.isinf(rmse):
                return 1e10
            return rmse

        except Exception:
            return 1e10

    if HAS_OPTUNA:
        study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        bp = study.best_params
        order = (bp["p"], bp["d"], bp["q"])
        seasonal = (bp["P"], bp["D"], bp["Q"], 12)
        use_exog = bp.get("use_exog", False)
        return order, seasonal, use_exog
    else:
        return (1, 1, 1), (1, 1, 0, 12), exog is not None


# PER-STATION SARIMAX FITTING
class StationSARIMAX:

    def __init__(self, targets, n_optuna_trials=20):
        self.targets = targets
        self.n_optuna_trials = n_optuna_trials
        self.models_ = {}     # (station, target) -> fitted result
        self.configs_ = {}    # (station, target) -> (order, seasonal, use_exog)
        self.exog_cols_ = {}  # (station, target) -> list of exog column names
        self.y_last_ = {}     # (station, target) -> last y value (for fallback)

    def fit(self, all_dws, cutoff_date=None):
        if not HAS_STATSMODELS:
            print("   Warning: statsmodels not installed — skipping SARIMAX.")
            return self

        if cutoff_date is None:
            cutoff_date = pd.Timestamp("2015-12-31")

        print("\n== SARIMAX Per-Station Models ==")
        n_fitted, n_failed = 0, 0

        for station in sorted(all_dws.keys()):
            sdf = all_dws[station]

            for target in self.targets:
                exog_candidates = EXOG_CANDIDATES.get(target, [])
                y, exog, idx = _prepare_station_series(
                    sdf, target, cutoff_date, exog_candidates)

                if y is None:
                    continue

                # Store last known value as fallback
                self.y_last_[(station, target)] = float(y.iloc[-1])

                # Optuna search
                try:
                    order, seasonal, use_exog = _optuna_sarimax(
                        y, exog, target, station,
                        n_trials=self.n_optuna_trials)
                except Exception:
                    order = (1, 1, 1)
                    seasonal = (1, 1, 0, 12)
                    use_exog = False

                # Fit final model on ALL available data
                ex = exog if use_exog else None
                result, aic = _fit_sarimax_single(y, ex, order, seasonal)

                if result is not None:
                    self.models_[(station, target)] = result
                    self.configs_[(station, target)] = (order, seasonal, use_exog)
                    if use_exog and exog is not None:
                        self.exog_cols_[(station, target)] = list(exog.columns)
                    else:
                        self.exog_cols_[(station, target)] = []
                    n_fitted += 1
                else:
                    n_failed += 1

        print(f"   SARIMAX: {n_fitted} fitted, {n_failed} failed "
              f"across {len(all_dws)} stations x {len(self.targets)} targets")
        return self

    def predict_for_rows(self, df, all_dws):
        if not HAS_STATSMODELS:
            return {t: np.full(len(df), np.nan) for t in self.targets}

        # Parse dates
        if "date" not in df.columns or df["date"].dtype == object:
            if "Sample Date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["Sample Date"], dayfirst=True)

        predictions = {t: np.full(len(df), np.nan) for t in self.targets}

        for i, row in df.iterrows():
            stn = row.get("_dws_station", None)
            if stn is None:
                stn = coord_to_station(row["Latitude"], row["Longitude"])
            if stn is None:
                continue

            pos = df.index.get_loc(i)
            cur_date = pd.Timestamp(row["date"])
            # Round to month start for SARIMAX alignment
            cur_month = cur_date.to_period("M").to_timestamp()

            for target in self.targets:
                key = (stn, target)
                if key not in self.models_:
                    # Fallback: use last known value
                    if key in self.y_last_:
                        predictions[target][pos] = self.y_last_[key]
                    continue

                result = self.models_[key]
                try:
                    # Get the last date in the fitted model
                    last_fitted = result.model.data.dates[-1]
                    steps = max(1, ((cur_month.year - last_fitted.year) * 12
                                    + cur_month.month - last_fitted.month))

                    # Prepare exogenous for forecast period if needed
                    exog_cols = self.exog_cols_.get(key, [])
                    exog_fc = None
                    if exog_cols and stn in all_dws:
                        sdf = all_dws[stn]
                        # Try to get exog values for forecast period
                        fc_dates = pd.date_range(
                            last_fitted + pd.DateOffset(months=1),
                            periods=steps, freq="MS")
                        exog_vals = []
                        for d in fc_dates:
                            month_data = sdf[
                                (sdf["date"].dt.to_period("M") ==
                                 d.to_period("M"))
                            ]
                            if len(month_data) > 0:
                                vals = month_data[exog_cols].mean()
                            else:
                                # Use overall station mean as fallback
                                vals = sdf[exog_cols].mean()
                            exog_vals.append(vals)
                        exog_fc = pd.DataFrame(exog_vals, index=fc_dates)
                        exog_fc = exog_fc.fillna(exog_fc.mean())
                        # If still NaN, drop exog
                        if exog_fc.isna().any().any():
                            exog_fc = None

                    fc = result.forecast(steps=steps, exog=exog_fc)
                    pred = float(fc.iloc[-1])

                    # Sanitise
                    if np.isnan(pred) or np.isinf(pred) or pred < 0:
                        if key in self.y_last_:
                            pred = self.y_last_[key]
                        else:
                            pred = np.nan
                    predictions[target][pos] = pred

                except Exception:
                    # Fallback to last known value
                    if key in self.y_last_:
                        predictions[target][pos] = self.y_last_[key]

        # Report coverage
        for target in self.targets:
            n_valid = np.sum(~np.isnan(predictions[target]))
            print(f"   SARIMAX {target}: {n_valid}/{len(df)} predictions")

        return predictions

    def evaluate_cv(self, all_dws, cutoff_date=None):
        if not HAS_STATSMODELS:
            return {}

        if cutoff_date is None:
            cutoff_date = pd.Timestamp("2010-12-31")

        print("\n   SARIMAX time-series CV...")
        results = {}

        for target in self.targets:
            dws_col = _COL_FOR_TARGET.get(target)
            if dws_col is None:
                continue

            all_actual, all_pred = [], []

            for station in sorted(all_dws.keys()):
                key = (station, target)
                if key not in self.models_:
                    continue

                sdf = all_dws[station]
                # Get actual values in the 12 months after cutoff
                val_start = cutoff_date + pd.DateOffset(days=1)
                val_end = cutoff_date + pd.DateOffset(months=12)
                actual = sdf[
                    (sdf["date"] >= val_start) &
                    (sdf["date"] <= val_end) &
                    sdf[dws_col].notna()
                ]

                if len(actual) == 0:
                    continue

                # Monthly average of actuals
                actual_monthly = actual.set_index("date")[dws_col].resample("MS").mean().dropna()

                result = self.models_[key]
                try:
                    steps = len(actual_monthly)
                    fc = result.forecast(steps=steps)
                    fc = np.nan_to_num(np.array(fc, dtype=np.float64),
                                       nan=0.0, posinf=0.0, neginf=0.0)
                    fc = np.clip(fc, 0, None)

                    all_actual.extend(actual_monthly.values[:len(fc)])
                    all_pred.extend(fc[:len(actual_monthly)])
                except Exception:
                    pass

            if len(all_actual) >= 5:
                from sklearn.metrics import r2_score, mean_squared_error
                actual_arr = np.array(all_actual)
                pred_arr = np.array(all_pred)
                r2 = r2_score(actual_arr, pred_arr)
                rmse = np.sqrt(mean_squared_error(actual_arr, pred_arr))
                results[target] = {"r2": r2, "rmse": rmse,
                                    "n_points": len(all_actual)}
                print(f"   {target}: SARIMAX CV R2={r2:.4f}  "
                      f"RMSE={rmse:.1f}  (n={len(all_actual)})")
            else:
                print(f"   {target}: insufficient validation data")

        return results

def blend_predictions(tree_preds, sarimax_preds, blend_weight=0.7):
    final = tree_preds.copy()
    has_sarimax = ~np.isnan(sarimax_preds)

    if has_sarimax.any():
        final[has_sarimax] = (
            (1 - blend_weight) * tree_preds[has_sarimax]
            + blend_weight * sarimax_preds[has_sarimax]
        )
        n_blended = has_sarimax.sum()
        print(f"      Blended {n_blended}/{len(final)} rows "
              f"(SARIMAX weight={blend_weight:.1%})")

    return final
