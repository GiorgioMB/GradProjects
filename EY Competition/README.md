# Optimizing Clean Water Supply — EY Open Science Data Challenge

> **My first data science competition as a PhD student.**
> Built end-to-end in Python, from raw satellite imagery to production-ready ensemble predictions.

---

## The Challenge

The [EY Open Science Data Challenge](https://challenge.ey.com/challenges/2026-optimizing-clean-water-supply/overview) tasks participants with predicting water quality indicators for monitoring stations across **South Africa**, using only the station's coordinates and sampling date.

**Three regression targets:**
| Target | What it measures |
|---|---|
| **Total Alkalinity** | Water's acid-neutralising capacity (mg/L CaCO₃) |
| **Electrical Conductance** | Dissolved ion concentration (µS/cm) |
| **Dissolved Reactive Phosphorus** | Bioavailable phosphorus driving eutrophication (µg/L) |

**The constraint:** At inference time, `(Latitude, Longitude, Sample Date)` are received, nothing else. Every feature must be derived from publicly available geospatial data.

---

## Approach & Key Design Decisions

This pipeline constructs a rich feature space from **five independent geospatial data sources**, then trains a **performance-weighted multi-model ensemble** with spatial cross-validation.

### Feature Engineering (50+ features from 5 sources)

| Source | API / Dataset | Features Derived |
|---|---|---|
|  **Satellite Imagery** | Landsat C2 L2 via [Planetary Computer](https://planetarycomputer.microsoft.com/) | Surface reflectance (green, red, NIR, SWIR), spectral indices (NDVI, NDMI, MNDWI), band ratios (NIR/Green, SWIR ratio, NBR) |
|  **Land Use** | [OpenStreetMap](https://www.openstreetmap.org/) via osmnx | Counts of industrial sites, farmland, mines, wastewater plants within 5 km |
|  **Terrain** | [NASADEM](https://www.earthdata.nasa.gov/sensors/nasadem) via Planetary Computer | Elevation, slope, aspect, valley depth, elevation x slope interactions |
|  **Weather** | [Open-Meteo Archive API](https://open-meteo.com/) | 7-day & 30-day rainfall accumulation, 30-day mean temperature, rain intensity ratio |
|  **Soil** | [ISRIC SoilGrids](https://soilgrids.org/) | Clay content, pH, cation exchange capacity *(API unreliable — handled gracefully)* |

On top of raw features, the pipeline engineers:
- **Temporal features** — cyclical month encoding, season, day of year
- **Spatial features** — rounded lat/lon clusters, lat x lon interaction
- **Per-location temporal lags** — lag-1, diff-1, rolling-3-mean on sensor & weather data per monitoring station, days since last measurement, observation count
- **Spectral interactions** — NIR/Green ratio, SWIR ratio, NBR, Green/Red ratio
- **Cross-domain interactions** — elevation x slope, elevation x rainfall

### Modelling

A **Performance-Weighted Ensemble** of four diverse base learners:

| Model | Role | NaN Handling |
|---|---|---|
| **XGBoost** | Primary gradient boosting (3000 trees, tuned) | Native |
| **LightGBM** | Secondary gradient boosting for diversity | Native |
| **TabNet** | Attention-based deep learning for tabular data | Custom sklearn wrapper with internal imputation & standardisation |
| **Random Forest** | Bagging diversity | Pipeline-wrapped with median imputation |

**Weight determination:** During training, each model is evaluated via internal cross-validation. Weights are set proportional to each model's $R^2$: for the $i$-th model, $w_i = \min\left\\{0, R^2_i\right\\}$. This means the ensemble automatically drops any model that underperforms predicting the mean.

### Evaluation Protocol

**Nested spatial cross-validation** — a rigorous evaluation avoiding both spatial leakage and weight-selection bias:

```
Outer loop (5-fold GroupKFold by location):
  -> Unbiased performance estimation
  
  Inner loop (5-fold GroupKFold on training split):
    -> Determines per-model weights for the ensemble
    -> Refits all models on full training split
  
  -> Predict on held-out spatial test fold
```

Groups are defined by rounding coordinates to 2 decimal places (~1 km), ensuring nearby stations are never split across train and test.

- **No spatial leakage** — GroupKFold ensures entire monitoring stations are held out, not individual observations
- **Temporal lag safety** — lag features use only past non-target measurements, safe for inference
- **Robust to missing data** — MICE imputation with Random Forest estimator, plus median fallback for any remaining NaNs
- **Log-transform as a flag** — configurable `LOG_TRANSFORM` toggle for experimentation (defaults to raw scale)
- **NaN monitoring** — the pipeline prints warnings whenever `log1p`/`expm1` produces NaN values

---

## Project Structure

```
EY Competition/
├── main.py                   #  Pipeline orchestrator (Steps 1–7)
├── modeling.py               #  Feature engineering, models, ensemble, evaluation
├── data_fetch.py             #  Satellite, OSM, terrain data acquisition
├── fetch_climate_soil.py     #  Weather & soil API integration
├── imputation.py             #  MICE imputation with missingness diagnosis
└── config.py                 #  Paths, targets, satellite settings
```

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost lightgbm joblib tqdm
pip install pystac-client planetary-computer odc-stac osmnx requests

# Optional
pip install pytorch-tabnet
```

### Local Execution

```bash
cd comp/
python main.py
```

The pipeline is fully **resumable** — satellite and enrichment data are cached to disk. If interrupted, re-running picks up where it left off.


### Configuration

Edit `config.py` for paths and satellite settings. In `main.py`, toggle:

```python
LOG_TRANSFORM = False  # Set True to train on log1p(y), False for raw scale
```

---

##  Pipeline Walkthrough

| Step | What happens |
|---|---|
| **1. Load Data** | Reads 9,320 training samples with 3 targets |
| **2. Satellite Fetch** | Queries Landsat C2 L2 via Planetary Computer STAC API for each (lat, lon, date). Computes median composite within a 60-day window. Extracts surface reflectance bands and spectral indices. Fully parallelised (8 threads) with incremental CSV saves. |
| **3. Rescue** | Re-attempts failed satellite downloads with relaxed cloud cover and wider time windows. |
| **4. Enrichment** | Adds OSM land-use counts, NASADEM terrain features, Open-Meteo weather history, and SoilGrids soil chemistry for each location+date. Cached per-row. |
| **5. Imputation** | Diagnoses missingness pattern (MAR test via AUC), then applies MICE (Iterative Imputer with RF estimator). Final median fallback for any residuals. |
| **6. Modelling** | Engineers 50+ features, trains a performance-weighted ensemble per target and saves production models to disk. |
| **7. Submission** | Fetches features for 201 test rows through the same pipeline, predicts with trained models, and writes `submission.csv`. |

---

## Technical Highlights

- **Custom `PerformanceWeightedEnsemble`**: a fully sklearn-compatible estimator with `fit`/`predict` interface, internal CV for adaptive weighting, and automatic exclusion of underperforming models
- **Custom `SklearnTabNetRegressor`**: wraps pytorch-tabnet for sklearn compatibility, with built-in NaN imputation, standardisation, and early stopping
- **Spatial CV with proper nesting**: outer folds for metrics, inner folds for weight tuning, all grouped by geographic location
- **Incremental caching everywhere**: satellite fetches, OSM/terrain enrichment, and weather data are all cached and resumable. The full pipeline can be interrupted and restarted without losing progress.
- **Production-ready model persistence**: trained ensembles saved via `joblib` for deployment without retraining
