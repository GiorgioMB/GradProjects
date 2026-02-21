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
| **Satellite Imagery** | Landsat C2 L2 via [Planetary Computer](https://planetarycomputer.microsoft.com/) | Surface reflectance (green, red, NIR, SWIR16, SWIR22), spectral indices (NDVI, NDMI, MNDWI) |
| **Land Use (OSM)** | [OpenStreetMap](https://www.openstreetmap.org/) via osmnx | Counts of industrial sites, farmland, mines, wastewater plants within 5 km |
| **Land Cover** | [ESA WorldCover 10m](https://esa-worldcover.org/) via Planetary Computer | Fractional cover of 8 land classes (tree, shrub, grass, crop, built-up, bare, water, wetland), landscape diversity index |
| **Terrain** | [NASADEM](https://www.earthdata.nasa.gov/sensors/nasadem) via Planetary Computer | Elevation (mean, std), slope (mean, std), aspect |
| **Weather** | [Open-Meteo Archive API](https://open-meteo.com/) | 7/30-day rainfall, temperature, evapotranspiration, wind speed, humidity, solar radiation, water balance (rain − ET) |
| **Soil** | [ISRIC SoilGrids](https://soilgrids.org/) | Clay content, pH, cation exchange capacity *(API unreliable — handled gracefully)* |
| **Surface Water** | [JRC Global Surface Water](https://global-surface-water.appspot.com/) via Planetary Computer | Water occurrence (mean/max), water fraction, seasonality (months/year) |
| **Geology** | [Macrostrat](https://macrostrat.org/) | Lithology category (sedimentary/carbonate/metamorphic/igneous/unconsolidated), rock age (Ma), karst indicator |
| **Population** | ESA WorldCover built-up fraction (5 km buffer) | Population pressure proxy, urbanisation intensity |
| **Water Infrastructure** | [OpenStreetMap](https://www.openstreetmap.org/) via osmnx | Dam/weir count within 10 km, reservoir count, nearest dam distance, dam presence flag |
| **Water Body Type** | [OpenStreetMap](https://www.openstreetmap.org/) via osmnx | River/stream/lake/canal counts, dominant water body classification, flowing vs standing water |

On top of raw features, `engineer_features()` creates ~60 derived features:
- **Temporal** — cyclical month encoding (sin/cos), season (southern hemisphere), day of year
- **17 Spectral indices** — NDVI, NDMI, MNDWI, NBR, BSI, EVI, SAVI, NDWI, LSWI, TCW, TCB, TCG, NWI, WRI, AWEI_sh, AWEI_nsh, Green/Red ratio
- **Terrain derivatives** — elevation x slope, elevation x aspect, slope x aspect, terrain ruggedness (std x slope)
- **Weather interactions** — ET x temperature (aridity), water_balance x elevation, humidity x temperature (bioactivity), radiation x water (algal bloom proxy)
- **Land cover combinations** — cropland x rainfall (nutrient runoff), urban x rainfall, natural-vs-disturbed ratio
- **Geology interactions** — karst x rainfall (alkalinity driver), karst x water_occurrence
- **Water interactions** — water_fraction x elevation, dam_presence x rainfall, stream_density x rainfall
- **Nitrogen deposition proxies** — agricultural (cropland x rain x NDVI), urban (built-up x rain), combined N-pressure

Plus **51 KNN spatial features** from `SpatialKNNEncoder`:
- For each target, at k = {5, 10, 15, 25}: mean, median, std, distance-weighted mean of nearest neighbours
- Plus `nn_dist_mean` — average distance to 5 nearest neighbours
- KNN encoder is **refit inside each CV fold** to prevent spatial leakage

Plus **cross-target features** from `CrossTargetFeaturizer` (when inter-target correlation is detected):
- The pipeline first checks pairwise Pearson and Spearman correlations between all 3 targets
- If any pair has $|\text{Spearman  } \rho|>0.15$, it generates out-of-fold (OOF) predictions of each target using a lightweight Ridge model, then feeds those predictions as features for the *other* targets
- This exploits the physical relationship between water quality indicators (e.g. Alkalinity and Conductance are both driven by dissolved minerals) without leaking. OOF ensures row $i$'s cross-target feature was predicted by a model that never saw row $i$
- At test time, the Ridge models fitted on all training data predict each target first, then those predictions become features for the full ensemble

### Modelling

A custom $R^2$**-Weighted Ensemble** of five diverse base learners per target:

| Model | Role | NaN Handling |
|---|---|---|
| **XGBoost** | Primary gradient boosting (conservative: max_depth 3–4, min_child_weight 25–50, reg_lambda 8–15) | Native |
| **LightGBM** | Secondary gradient boosting for diversity | Native |
| **CatBoost** | Third gradient booster with ordered boosting for categorical-like features | Native |
| **ExtraTrees** | Extremely randomised trees for variance reduction | Pipeline-wrapped with median imputation + scaling |
| **Ridge** | Linear baseline for stability and diversity | Pipeline-wrapped with median imputation + scaling |

**Hyperparameter tuning:** Optuna (30 trials) with conservative search bounds:
- `max_depth`: 3–5
- `min_child_weight`: 20–60
- `reg_lambda` / `reg_alpha`: 3–20
- `learning_rate`: 0.005–0.05

**Ensemble weight determination:** Each model is evaluated via internal GroupKFold CV. Final ensemble weights are proportional to each model's $R^2$ — models with $R^2\le 0$ receive zero weight.

**Feature pruning:** An optional step (`prune_features()`) removes low-importance features via permutation importance.

### Evaluation Protocol

**Spatial cross-validation** with per-fold KNN refitting:

```
GroupKFold (5-fold, grouped by rounded lat/lon ≈ 1 km):
  For each fold:
    1. Refit KNN encoder on training split only
    2. Transform both train and test splits
    3. Train all 5 base models on train split
    4. Predict on held-out spatial test fold
    5. Report R^2 and train-test gap (overfitting monitor)
```


## 📊 EDA Report

An automated Exploratory Data Analysis report is generated as part of the pipeline (between imputation and modelling). It produces a self-contained HTML file (`eda_report.html`) with 9 sections:

1. **Dataset Overview** — shape, target coverage, missingness summary
2. **Target Distributions** — histograms and statistics for each target
3. **Feature–Target Correlations** — Spearman $\rho$ (top-30 per target + cross-target heatmap)
4. **Mutual Information** — nonlinear predictive power ranking
5. **Feature Distributions** — histograms of the top-12 most predictive features
6. **Spatial Analysis** — scatter maps of targets and station density
7. **Temporal Analysis** — monthly trends and seasonal boxplots
8. **Collinearity Diagnosis** — high-correlation pairs and heatmap
9. **Modellability Assessment** — quick RandomForest spatial CV to gauge signal strength

Can also be run standalone:
```bash
python eda_report.py --top 30
```

---

## Project Structure

```
EY Competition/
├── main.py                   #  Pipeline orchestrator (load -> enrich -> impute -> EDA -> model -> submit)
├── modeling.py               #  Feature engineering, models, ensemble, evaluation, Optuna tuning
├── data_fetch.py             #  Satellite, OSM, terrain data acquisition
├── fetch_climate_soil.py     #  Weather & soil API integration
├── fetch_geo_features.py     #  WorldCover, JRC water, geology, population, water infra
├── imputation.py             #  BayesianRidge IterativeImputer with missingness diagnosis
├── eda_report.py             #  Auto-generated HTML EDA report (9 sections)
└── config.py                 #  Paths, targets, dead features, satellite settings
├── run_parallel.sh           #  SLURM job script (64 cores, 256G RAM, 120h)
```

---

## How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost optuna joblib tqdm
pip install pystac-client planetary-computer odc-stac osmnx requests rasterio
pip install matplotlib seaborn scipy
```

### Local Execution

```bash
cd EY Competition/
python main.py
```

The pipeline is fully **resumable**, satellite and enrichment data are cached to disk. If interrupted, re-running picks up where it left off.

**Flags:**
- `--fast` — skip enrichment, use only base + satellite data (quick test)
- `--skip-fetch` — load pre-enriched data from `water_quality_post_enrichment.csv`


### Configuration

Edit `config.py` for paths and satellite settings:
- `TARGETS` — the three target columns
- `SAT_CACHE`, `OSM_CACHE`, `GEO_CACHE` — cache file paths
- `DEAD_FEATURES` — columns excluded from modelling (SoilGrids NaN, constant-zero land cover)
- `SAT_WINDOW_DAYS`, `SAT_CLOUD_MAX` — satellite query parameters

---

##  Pipeline Walkthrough

| Step | What happens |
|---|---|
| **1. Load Data** | Reads 9,319 training samples with 3 targets |
| **2. Satellite Merge** | Loads cached Landsat C2 L2 features via float-tolerant key matching (lat/lon rounded to 4 decimals). Fetches missing rows via Planetary Computer STAC API (8 threads, incremental saves). |
| **3. Enrichment** | Adds 11 data sources: OSM land-use, NASADEM terrain, Open-Meteo weather, SoilGrids soil, ESA WorldCover, JRC surface water, Macrostrat geology, population density, water infrastructure, water body type. All cached per-row. |
| **4. Imputation** | BayesianRidge IterativeImputer with missingness diagnosis. Saves imputer state for test-time consistency. Final median fallback for any residuals. |
| **4b. EDA Report** | Generates a self-contained HTML report with 9 analytical sections covering distributions, correlations, spatial/temporal patterns, collinearity, and modellability assessment. |
| **5. Modelling** | Engineers ~60 features -> adds 51 KNN spatial features (refit per fold) -> checks inter-target correlations and adds cross-target OOF features if exploitable -> Optuna tunes hyperparameters (30 trials) -> trains 5-model $R^2$-weighted ensemble per target -> spatial GroupKFold CV with train-test gap monitoring -> saves production models. |
| **6. Submission** | Fetches features for 200 test rows through the same pipeline, predicts with trained ensemble, inverts log1p, clips to $\ge$0, writes `submission.csv`. |

---

## Technical Highlights
- **Custom `WeightedEnsemble`** — $R^2$-proportional weighting with automatic exclusion of underperforming models, internal GroupKFold for weight determination
- **Custom `SpatialKNNEncoder`** — $k$-nearest-neighbour target features at multiple scales (k=5,10,15,25), refit inside each CV fold to prevent leakage, with automatic $k$-clamping for small folds
- **Custom `CrossTargetFeaturizer`** — OOF predictions of correlated targets as features, exploiting inter-target correlations without leaking; automatically disabled when correlations are too weak
- **Per-target regularisation** — XGBoost/LightGBM/CatBoost hyperparameters tuned independently per target with conservative Optuna bounds
- **Anti-overfitting monitoring** — train-test $R^2$ gap printed per fold; consistently high gaps trigger manual regularisation review
- **Feature pruning with safety floor** — permutation importance pruning
- **Incremental caching everywhere** — satellite, OSM/terrain, weather, and geo data are all cached and resumable
- **Production-ready model persistence** — trained ensembles + imputer state saved via `joblib` for deployment without retraining
- **Integrated EDA** — automated 9-section HTML report generated before modelling to diagnose data quality issues early

---

## 📬 About Me

I'm a PhD student and this was my first data science competition. I built this entire pipeline from scratch — from querying satellite APIs to designing a custom ensemble architecture. What started as a learning exercise became a fully-engineered, production-grade ML system with a strong focus on generalisation to unseen locations.

---

*Built with Python · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · Planetary Computer · Open-Meteo*

