# Optimizing Clean Water Supply: Geospatial Machine Learning Pipeline
**EY Open Science Data Challenge**

This repository contains an end-to-end machine learning pipeline designed to predict water quality indicators (Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus) for monitoring stations across South Africa. 

Operating under inference constraints, requiring only spatial coordinates and sampling dates, this system dynamically constructs a high-dimensional feature space from dispersed geospatial APIs and synthesizes predictions through a spatially-aware, multi-layer stacking ensemble.

---

## Project Impact & Technical Scope

Predicting water quality without in-situ chemical testing requires robust proxy modeling. This project:
* **Asynchronous Data Ingestion:** Orchestrates concurrent retrieval of multidimensional satellite imagery (Landsat C2 L2) and terrain data via Planetary Computer STAC APIs.
* **Leakage-Proof Spatial Modeling:** Implements a custom $k$-Nearest-Neighbor spatial encoder that refits dynamically inside cross-validation folds to prevent spatial data leakage.
* **Hybrid Ensemble Architecture:** Combines gradient boosting (XGBoost, LightGBM, CatBoost) with a PyTorch-based Long Short-Term Memory (LSTM) network to capture both non-linear spatial interactions and deep temporal dynamics.
* **Automated Diagnostics:** Integrates adversarial validation to detect train-test distribution shifts and generates comprehensive HTML EDA reports dynamically.

---

## Repository Architecture & File Index

The codebase is modularized to separate data acquisition, imputation, modeling, and evaluation protocols.

### Core Orchestration
* **`main.py`** The primary execution orchestrator. Manages the pipeline state machine from raw data loading to final submission generation. Has robust caching as well as CLI flags (`--fast`, `--skip-fetch`, `--per-station`) for flexible execution environments.
* **`config.py`** Centralized configuration matrix defining target variables, static file paths, API retry thresholds, and satellite cloud-cover tolerances.

### Data Acquisition & Feature Engineering
* **`data_fetch.py`** Manages direct connections to the Microsoft Planetary Computer. Extracts surface reflectance and computes spectral indices (NDVI, NDMI, MNDWI) while resolving local OpenStreetMap land-use topologies.
* **`fetch_geo_features.py`** Retrieves and aggregates localized spatial vectors, including ESA WorldCover fractions, JRC Global Surface Water seasonality, Macrostrat geological lithology, and population density metrics.
* **`fetch_climate_soil.py`** Integrates historical meteorological variables via Open-Meteo (precipitation, evapotranspiration, radiation) and cross-references ISRIC SoilGrids for subterranean chemical properties.
* **`dws_data.py`** Scrapes the South African Department of Water and Sanitation (DWS) repository. Extracts auxiliary chemistry variables, computes spatial neighbor aggregations, and builds augmented training sets while enforcing strict temporal leakage prevention.

### Machine Learning Engine
* **`modeling.py`** The predictive core. Highlights include:
    * **`StationLSTMModel`**: A PyTorch neural network that learns historical temporal dynamics at individual monitoring stations.
    * **`SpatialKNNEncoder`**: Computes local chemistry averages using Haversine distance weighting, rigorously isolating training data within CV folds.
    * **`StackingEnsemble` & `MultiSeedEnsemble`**: A Level-2 RidgeCV meta-learner that synthesizes out-of-fold predictions from diverse base models (XGBoost, LightGBM, ExtraTrees), wrapped in a multi-seed variance reduction layer.
    * **Hyperparameter Optimization**: Automated Optuna trials for gradient boosters, leveraging spatial `GroupKFold` cross-validation.
* **`imputation.py`** Executes multivariate missing-value imputation utilizing `BayesianRidge` within an `IterativeImputer`. Persists the trained imputer state (`imputer_state.joblib`) to ensure consistent data distributions during inference.

### Analysis & Reporting
* **`eda_report.py`** An automated analytical engine that generates a self-contained, 9-section HTML report. Visualizes target distributions, mutual information rankings, PCA embeddings, spatial density hex-maps, and hierarchical collinearity dendrograms.

---

## Technology Stack

* **Core ML:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch` (PyTorch), `optuna`
* **Geospatial & APIs:** `pystac-client`, `planetary-computer`, `rasterio`, `osmnx`, `shapely`
* **Data Processing:** `pandas`, `numpy`, `scipy`, `joblib`

---

## Execution

The pipeline is fully resumable; API retrievals are incrementally cached to disk. 

**Standard Execution (Full Pipeline):**
```bash
python main.py
