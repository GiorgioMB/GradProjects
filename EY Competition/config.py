import os

# Paths
DATA_DIR   = "./"
TRAIN_FILE = os.path.join(DATA_DIR, "water_quality_training_dataset.csv")
SAT_CACHE  = os.path.join(DATA_DIR, "features_satellite.csv")
OSM_CACHE  = os.path.join(DATA_DIR, "features_osm_terrain.csv")
GEO_CACHE  = os.path.join(DATA_DIR, "features_geo.csv")
IMPUTED_DATA = os.path.join(DATA_DIR, "water_quality_full_imputed.csv")
DWS_DIR    = os.path.join(DATA_DIR, "dws_data")

# Satellite settings
SAT_WINDOW_DAYS   = 60
SAT_CLOUD_MAX     = 30
RETRY_WINDOW_DAYS = 120
RETRY_CLOUD_MAX   = 50

# Model settings
TARGETS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus',
]

# Feature groups (for reference)
DEAD_FEATURES = [
    'soil_clay', 'soil_phh2o', 'soil_cec',           # SoilGrids 99.9 % NaN
    'lc_snow_ice', 'lc_mangroves', 'lc_moss_lichen', # Always 0 in South Africa
]

EXTENDED_WEATHER = [
    'et_30d_sum', 'et_7d_sum', 'water_balance_30d',
    'wind_30d_mean', 'humidity_30d_mean', 'radiation_30d_mean',
]
