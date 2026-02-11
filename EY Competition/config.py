import os
# Paths
DATA_DIR = "./"
TRAIN_FILE = os.path.join(DATA_DIR, "water_quality_training_dataset.csv")
SAT_CACHE = os.path.join(DATA_DIR, "features_satellite.csv")
OSM_CACHE = os.path.join(DATA_DIR, "features_osm_terrain.csv")
IMPUTED_DATA = os.path.join(DATA_DIR, "water_quality_full_imputed.csv")

# Satellite Settings
SAT_WINDOW_DAYS = 60
SAT_CLOUD_MAX = 30
RETRY_WINDOW_DAYS = 120  # For rescue attempts
RETRY_CLOUD_MAX = 50     # For rescue attempts

# Model Settings
TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
