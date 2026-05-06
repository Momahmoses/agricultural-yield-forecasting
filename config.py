import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

STUDY_REGIONS = {
    "Kaduna": {"lat": 10.5105, "lon": 7.4165, "crop": "maize"},
    "Katsina": {"lat": 12.9908, "lon": 7.6018, "crop": "sorghum"},
    "Niger_State": {"lat": 9.9309, "lon": 5.5983, "crop": "rice"},
    "Benue": {"lat": 7.3369, "lon": 8.7404, "crop": "yam"},
    "Kebbi": {"lat": 12.4539, "lon": 4.1975, "crop": "wheat"},
}

FEATURE_COLS = [
    "ndvi_mean", "ndvi_max", "ndvi_std",
    "rainfall_mm", "temperature_max_c", "temperature_min_c",
    "soil_moisture", "solar_radiation", "relative_humidity",
    "elevation_m", "slope_deg", "soil_organic_carbon",
    "growing_season_days",
]
TARGET_COL = "yield_kg_per_ha"

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
