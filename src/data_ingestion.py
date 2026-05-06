"""Data ingestion: NDVI time series, weather, soil, and yield records."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SAMPLE_DIR, STUDY_REGIONS, FEATURE_COLS, TARGET_COL

CROP_BASE_YIELDS = {
    "maize": 2800, "sorghum": 1600, "rice": 3200, "yam": 8000, "wheat": 1800
}


def generate_synthetic_data(n_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = list(STUDY_REGIONS.keys())
    region_labels = rng.choice(regions, size=n_samples)

    lats, lons, crops = [], [], []
    for r in region_labels:
        info = STUDY_REGIONS[r]
        offset = 0.8
        lats.append(info["lat"] + rng.uniform(-offset, offset))
        lons.append(info["lon"] + rng.uniform(-offset, offset))
        crops.append(info["crop"])

    ndvi_mean = rng.uniform(0.2, 0.85, n_samples)
    ndvi_max = ndvi_mean + rng.uniform(0.0, 0.15, n_samples)
    ndvi_std = rng.uniform(0.02, 0.15, n_samples)
    rainfall = rng.normal(700, 250, n_samples).clip(100, 1800)
    temp_max = rng.normal(33, 4, n_samples).clip(22, 42)
    temp_min = temp_max - rng.uniform(8, 16, n_samples)
    soil_moisture = rng.uniform(0.1, 0.6, n_samples)
    solar_rad = rng.uniform(15, 28, n_samples)
    humidity = rng.uniform(30, 90, n_samples)
    elevation = rng.uniform(50, 900, n_samples)
    slope = rng.exponential(2, n_samples).clip(0, 20)
    soil_oc = rng.uniform(0.2, 3.5, n_samples)
    growing_days = rng.randint(90, 180, n_samples)

    base_yields = np.array([CROP_BASE_YIELDS[c] for c in crops], dtype=float)
    yield_kg = (
        base_yields
        * (0.5 + 0.5 * ndvi_mean)
        * (0.6 + 0.4 * (rainfall / 1800))
        * (0.7 + 0.3 * soil_moisture)
        * (1 - 0.3 * (slope / 20))
        * (0.8 + 0.2 * (soil_oc / 3.5))
        + rng.normal(0, 150, n_samples)
    ).clip(200, 15000)

    df = pd.DataFrame({
        "latitude": lats, "longitude": lons,
        "region": region_labels, "crop": crops,
        "ndvi_mean": ndvi_mean, "ndvi_max": ndvi_max, "ndvi_std": ndvi_std,
        "rainfall_mm": rainfall, "temperature_max_c": temp_max,
        "temperature_min_c": temp_min, "soil_moisture": soil_moisture,
        "solar_radiation": solar_rad, "relative_humidity": humidity,
        "elevation_m": elevation, "slope_deg": slope,
        "soil_organic_carbon": soil_oc, "growing_season_days": growing_days,
        TARGET_COL: yield_kg,
    })
    return df


def load_or_generate(filepath: str = None) -> gpd.GeoDataFrame:
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = generate_synthetic_data()
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        out = os.path.join(SAMPLE_DIR, "yield_data.csv")
        df.to_csv(out, index=False)
        print(f"Synthetic data saved → {out}")

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
