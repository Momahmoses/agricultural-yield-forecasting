import numpy as np
import pandas as pd
import geopandas as gpd
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_ndvi(nir, red):
    return (nir - red) / (nir - red + 1e-8)


def compute_evi(nir, red, blue, G=2.5, C1=6, C2=7.5, L=1):
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + 1e-8)


def generate_synthetic_dataset(n_samples=6000, seed=42):
    np.random.seed(seed)
    crops = ["maize", "sorghum", "millet", "rice", "groundnut"]
    base_yields = {"maize": 2.5, "sorghum": 1.8, "millet": 1.2, "rice": 3.5, "groundnut": 1.6}

    data = {
        "crop_type": np.random.choice(crops, n_samples),
        "ndvi_mean": np.random.uniform(0.1, 0.85, n_samples),
        "ndvi_peak": np.random.uniform(0.3, 0.95, n_samples),
        "evi_mean": np.random.uniform(0.05, 0.6, n_samples),
        "temperature_mean_c": np.random.normal(28, 4, n_samples),
        "temperature_max_c": np.random.normal(35, 5, n_samples),
        "rainfall_growing_season_mm": np.random.gamma(shape=4, scale=150, n_samples),
        "solar_radiation_mj": np.random.normal(18, 3, n_samples),
        "soil_nitrogen": np.random.uniform(0.05, 0.35, n_samples),
        "soil_ph": np.random.uniform(5.5, 7.5, n_samples),
        "soil_moisture": np.random.uniform(0.1, 0.5, n_samples),
        "farm_size_ha": np.random.lognormal(mean=0.5, sigma=0.8, size=n_samples),
        "irrigation_boolean": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        "fertilizer_kg_ha": np.random.exponential(scale=40, size=n_samples),
        "days_to_harvest": np.random.normal(120, 20, n_samples),
        "elevation_m": np.random.uniform(200, 900, n_samples),
        "latitude": np.random.uniform(9.0, 13.5, n_samples),
        "longitude": np.random.uniform(4.0, 14.0, n_samples),
    }
    df = pd.DataFrame(data)
    df["crop_type_code"] = pd.Categorical(df["crop_type"]).codes

    base = df["crop_type"].map(base_yields)
    noise = np.random.normal(0, 0.2, n_samples)
    yield_score = (
        base
        + 1.2 * df["ndvi_mean"]
        + 0.8 * df["ndvi_peak"]
        + 0.5 * df["evi_mean"]
        - 0.03 * np.abs(df["temperature_mean_c"] - 27)
        + 0.002 * df["rainfall_growing_season_mm"]
        + 0.5 * df["soil_nitrogen"]
        + 0.3 * df["irrigation_boolean"]
        + 0.005 * df["fertilizer_kg_ha"]
        - 0.0001 * df["elevation_m"]
        + noise
    )
    df["yield_tons_ha"] = np.clip(yield_score, 0.3, 8.0)
    return df


def flag_underperforming(df, threshold_percentile=25):
    threshold = df["yield_tons_ha"].quantile(threshold_percentile / 100)
    df["underperforming"] = df["yield_tons_ha"] < threshold
    return df
