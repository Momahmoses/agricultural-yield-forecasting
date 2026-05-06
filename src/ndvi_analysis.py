"""NDVI time series analysis and farmland health monitoring."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUTS_DIR


def simulate_ndvi_timeseries(n_farms: int = 20, n_weeks: int = 52, seed: int = 42):
    """Simulate weekly NDVI for crop growing seasons."""
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks)
    records = []
    for farm_id in range(n_farms):
        peak_week = rng.integers(14, 24)
        amplitude = rng.uniform(0.4, 0.7)
        base = rng.uniform(0.1, 0.25)
        ndvi = base + amplitude * np.exp(-0.5 * ((weeks - peak_week) / 6) ** 2)
        ndvi += rng.normal(0, 0.02, n_weeks)
        ndvi = ndvi.clip(0.05, 0.95)
        for w, v in enumerate(ndvi):
            records.append({"farm_id": farm_id, "week": w, "ndvi": v,
                            "status": classify_farm_health(v, w, peak_week)})
    return pd.DataFrame(records)


def classify_farm_health(ndvi: float, week: int, peak_week: int) -> str:
    if week < 8 or week > 40:
        return "off_season"
    if ndvi > 0.6:
        return "excellent"
    elif ndvi > 0.45:
        return "good"
    elif ndvi > 0.3:
        return "moderate"
    return "poor"


def plot_ndvi_timeseries(df: pd.DataFrame, n_farms: int = 5):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_farms))
    for i, farm_id in enumerate(df["farm_id"].unique()[:n_farms]):
        farm_df = df[df["farm_id"] == farm_id]
        ax.plot(farm_df["week"], farm_df["ndvi"], color=colors[i],
                label=f"Farm {farm_id}", alpha=0.8, lw=1.5)
    ax.axhline(0.45, color="orange", ls="--", lw=1, label="Good threshold (0.45)")
    ax.axhline(0.3, color="red", ls="--", lw=1, label="Poor threshold (0.30)")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("NDVI")
    ax.set_title("NDVI Time Series — Farmland Health Monitoring")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUTS_DIR, "ndvi_timeseries.png"), dpi=150)
    plt.close()
    print("NDVI time series plot saved.")


def underperforming_farms(df: pd.DataFrame) -> pd.DataFrame:
    growing = df[(df["week"] >= 8) & (df["week"] <= 40)]
    farm_stats = growing.groupby("farm_id")["ndvi"].agg(["mean", "min", "max"])
    underperforming = farm_stats[farm_stats["mean"] < 0.35].reset_index()
    return underperforming
