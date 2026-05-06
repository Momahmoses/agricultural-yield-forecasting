"""Yield forecast maps and regional analysis plots."""

import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUTS_DIR, STUDY_REGIONS


def create_yield_map(gdf: gpd.GeoDataFrame, predictions: np.ndarray):
    center_lat = gdf["latitude"].mean()
    center_lon = gdf["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")

    y_min, y_max = predictions.min(), predictions.max()
    colormap = cm.get_cmap("RdYlGn")

    for (_, row), pred in zip(gdf.iterrows(), predictions):
        norm = (pred - y_min) / (y_max - y_min + 1e-9)
        r, g, b, _ = colormap(norm)
        color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"Region: {row['region']}<br>Crop: {row['crop']}<br>"
                f"Predicted: {pred:,.0f} kg/ha<br>NDVI: {row['ndvi_mean']:.2f}",
                max_width=220,
            ),
        ).add_to(m)

    for region, info in STUDY_REGIONS.items():
        folium.Marker(
            [info["lat"], info["lon"]],
            popup=f"<b>{region}</b> — {info['crop']}",
            icon=folium.Icon(color="green", icon="leaf"),
        ).add_to(m)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out = os.path.join(OUTPUTS_DIR, "yield_map.html")
    m.save(out)
    print(f"Yield map saved → {out}")


def region_yield_summary(gdf: gpd.GeoDataFrame, predictions: np.ndarray):
    gdf = gdf.copy()
    gdf["predicted_yield"] = predictions
    summary = gdf.groupby(["region", "crop"])["predicted_yield"].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["Region", "Crop", "Mean Yield (kg/ha)", "Std Dev", "Samples"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(summary["Region"], summary["Mean Yield (kg/ha)"],
                  yerr=summary["Std Dev"], capsize=5,
                  color=plt.cm.Set2(np.linspace(0, 1, len(summary))))
    ax.set_ylabel("Predicted Yield (kg/ha)")
    ax.set_title("Agricultural Yield Forecast by Region — North-Central & Northwest Nigeria")
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                row["Crop"], ha="center", fontsize=9, style="italic")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "region_yield_summary.png"), dpi=150)
    plt.close()
    print("Region yield summary chart saved.")
