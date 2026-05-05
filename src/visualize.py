import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from folium.plugins import HeatMap


def plot_yield_map(df, output_path, center=(10.5, 8.0)):
    m = folium.Map(location=list(center), zoom_start=7, tiles="CartoDB positron")

    max_yield = df["predicted_yield_tons_ha"].max()
    for _, row in df.iterrows():
        intensity = row["predicted_yield_tons_ha"] / max_yield
        color = f"#{int(255*(1-intensity)):02x}{int(200*intensity):02x}00"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Crop:</b> {row['crop_type']}<br>"
                f"<b>Predicted Yield:</b> {row['predicted_yield_tons_ha']:.2f} t/ha<br>"
                f"<b>NDVI:</b> {row['ndvi_mean']:.3f}",
                max_width=220
            )
        ).add_to(m)

    m.save(output_path)
    print(f"Yield map saved: {output_path}")


def plot_yield_by_crop(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = df.groupby("crop_type")["predicted_yield_tons_ha"].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x="crop_type", y="predicted_yield_tons_ha", order=order,
                palette="YlGn", ax=axes[0])
    axes[0].set_title("Predicted Yield by Crop Type")
    axes[0].set_xlabel("Crop")
    axes[0].set_ylabel("Yield (t/ha)")

    sns.scatterplot(data=df, x="ndvi_mean", y="predicted_yield_tons_ha",
                    hue="crop_type", alpha=0.4, ax=axes[1])
    axes[1].set_title("NDVI vs Predicted Yield")
    axes[1].set_xlabel("NDVI Mean")
    axes[1].set_ylabel("Yield (t/ha)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_underperforming(df, output_path):
    under = df[df["underperforming"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="longitude", y="latitude",
                    color="green", alpha=0.3, label="Normal", ax=ax)
    sns.scatterplot(data=under, x="longitude", y="latitude",
                    color="red", alpha=0.7, label="Underperforming", ax=ax)
    ax.set_title("Underperforming Farmland Locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(fi_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="YlOrRd_r", ax=ax)
    ax.set_title("Feature Importance — Yield Forecasting Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
