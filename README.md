# Agricultural Yield Forecasting & Farmland Monitoring

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Satellite imagery (NDVI/EVI from Sentinel-2/Landsat-8) combined with weather and soil data to predict crop yields and identify underperforming farmland across North-Central and Northwest Nigeria.

---

## Problem Statement

Nigeria imports food worth billions of naira annually despite having vast agricultural land. Yield gaps from poor crop management and climate stress go undetected without satellite monitoring. This system provides per-farm yield forecasts and anomaly detection to guide agronomic interventions.

---

## Features

| Feature | Description |
|---------|-------------|
| NDVI & EVI Computation | Multi-spectral index extraction from Sentinel-2 / Landsat-8 |
| LightGBM Yield Regression | Per-farm yield prediction with feature importance |
| Underperforming Farm Detection | Bottom 25th percentile flagging for targeted intervention |
| Interactive Yield Map | Folium map with per-farm tooltips |
| Crop-Level Breakdown | Performance comparison across 7 crops |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Remote Sensing | Sentinel-2, Landsat-8 (NDVI, EVI) |
| Machine Learning | LightGBM, scikit-learn |
| Geospatial | GeoPandas, Folium, Rasterio |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |

---

## Project Structure

```
agricultural-yield-forecasting/
├── src/
│   ├── data_loader.py     # NDVI/EVI computation and data pipeline
│   ├── model.py           # LightGBM training and prediction
│   └── visualize.py       # Yield maps and performance charts
├── data/raw/              # Satellite imagery, weather, soil CSVs
├── models/                # Saved LightGBM model
├── config.yaml            # Crop parameters, thresholds, region config
├── main.py                # Pipeline entry point
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/agricultural-yield-forecasting.git
cd agricultural-yield-forecasting
pip install -r requirements.txt
python main.py
```

---

## Data Sources

- Sentinel-2 Level-2A surface reflectance (ESA Copernicus)
- Landsat-8 Collection 2 imagery (USGS)
- FAO HarvestChoice soil and terrain data
- NIMET growing season rainfall and temperature data
- CBN / NAERLS crop yield surveys

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
