# Agricultural Yield Forecasting & Farmland Monitoring

A GIS + ML system for predicting crop yields across North-Central and Northwest Nigeria using satellite NDVI, weather data, and soil properties.

## Overview

Combines Sentinel/Landsat-derived NDVI time series with climate and soil data to:
- Forecast crop yields (kg/ha) per farm using Gradient Boosting regression
- Monitor farmland health via weekly NDVI time series
- Identify underperforming farms for targeted intervention
- Generate interactive regional yield forecast maps

## Features

- **NDVI Analysis**: Weekly NDVI time series simulation for growing season monitoring
- **Yield Model**: GradientBoostingRegressor trained on 13 agronomic features
- **Farm Health Detection**: Classify farms as Excellent / Good / Moderate / Poor
- **Interactive Maps**: Folium choropleth yield maps per region
- **Regional Reports**: Yield forecast summaries by region and crop type

## Project Structure

```
agricultural-yield-forecasting/
├── src/
│   ├── data_ingestion.py     # Load/generate yield + NDVI + weather data
│   ├── model.py              # GradientBoosting training & evaluation
│   ├── ndvi_analysis.py      # NDVI time series analysis
│   └── visualization.py      # Yield maps and summary charts
├── data/sample/              # Synthetic training data
├── outputs/                  # Model outputs and visualizations
├── config.py
├── main.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Crops & Regions

| Region | State | Primary Crop |
|--------|-------|-------------|
| Kaduna | NW Nigeria | Maize |
| Katsina | NW Nigeria | Sorghum |
| Niger State | NC Nigeria | Rice |
| Benue | NC Nigeria | Yam |
| Kebbi | NW Nigeria | Wheat |

## Data Sources (Production)

- NDVI: Sentinel-2 MSI / Landsat 8-9 OLI (Google Earth Engine)
- Rainfall: CHIRPS v2.0
- Temperature: ERA5 Reanalysis (Copernicus CDS)
- Soil: ISRIC World Soil Database / SoilGrids 250m

## Author

**MOMAH MOSES .C.**  
Data Scientist & ML Engineer | [GitHub](https://github.com/Momahmoses)
