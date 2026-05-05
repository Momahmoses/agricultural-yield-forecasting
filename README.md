# Agricultural Yield Forecasting & Farmland Monitoring

Satellite imagery (NDVI/EVI from Sentinel-2/Landsat-8) combined with weather and soil data to predict crop yields and identify underperforming farmland across North-Central and Northwest Nigeria.

## Features
- NDVI and EVI computation from multi-spectral imagery
- LightGBM regression for yield prediction
- Underperforming farm detection (bottom 25th percentile)
- Interactive yield map with per-farm tooltips
- Crop-level performance breakdown

## Project Structure
```
agricultural-yield-forecasting/
├── src/
│   ├── data_loader.py     # NDVI/EVI computation, data generation
│   ├── model.py           # LightGBM training and prediction
│   └── visualize.py       # Yield maps and charts
├── data/raw/              # Satellite imagery, weather, soil CSVs
├── models/                # Saved model
├── outputs/               # Maps, reports, charts
├── config.yaml
├── main.py
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Data Sources
| Layer | Source |
|-------|--------|
| Satellite imagery | Sentinel-2 / Landsat-8 via Google Earth Engine |
| Weather | ERA5 Reanalysis (Copernicus) |
| Soil data | ISRIC World Soil Information |
| Farm boundaries | Open Street Map / FAO GeoNetwork |

## Output
- `outputs/yield_forecast_map.html` — interactive farm yield map
- `outputs/yield_forecast_report.csv` — full prediction results
- `outputs/underperforming_farms.csv` — flagged low-yield farms
- `outputs/yield_by_crop.png` — crop-level boxplot
- `outputs/feature_importance.png` — top predictors

## Author
MOMAH MOSES .C.
