"""Main pipeline: Agricultural Yield Forecasting & Farmland Monitoring."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_ingestion import load_or_generate
from src.model import prepare_data, train, evaluate, feature_importance_plot
from src.ndvi_analysis import simulate_ndvi_timeseries, plot_ndvi_timeseries, underperforming_farms
from src.visualization import create_yield_map, region_yield_summary


def main():
    print("=" * 60)
    print("  Agricultural Yield Forecasting & Farmland Monitoring")
    print("  Target: North-Central & Northwest Nigeria")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    gdf = load_or_generate()
    print(f"  {len(gdf):,} records | Regions: {gdf['region'].nunique()} | Crops: {gdf['crop'].nunique()}")
    print(f"  Avg yield: {gdf['yield_kg_per_ha'].mean():,.0f} kg/ha")

    print("\n[2/5] Training XGBoost yield model...")
    X_train, X_test, y_train, y_test = prepare_data(gdf)
    model, scaler = train(X_train, y_train)

    print("\n[3/5] Evaluating model...")
    predictions = evaluate(model, scaler, X_test, y_test)
    feature_importance_plot(model)

    print("\n[4/5] NDVI time series analysis...")
    ndvi_df = simulate_ndvi_timeseries(n_farms=30)
    plot_ndvi_timeseries(ndvi_df)
    underperforming = underperforming_farms(ndvi_df)
    print(f"  Underperforming farms detected: {len(underperforming)}")
    if len(underperforming):
        print(underperforming.to_string(index=False))

    print("\n[5/5] Generating yield forecast maps...")
    test_gdf = gdf.iloc[len(X_train):].reset_index(drop=True)
    create_yield_map(test_gdf, predictions)
    region_yield_summary(test_gdf, predictions)

    print("\n✓ Pipeline complete. Outputs saved to ./outputs/")


if __name__ == "__main__":
    main()
