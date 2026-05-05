import os
from src.data_loader import load_config, generate_synthetic_dataset, flag_underperforming
from src.model import train, predict, save_model, feature_importance
from src.visualize import (
    plot_yield_map, plot_yield_by_crop,
    plot_underperforming, plot_feature_importance
)


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    config = load_config("config.yaml")
    print(f"[1/5] Config loaded — targets: {', '.join(config['targets'])}")

    df = generate_synthetic_dataset(n_samples=6000)
    print(f"[2/5] Dataset ready — {len(df)} farm records")

    model, metrics, _ = train(df, config)
    print(f"[3/5] Model trained — MAE: {metrics['mae']:.3f} t/ha | RMSE: {metrics['rmse']:.3f} | R²: {metrics['r2']:.4f}")
    save_model(model, config["output"]["model_path"])

    result_df = predict(model, df)
    result_df = flag_underperforming(result_df)
    result_df.to_csv(config["output"]["report"], index=False)

    under = result_df[result_df["underperforming"]]
    under.to_csv(config["output"]["underperforming"], index=False)
    print(f"[4/5] Predictions complete — {len(under)} underperforming farms flagged")

    fi_df = feature_importance(model)
    plot_feature_importance(fi_df, "outputs/feature_importance.png")
    plot_yield_by_crop(result_df, "outputs/yield_by_crop.png")
    plot_underperforming(result_df, "outputs/underperforming_map_static.png")
    plot_yield_map(result_df, config["output"]["yield_map"])
    print("[5/5] All visualizations saved to /outputs/")
    print("\nDone.")


if __name__ == "__main__":
    main()
