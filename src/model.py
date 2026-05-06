"""XGBoost model for agricultural yield forecasting."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURE_COLS, TARGET_COL, OUTPUTS_DIR, XGBOOST_PARAMS


def prepare_data(gdf):
    X = gdf[FEATURE_COLS].values
    y = gdf[TARGET_COL].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train(X_train, y_train):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    model = GradientBoostingRegressor(**XGBOOST_PARAMS)
    model.fit(X_tr, y_train)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUTS_DIR, "yield_model.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "yield_scaler.pkl"))
    print("Yield model trained and saved.")
    return model, scaler


def evaluate(model, scaler, X_test, y_test):
    X_ts = scaler.transform(X_test)
    y_pred = model.predict(X_ts)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"  MAE:  {mae:.1f} kg/ha")
    print(f"  RMSE: {rmse:.1f} kg/ha")
    print(f"  R²:   {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=15)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5)
    axes[0].set_xlabel("Actual Yield (kg/ha)")
    axes[0].set_ylabel("Predicted Yield (kg/ha)")
    axes[0].set_title(f"Actual vs Predicted (R²={r2:.3f})")

    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=40, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", lw=1.5, ls="--")
    axes[1].set_xlabel("Residual (kg/ha)")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "model_evaluation.png"), dpi=150)
    plt.close()
    return y_pred


def feature_importance_plot(model):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([FEATURE_COLS[i] for i in idx], importances[idx], color="forestgreen")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances — Yield Forecasting Model")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
