import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib


FEATURES = [
    "crop_type_code", "ndvi_mean", "ndvi_peak", "evi_mean",
    "temperature_mean_c", "temperature_max_c", "rainfall_growing_season_mm",
    "solar_radiation_mj", "soil_nitrogen", "soil_ph", "soil_moisture",
    "farm_size_ha", "irrigation_boolean", "fertilizer_kg_ha",
    "days_to_harvest", "elevation_m"
]
TARGET = "yield_tons_ha"


def train(df, config):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )

    model = lgb.LGBMRegressor(
        n_estimators=config["model"]["n_estimators"],
        learning_rate=config["model"]["learning_rate"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    return model, metrics, (X_test, y_test, y_pred)


def predict(model, df):
    df = df.copy()
    df["predicted_yield_tons_ha"] = model.predict(df[FEATURES])
    return df


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def feature_importance(model, top_n=10):
    fi = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(top_n)
    return fi
