#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Train the XGBoost RUL model used by the ``predict_rul`` tool of the ALM agent.

The DLI lab image ships two pre-trained artifacts that are *not* part of the open-source
GenerativeAIExamples repo:

    models/scaler_model.pkl      (sklearn StandardScaler, loaded with joblib)
    models/xgb_model_fd001.pkl   (xgboost.XGBRegressor, loaded with pickle)

This script recreates them from the ``train_FD001`` table of ``database/nasa_turbo.db`` so the
agent works end-to-end in Google Colab. It must be run with the *same* Python environment that
runs ``nat serve`` (pickles are version specific) -- ``colab_bootstrap.py`` takes care of that.

Usage:
    python train_rul_model.py [--db-path database/nasa_turbo.db] [--models-dir models]
"""

import argparse
import logging
import pickle
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Must match ``required_columns`` in predictors/predict_rul_tool.py
FEATURES = [
    "sensor_measurement_2", "sensor_measurement_3", "sensor_measurement_4",
    "sensor_measurement_7", "sensor_measurement_8", "sensor_measurement_11",
    "sensor_measurement_12", "sensor_measurement_13", "sensor_measurement_15",
    "sensor_measurement_17", "sensor_measurement_20", "sensor_measurement_21",
]
RUL_CAP = 125  # standard piece-wise linear RUL cap for C-MAPSS


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the XGBoost RUL model for the ALM agent")
    parser.add_argument("--db-path", default="database/nasa_turbo.db")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--force", action="store_true", help="retrain even if the artifacts exist")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    scaler_path = models_dir / "scaler_model.pkl"
    model_path = models_dir / "xgb_model_fd001.pkl"
    if scaler_path.exists() and model_path.exists() and not args.force:
        logger.info("RUL model artifacts already exist in %s (use --force to retrain)", models_dir)
        return 0

    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    if not Path(args.db_path).exists():
        logger.error("Database not found at %s -- run setup_database.py first", args.db_path)
        return 1

    with sqlite3.connect(args.db_path) as conn:
        df = pd.read_sql_query(f"SELECT unit_number, {', '.join(FEATURES)}, RUL FROM train_FD001", conn)
    logger.info("Loaded %d rows / %d engines from train_FD001", len(df), df.unit_number.nunique())

    # Hold out 20% of the engines for a quick sanity check
    units = np.sort(df.unit_number.unique())
    rng = np.random.default_rng(42)
    val_units = set(rng.choice(units, size=max(1, len(units) // 5), replace=False).tolist())
    train_df = df[~df.unit_number.isin(val_units)]
    val_df = df[df.unit_number.isin(val_units)]

    y_train = train_df["RUL"].clip(upper=RUL_CAP).values
    y_val = val_df["RUL"].clip(upper=RUL_CAP).values

    scaler = StandardScaler().fit(train_df[FEATURES].values)
    model = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.9, random_state=42, n_jobs=-1)
    model.fit(scaler.transform(train_df[FEATURES].values), y_train)
    rmse = float(np.sqrt(mean_squared_error(y_val, model.predict(scaler.transform(val_df[FEATURES].values)))))
    logger.info("Validation RMSE on held-out engines (RUL capped at %d): %.2f cycles", RUL_CAP, rmse)

    # Refit on all engines for the final artifacts
    scaler = StandardScaler().fit(df[FEATURES].values)
    model.fit(scaler.transform(df[FEATURES].values), df["RUL"].clip(upper=RUL_CAP).values)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved %s and %s", scaler_path, model_path)
    print(f"✅ RUL model trained (val RMSE {rmse:.1f} cycles) -> {models_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
