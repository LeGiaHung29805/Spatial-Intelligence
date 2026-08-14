import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from sqlalchemy import text


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.CSDL.config.db_config import get_engine
from src.api_clients.module2_api.heatmap_client import trigger_heatmap_update
from src.module2_prediction.prediction_features import (
    FLOOD_FEATURES,
    LANDSLIDE_FEATURES,
    TIME_STEPS,
    build_flood_sequence,
    positive_class_probabilities,
    prepare_landslide_features,
    validate_estimator_features,
)
from src.utils.model_validation import resolve_model_path


DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_latest_active_model(engine, target: str):
    query = text(
        """
        SELECT model_path, scaler_path
        FROM batxat_model_registry
        WHERE model_target = :target AND is_active = TRUE
        ORDER BY created_at DESC
        LIMIT 1
        """
    )
    with engine.connect() as connection:
        result = connection.execute(query, {"target": target.upper()}).fetchone()
    if not result:
        raise FileNotFoundError(f"No active {target.upper()} model is registered")

    model_path = resolve_model_path(result.model_path)
    scaler_path = None
    if result.scaler_path and result.scaler_path.strip().upper() != "N/A":
        scaler_path = resolve_model_path(result.scaler_path)
    return model_path, scaler_path


def load_prediction_artifacts(engine):
    flood_path, scaler_path = load_latest_active_model(engine, "FLOOD")
    if scaler_path is None:
        raise ValueError("The active FLOOD model has no scaler")
    landslide_path, _ = load_latest_active_model(engine, "LANDSLIDE")

    flood_model = load_model(flood_path, compile=False)
    flood_scaler = joblib.load(scaler_path)
    landslide_model = joblib.load(landslide_path)

    validate_estimator_features(flood_scaler, FLOOD_FEATURES, "Flood scaler")
    validate_estimator_features(landslide_model, LANDSLIDE_FEATURES, "Landslide model")
    expected_input = (None, TIME_STEPS, len(FLOOD_FEATURES))
    if tuple(flood_model.input_shape) != expected_input:
        raise ValueError(
            f"Flood model input contract mismatch: expected {expected_input}, "
            f"got {flood_model.input_shape}"
        )
    return flood_model, flood_scaler, landslide_model


def predict_spatial_landslide_risk(engine, model, weather):
    query = text(
        """
        SELECT id, elevation_z, dist_to_water, slope, landcover
        FROM batxat_buildings
        ORDER BY id
        """
    )
    with engine.connect() as connection:
        buildings = pd.read_sql(query, connection)
    if buildings.empty:
        logger.warning("No buildings are available for landslide prediction")
        return buildings.assign(landslide_prob=pd.Series(dtype=float)), 0.0

    features = prepare_landslide_features(buildings, weather)
    buildings["landslide_prob"] = positive_class_probabilities(model, features)
    maximum_risk = round(float(buildings["landslide_prob"].max() * 100), 2)
    return buildings, maximum_risk


def persist_prediction_results(engine, buildings, flood_score, landslide_score, status):
    building_updates = [
        {"p_id": int(row.id), "p_prob": float(row.landslide_prob)}
        for row in buildings.itertuples(index=False)
    ]
    with engine.begin() as connection:
        if building_updates:
            connection.execute(
                text(
                    "UPDATE batxat_buildings "
                    "SET landslide_prob = :p_prob WHERE id = :p_id"
                ),
                building_updates,
            )
        connection.execute(
            text(
                """
                INSERT INTO batxat_daily_risk
                    (forecast_date, flood_risk_pct, landslide_risk_pct, alert_level)
                VALUES (CURRENT_DATE, :f, :l, :s)
                ON CONFLICT (forecast_date) DO UPDATE SET
                    flood_risk_pct = EXCLUDED.flood_risk_pct,
                    landslide_risk_pct = EXCLUDED.landslide_risk_pct,
                    alert_level = EXCLUDED.alert_level
                """
            ),
            {"f": flood_score, "l": landslide_score, "s": status},
        )
        connection.execute(
            text("UPDATE batxat_system_state SET active_flood_level = 80.0 + (:f / 10.0)"),
            {"f": flood_score},
        )


def run_integrated_predictions(*, persist: bool = True, notify: bool = True):
    engine = get_engine()
    logger.info("Starting integrated flood and landslide prediction")
    flood_model, flood_scaler, landslide_model = load_prediction_artifacts(engine)

    history_path = DATA_DIR / "flood_full_history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"Flood history does not exist: {history_path}")
    history = pd.read_csv(history_path)
    sequence = build_flood_sequence(history, flood_scaler)
    flood_output = np.asarray(flood_model.predict(sequence, verbose=0), dtype=float)
    if flood_output.size != 1 or not np.isfinite(flood_output).all():
        raise ValueError(f"Flood model returned an invalid output shape: {flood_output.shape}")
    flood_score = round(float(flood_output.reshape(-1)[0] * 100), 2)

    latest_weather = history.sort_values("datetime").iloc[-1]
    buildings, landslide_score = predict_spatial_landslide_risk(
        engine,
        landslide_model,
        latest_weather,
    )

    combined_risk = max(flood_score, landslide_score)
    status = "DANGER" if combined_risk > 75 else "WARNING" if combined_risk > 45 else "SAFE"
    result = {
        "flood_score": flood_score,
        "landslide_score": landslide_score,
        "combined_risk": combined_risk,
        "status": status,
        "building_count": int(len(buildings)),
        "persisted": persist,
    }
    logger.info("Prediction result: %s", result)

    if persist:
        persist_prediction_results(engine, buildings, flood_score, landslide_score, status)
        if notify:
            trigger_heatmap_update()
    return result


def main():
    parser = argparse.ArgumentParser(description="Run integrated GuardBatXat risk prediction")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load real artifacts and data, but do not update the database or notify Spring Boot",
    )
    args = parser.parse_args()
    run_integrated_predictions(persist=not args.dry_run, notify=not args.dry_run)


if __name__ == "__main__":
    main()
