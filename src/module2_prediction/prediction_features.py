"""Shared feature contracts for training, validation, and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


TIME_STEPS = 5

FLOOD_FEATURES = [
    "precip_BAT_XAT",
    "precip_3d_BAT_XAT",
    "soil_BAT_XAT",
    "soil_fatigue_BAT_XAT",
    "interaction_risk_BAT_XAT",
    "precip_TRINH_TUONG_lag1",
    "soil_TRINH_TUONG_lag1",
    "soil_fatigue_TRINH_TUONG_lag1",
    "interaction_risk_TRINH_TUONG_lag1",
    "precip_Y_TY_lag1",
    "soil_Y_TY_lag1",
    "soil_fatigue_Y_TY_lag1",
    "interaction_risk_Y_TY_lag1",
    "precip_SANG_MA_SAO_lag1",
    "soil_SANG_MA_SAO_lag1",
    "soil_fatigue_SANG_MA_SAO_lag1",
    "interaction_risk_SANG_MA_SAO_lag1",
    "basin_precip",
    "water_level",
    "sin_season",
    "cos_season",
]

LANDSLIDE_FEATURES = [
    "Slope",
    "Elevation",
    "Dist_to_Water",
    "precip_3d_BAT_XAT",
    "soil_BAT_XAT",
    "Landcover",
]

LAG_STATIONS = ("TRINH_TUONG", "Y_TY", "SANG_MA_SAO")
LAG_VARIABLES = ("precip", "soil", "soil_fatigue", "interaction_risk")


def require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def validate_estimator_features(estimator, expected: Sequence[str], label: str) -> None:
    expected_list = list(expected)
    actual_count = getattr(estimator, "n_features_in_", None)
    if actual_count is not None and int(actual_count) != len(expected_list):
        raise ValueError(
            f"{label} expects {actual_count} features, but the pipeline provides "
            f"{len(expected_list)}"
        )

    actual_names = getattr(estimator, "feature_names_in_", None)
    if actual_names is not None:
        actual_list = [str(name) for name in actual_names]
        if actual_list != expected_list:
            raise ValueError(
                f"{label} feature contract mismatch. Expected {expected_list}, "
                f"artifact contains {actual_list}"
            )


def prepare_flood_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    require_columns(history, ["datetime"], "Flood history")
    frame = history.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    if frame["datetime"].isna().any():
        raise ValueError("Flood history contains invalid datetime values")
    frame = frame.sort_values("datetime").reset_index(drop=True)

    lag_sources = [
        f"{variable}_{station}"
        for station in LAG_STATIONS
        for variable in LAG_VARIABLES
    ]
    non_derived = [
        feature
        for feature in FLOOD_FEATURES
        if not feature.endswith("_lag1") and feature not in {"sin_season", "cos_season"}
    ]
    require_columns(frame, lag_sources + non_derived, "Flood history")

    day_of_year = frame["datetime"].dt.dayofyear
    frame["sin_season"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["cos_season"] = np.cos(2 * np.pi * day_of_year / 365.25)

    for source in lag_sources:
        frame[f"{source}_lag1"] = pd.to_numeric(frame[source], errors="coerce").shift(1).fillna(0.0)

    frame[FLOOD_FEATURES] = frame[FLOOD_FEATURES].apply(pd.to_numeric, errors="coerce")
    return frame


def build_flood_sequence(history: pd.DataFrame, scaler, time_steps: int = TIME_STEPS) -> np.ndarray:
    validate_estimator_features(scaler, FLOOD_FEATURES, "Flood scaler")
    frame = prepare_flood_feature_frame(history)
    if len(frame) < time_steps:
        raise ValueError(f"Flood history needs at least {time_steps} rows, got {len(frame)}")

    latest = frame.tail(time_steps)[FLOOD_FEATURES]
    if latest.isna().any().any() or not np.isfinite(latest.to_numpy(dtype=float)).all():
        invalid = latest.columns[latest.isna().any()].tolist()
        raise ValueError(f"Latest flood window contains invalid values: {invalid}")

    scaled = scaler.transform(latest)
    sequence = np.asarray(scaled, dtype=np.float32).reshape(1, time_steps, len(FLOOD_FEATURES))
    if not np.isfinite(sequence).all():
        raise ValueError("Flood scaler produced non-finite values")
    return sequence


def prepare_landslide_features(
    buildings: pd.DataFrame,
    weather: Mapping[str, object] | pd.Series,
) -> pd.DataFrame:
    require_columns(
        buildings,
        ["slope", "elevation_z", "dist_to_water", "landcover"],
        "Building data",
    )
    for field in ("precip_3d_BAT_XAT", "soil_BAT_XAT"):
        if field not in weather:
            raise ValueError(f"Weather data is missing required field: {field}")

    features = pd.DataFrame(
        {
            "Slope": buildings["slope"],
            "Elevation": buildings["elevation_z"],
            "Dist_to_Water": buildings["dist_to_water"],
            "precip_3d_BAT_XAT": weather["precip_3d_BAT_XAT"],
            "soil_BAT_XAT": weather["soil_BAT_XAT"],
            "Landcover": buildings["landcover"],
        },
        index=buildings.index,
    )
    features = features[LANDSLIDE_FEATURES].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        invalid = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Building/weather features contain invalid values: {invalid}")
    return features


def positive_class_probabilities(classifier, features: pd.DataFrame) -> np.ndarray:
    validate_estimator_features(classifier, LANDSLIDE_FEATURES, "Landslide model")
    classes = list(getattr(classifier, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"Landslide model has no positive class 1: {classes}")
    probabilities = np.asarray(classifier.predict_proba(features))[:, classes.index(1)]
    if not np.isfinite(probabilities).all():
        raise ValueError("Landslide model produced non-finite probabilities")
    return probabilities
