import unittest

import numpy as np
import pandas as pd

from src.module2_prediction.prediction_features import (
    FLOOD_FEATURES,
    LANDSLIDE_FEATURES,
    build_flood_sequence,
    prepare_flood_feature_frame,
    prepare_landslide_features,
)


class IdentityScaler:
    n_features_in_ = len(FLOOD_FEATURES)
    feature_names_in_ = np.asarray(FLOOD_FEATURES, dtype=object)

    def transform(self, values):
        return np.asarray(values, dtype=float)


def flood_history(rows=6):
    data = {
        "datetime": pd.date_range("2026-08-01", periods=rows, freq="D"),
        "precip_BAT_XAT": np.arange(rows, dtype=float),
        "precip_3d_BAT_XAT": np.arange(rows, dtype=float) + 1,
        "soil_BAT_XAT": np.full(rows, 0.5),
        "soil_fatigue_BAT_XAT": np.full(rows, 0.25),
        "interaction_risk_BAT_XAT": np.arange(rows, dtype=float) * 0.5,
        "basin_precip": np.arange(rows, dtype=float) + 2,
        "water_level": np.arange(rows, dtype=float) + 80,
    }
    for station in ("TRINH_TUONG", "Y_TY", "SANG_MA_SAO"):
        data[f"precip_{station}"] = np.arange(rows, dtype=float) + 10
        data[f"soil_{station}"] = np.full(rows, 0.4)
        data[f"soil_fatigue_{station}"] = np.full(rows, 0.2)
        data[f"interaction_risk_{station}"] = np.arange(rows, dtype=float) + 20
    return pd.DataFrame(data)


class PredictionFeatureTests(unittest.TestCase):
    def test_flood_contract_builds_exact_model_shape(self):
        sequence = build_flood_sequence(flood_history(), IdentityScaler())
        self.assertEqual(sequence.shape, (1, 5, 21))
        self.assertTrue(np.isfinite(sequence).all())

    def test_flood_lag_uses_previous_day(self):
        prepared = prepare_flood_feature_frame(flood_history())
        self.assertEqual(prepared.loc[1, "precip_Y_TY_lag1"], 10.0)
        self.assertEqual(prepared.loc[0, "precip_Y_TY_lag1"], 0.0)

    def test_landslide_contract_uses_real_db_fields(self):
        buildings = pd.DataFrame(
            {
                "slope": [31.5],
                "elevation_z": [900.0],
                "dist_to_water": [42.0],
                "landcover": [30.0],
            }
        )
        features = prepare_landslide_features(
            buildings,
            {"precip_3d_BAT_XAT": 100.0, "soil_BAT_XAT": 0.7},
        )
        self.assertEqual(features.columns.tolist(), LANDSLIDE_FEATURES)
        self.assertEqual(features.loc[0, "Slope"], 31.5)
        self.assertEqual(features.loc[0, "Landcover"], 30.0)

    def test_missing_landcover_fails_instead_of_understating_risk(self):
        buildings = pd.DataFrame(
            {"slope": [10], "elevation_z": [500], "dist_to_water": [100]}
        )
        with self.assertRaisesRegex(ValueError, "landcover"):
            prepare_landslide_features(
                buildings,
                {"precip_3d_BAT_XAT": 1.0, "soil_BAT_XAT": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
