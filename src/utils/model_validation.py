from pathlib import Path
from typing import Optional
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = (PROJECT_ROOT / "models").resolve()


def resolve_model_path(raw_path: str) -> Path:
    if not raw_path or raw_path.strip().upper() == "N/A":
        raise ValueError("Model path is required")

    normalized = raw_path.replace("\\", "/")
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()

    if candidate != MODELS_DIR and MODELS_DIR not in candidate.parents:
        raise ValueError("Model path must stay inside the models directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"Model file does not exist: {candidate.name}")
    return candidate


def validate_model_files(
    model_target: str,
    model_path: str,
    scaler_path: Optional[str] = None,
) -> dict:
    target = (model_target or "").strip().upper()
    model_file = resolve_model_path(model_path)

    if target == "FLOOD":
        if not scaler_path or scaler_path.strip().upper() == "N/A":
            raise ValueError("FLOOD model requires a scaler")
        scaler_file = resolve_model_path(scaler_path)

        from keras.models import load_model
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning
        from src.module2_prediction.prediction_features import (
            FLOOD_FEATURES,
            TIME_STEPS,
            validate_estimator_features,
        )

        model = load_model(model_file, compile=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            scaler = joblib.load(scaler_file)
        validate_estimator_features(scaler, FLOOD_FEATURES, "Flood scaler")
        expected_input = (None, TIME_STEPS, len(FLOOD_FEATURES))
        if tuple(model.input_shape) != expected_input:
            raise ValueError(
                f"Flood model input contract mismatch: expected {expected_input}, "
                f"got {model.input_shape}"
            )
    elif target == "LANDSLIDE":
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning
        from src.module2_prediction.prediction_features import (
            LANDSLIDE_FEATURES,
            validate_estimator_features,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(model_file)
        validate_estimator_features(model, LANDSLIDE_FEATURES, "Landslide model")
        scaler_file = None
        if scaler_path and scaler_path.strip().upper() != "N/A":
            scaler_file = resolve_model_path(scaler_path)
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                joblib.load(scaler_file)
    else:
        raise ValueError(f"Unsupported model target: {model_target}")

    return {
        "valid": True,
        "modelTarget": target,
        "modelFile": model_file.name,
        "scalerFile": scaler_file.name if scaler_file else None,
    }
