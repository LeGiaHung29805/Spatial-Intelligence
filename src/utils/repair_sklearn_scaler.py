import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import MinMaxScaler


def repair_scaler(source: Path, destination: Path) -> float:
    if source.resolve() == destination.resolve():
        raise ValueError("Destination must differ from source; originals are never overwritten")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        scaler = joblib.load(source)

    if not isinstance(scaler, MinMaxScaler):
        raise TypeError(f"Only MinMaxScaler is supported, got {type(scaler).__name__}")

    samples = np.vstack([
        scaler.data_min_,
        (scaler.data_min_ + scaler.data_max_) / 2.0,
        scaler.data_max_,
    ])
    expected = scaler.transform(samples)

    joblib.dump(scaler, destination)
    reloaded = joblib.load(destination)
    actual = reloaded.transform(samples)
    max_diff = float(np.max(np.abs(expected - actual)))
    if max_diff > 1e-12:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Scaler conversion changed output; max diff={max_diff}")
    return max_diff


def main():
    parser = argparse.ArgumentParser(
        description="Re-serialize a MinMaxScaler and verify identical transforms"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    max_diff = repair_scaler(args.source, args.destination)
    print(f"Created {args.destination}; max transform difference={max_diff}")


if __name__ == "__main__":
    main()
