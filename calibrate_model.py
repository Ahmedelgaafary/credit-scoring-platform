from pathlib import Path

import pandas as pd

from src.risk.calibration import (
    ProbabilityCalibrator,
)


def main():

    print("=" * 70)
    print("PROBABILITY CALIBRATION")
    print("=" * 70)

    predictions_path = Path(
        "models/validation_predictions.csv"
    )

    predictions = pd.read_csv(
        predictions_path
    )

    y_true = predictions[
        "y_true"
    ]

    raw_probability = predictions[
        "probability_of_default"
    ]

    # --------------------------------------------------------------
    # Raw Brier score
    # --------------------------------------------------------------

    raw_brier = (
        ProbabilityCalibrator.brier_score(
            y_true,
            raw_probability,
        )
    )

    print(
        f"\nRaw Brier score: {raw_brier:.6f}"
    )

    # --------------------------------------------------------------
    # Fit calibrator
    # --------------------------------------------------------------

    calibrator = ProbabilityCalibrator(
        method="isotonic"
    )

    calibrator.fit(
        raw_probability,
        y_true,
    )

    # --------------------------------------------------------------
    # Calibrated probabilities
    # --------------------------------------------------------------

    calibrated_probability = (
        calibrator.predict(
            raw_probability
        )
    )

    calibrated_brier = (
        ProbabilityCalibrator.brier_score(
            y_true,
            calibrated_probability,
        )
    )

    print(
        f"Calibrated Brier score: "
        f"{calibrated_brier:.6f}"
    )

    # --------------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------------

    predictions[
        "calibrated_probability_of_default"
    ] = calibrated_probability

    output_path = Path(
        "models/"
        "calibrated_validation_predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Save calibrator
    # --------------------------------------------------------------

    calibrator_path = Path(
        "models/"
        "probability_calibrator.joblib"
    )

    calibrator.save(
        calibrator_path
    )

    print(
        f"\nPredictions saved to: "
        f"{output_path}"
    )

    print(
        f"Calibrator saved to: "
        f"{calibrator_path}"
    )


if __name__ == "__main__":
    main()