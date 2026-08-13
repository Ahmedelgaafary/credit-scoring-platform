from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.monitoring import (
    CreditMetrics,
    PerformanceMonitor,
    DriftMonitor,
)


PREDICTIONS_FILE = (
    Path("models")
    / "calibrated_validation_predictions.csv"
)


def main():

    print("=" * 70)
    print("CREDIT MODEL MONITORING TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load validation predictions
    # --------------------------------------------------------------

    print("\nLoading validation predictions...")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    print(
        f"Loaded shape: {df.shape}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # --------------------------------------------------------------
    # Identify target and probability columns
    # --------------------------------------------------------------

    target_candidates = [
        "TARGET",
        "target",
        "y_true",
    ]

    probability_candidates = [
        "calibrated_probability_of_default",
        "calibrated_probability",
        "calibrated_pd",
        "probability_of_default",
        "probability",
        "prediction",
    ]

    target_column = next(
        (
            column
            for column in target_candidates
            if column in df.columns
        ),
        None,
    )

    probability_column = next(
        (
            column
            for column in probability_candidates
            if column in df.columns
        ),
        None,
    )

    if target_column is None:
        raise ValueError(
            "Could not find target column."
        )

    if probability_column is None:
        raise ValueError(
            "Could not find probability column."
        )

    print(
        f"\nTarget column: "
        f"{target_column}"
    )

    print(
        f"Probability column: "
        f"{probability_column}"
    )

    y_true = df[target_column]

    probabilities = df[
        probability_column
    ]

    # --------------------------------------------------------------
    # Credit metrics
    # --------------------------------------------------------------

    print("\n")
    print("# CREDIT MODEL METRICS")
    print("=" * 70)

    metrics = CreditMetrics.evaluate(
        y_true,
        probabilities,
    )

    for name, value in metrics.items():

        print(
            f"{name.upper():<15}: "
            f"{value:.6f}"
        )

    # --------------------------------------------------------------
    # Performance monitor
    # --------------------------------------------------------------

    print("\n")
    print("# PERFORMANCE MONITOR")
    print("=" * 70)

    monitor = PerformanceMonitor()

    result = monitor.evaluate_period(
        y_true,
        probabilities,
        period="validation",
    )

    for name, value in result.items():

        print(
            f"{name:<15}: {value}"
        )

    print("\nPerformance history:")

    print(
        monitor.get_history()
        .to_string(index=False)
    )

    # --------------------------------------------------------------
    # Drift test
    # --------------------------------------------------------------

    print("\n")
    print("# DRIFT MONITOR")
    print("=" * 70)

    # Create two samples from the
    # validation probability distribution
    # to verify the drift engine.

    midpoint = len(probabilities) // 2

    reference = probabilities.iloc[
        :midpoint
    ]

    current = probabilities.iloc[
        midpoint:
    ]

    drift_result = DriftMonitor.check_feature(
        reference,
        current,
    )

    print(
        f"PSI: "
        f"{drift_result['psi']:.6f}"
    )

    print(
        f"Status: "
        f"{drift_result['status']}"
    )

    # --------------------------------------------------------------
    # Final status
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MONITORING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()