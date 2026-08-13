import numpy as np

from src.monitoring import (
    CreditMetrics,
    DriftMonitor,
)


def test_credit_metrics():

    y_true = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    probabilities = np.array(
        [0.05, 0.10, 0.20, 0.60, 0.70, 0.90]
    )

    metrics = CreditMetrics.evaluate(
        y_true,
        probabilities,
    )

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert -1.0 <= metrics["gini"] <= 1.0
    assert 0.0 <= metrics["ks"] <= 1.0
    assert metrics["brier_score"] >= 0.0


def test_drift_monitor():

    reference = np.array(
        [1, 2, 3, 4, 5]
    )

    current = np.array(
        [1, 2, 3, 4, 5]
    )

    result = DriftMonitor.check_feature(
        reference,
        current,
    )

    assert result["psi"] >= 0.0
    assert result["status"] == "stable"