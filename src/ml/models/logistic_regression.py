from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def create_logistic_regression() -> LogisticRegression:
    """Create the traditional credit-scoring baseline."""

    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )