from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def create_random_forest() -> RandomForestClassifier:
    """Create random-forest credit-risk model."""

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )