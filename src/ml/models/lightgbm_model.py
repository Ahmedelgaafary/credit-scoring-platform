from __future__ import annotations

from lightgbm import LGBMClassifier


def create_lightgbm() -> LGBMClassifier:
    """Create LightGBM credit-risk model."""

    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )