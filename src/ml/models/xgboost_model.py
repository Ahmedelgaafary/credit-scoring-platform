from __future__ import annotations

from xgboost import XGBClassifier


def create_xgboost() -> XGBClassifier:
    """Create XGBoost credit-risk model."""

    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )