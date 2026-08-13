from __future__ import annotations

from typing import Tuple

import pandas as pd

from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    target_column: str = "TARGET",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Split credit-scoring data into stratified
    training and validation sets.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found."
        )

    X = df.drop(
        columns=[
            target_column,
            "SK_ID_CURR",
        ],
        errors="ignore",
    )

    y = df[target_column]

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
    )