"""
                        metrics.py

                        Part of Credit Scoring Platform.
                        """
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


class CreditMetrics:
    """Performance metrics for credit-risk models."""

    @staticmethod
    def roc_auc(
        y_true: pd.Series | np.ndarray,
        probabilities: pd.Series | np.ndarray,
    ) -> float:
        return float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    @staticmethod
    def pr_auc(
        y_true: pd.Series | np.ndarray,
        probabilities: pd.Series | np.ndarray,
    ) -> float:
        return float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    @staticmethod
    def gini(
        y_true: pd.Series | np.ndarray,
        probabilities: pd.Series | np.ndarray,
    ) -> float:
        auc = CreditMetrics.roc_auc(
            y_true,
            probabilities,
        )

        return float(
            2 * auc - 1
        )

    @staticmethod
    def ks(
        y_true: pd.Series | np.ndarray,
        probabilities: pd.Series | np.ndarray,
    ) -> float:
        data = pd.DataFrame(
            {
                "target": np.asarray(
                    y_true
                ),
                "probability": np.asarray(
                    probabilities
                ),
            }
        )

        data = data.sort_values(
            "probability",
            ascending=False,
        )

        total_bad = (
            data["target"] == 1
        ).sum()

        total_good = (
            data["target"] == 0
        ).sum()

        if total_bad == 0 or total_good == 0:
            return 0.0

        cumulative_bad = (
            (data["target"] == 1)
            .cumsum()
            / total_bad
        )

        cumulative_good = (
            (data["target"] == 0)
            .cumsum()
            / total_good
        )

        return float(
            np.max(
                np.abs(
                    cumulative_bad
                    - cumulative_good
                )
            )
        )

    @staticmethod
    def brier_score(
        y_true: pd.Series | np.ndarray,
        probabilities: pd.Series | np.ndarray,
    ) -> float:
        return float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        )

    @classmethod
    def evaluate(
        cls,
        y_true,
        probabilities,
    ) -> dict[str, float]:

        return {
            "roc_auc": cls.roc_auc(
                y_true,
                probabilities,
            ),
            "pr_auc": cls.pr_auc(
                y_true,
                probabilities,
            ),
            "gini": cls.gini(
                y_true,
                probabilities,
            ),
            "ks": cls.ks(
                y_true,
                probabilities,
            ),
            "brier_score": cls.brier_score(
                y_true,
                probabilities,
            ),
        }