"""
                        performance.py

                        Part of Credit Scoring Platform.
                        """
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.monitoring.metrics import CreditMetrics


class PerformanceMonitor:
    """Monitor credit model performance over time."""

    def __init__(self):
        self.history: list[dict] = []

    def evaluate_period(
        self,
        y_true,
        probabilities,
        period: str | None = None,
    ) -> dict:

        metrics = CreditMetrics.evaluate(
            y_true,
            probabilities,
        )

        result = {
            "period": (
                period
                or datetime.now(
                    timezone.utc
                ).strftime("%Y-%m")
            ),
            **metrics,
        }

        self.history.append(
            result
        )

        return result

    def get_history(self) -> pd.DataFrame:

        return pd.DataFrame(
            self.history
        )

    def latest(self) -> dict:

        if not self.history:
            return {}

        return self.history[-1]