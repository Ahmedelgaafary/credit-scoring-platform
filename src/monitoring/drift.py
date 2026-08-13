"""
                        drift.py

                        Part of Credit Scoring Platform.
                        """
from __future__ import annotations

import numpy as np
import pandas as pd


class DriftMonitor:
    """Monitor feature and prediction drift."""

    @staticmethod
    def population_stability_index(
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10,
    ) -> float:

        reference = pd.Series(
            reference
        ).dropna()

        current = pd.Series(
            current
        ).dropna()

        if reference.empty or current.empty:
            return 0.0

        edges = np.unique(
            np.quantile(
                reference,
                np.linspace(
                    0,
                    1,
                    bins + 1,
                ),
            )
        )

        if len(edges) < 2:
            return 0.0

        reference_counts, _ = np.histogram(
            reference,
            bins=edges,
        )

        current_counts, _ = np.histogram(
            current,
            bins=edges,
        )

        reference_pct = (
            reference_counts
            / max(
                reference_counts.sum(),
                1,
            )
        )

        current_pct = (
            current_counts
            / max(
                current_counts.sum(),
                1,
            )
        )

        epsilon = 1e-6

        reference_pct = np.clip(
            reference_pct,
            epsilon,
            None,
        )

        current_pct = np.clip(
            current_pct,
            epsilon,
            None,
        )

        psi = np.sum(
            (
                current_pct
                - reference_pct
            )
            * np.log(
                current_pct
                / reference_pct
            )
        )

        return float(psi)

    @staticmethod
    def classify_psi(
        psi: float,
    ) -> str:

        if psi < 0.10:
            return "stable"

        if psi < 0.25:
            return "moderate_drift"

        return "significant_drift"

    @classmethod
    def check_feature(
        cls,
        reference: pd.Series,
        current: pd.Series,
    ) -> dict:

        psi = cls.population_stability_index(
            reference,
            current,
        )

        return {
            "psi": psi,
            "status": cls.classify_psi(
                psi
            ),
        }

    @classmethod
    def check_dataset(
        cls,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> pd.DataFrame:

        results = []

        common_columns = (
            reference.columns.intersection(
                current.columns
            )
        )

        for column in common_columns:

            if not (
                pd.api.types.is_numeric_dtype(
                    reference[column]
                )
                and
                pd.api.types.is_numeric_dtype(
                    current[column]
                )
            ):
                continue

            result = cls.check_feature(
                reference[column],
                current[column],
            )

            results.append(
                {
                    "feature": column,
                    **result,
                }
            )

        return pd.DataFrame(
            results
        ).sort_values(
            "psi",
            ascending=False,
        )