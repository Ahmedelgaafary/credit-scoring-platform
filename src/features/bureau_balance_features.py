from __future__ import annotations

import pandas as pd


class BureauBalanceFeatureEngineer:
    """Create bureau-level features from monthly bureau balances."""

    @staticmethod
    def create_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate monthly bureau balance records
        by SK_ID_BUREAU.
        """

        data = df.copy()

        # ----------------------------------------------------------
        # Status indicators
        # ----------------------------------------------------------

        data["STATUS_0"] = (
            data["STATUS"] == "0"
        ).astype(int)

        data["STATUS_1"] = (
            data["STATUS"] == "1"
        ).astype(int)

        data["STATUS_2"] = (
            data["STATUS"] == "2"
        ).astype(int)

        data["STATUS_3"] = (
            data["STATUS"] == "3"
        ).astype(int)

        data["STATUS_4"] = (
            data["STATUS"] == "4"
        ).astype(int)

        data["STATUS_5"] = (
            data["STATUS"] == "5"
        ).astype(int)

        data["STATUS_C"] = (
            data["STATUS"] == "C"
        ).astype(int)

        data["STATUS_X"] = (
            data["STATUS"] == "X"
        ).astype(int)

        # ----------------------------------------------------------
        # Delinquency indicators
        # ----------------------------------------------------------

        data["STATUS_OVERDUE"] = (
            data["STATUS"].isin(
                ["1", "2", "3", "4", "5"]
            )
        ).astype(int)

        data["STATUS_SEVERE_OVERDUE"] = (
            data["STATUS"].isin(
                ["3", "4", "5"]
            )
        ).astype(int)

        # ----------------------------------------------------------
        # Aggregate monthly history
        # ----------------------------------------------------------

        features = (
            data.groupby("SK_ID_BUREAU")
            .agg(
                BUREAU_BALANCE_MONTH_COUNT=(
                    "MONTHS_BALANCE",
                    "count",
                ),
                BUREAU_BALANCE_MONTH_MIN=(
                    "MONTHS_BALANCE",
                    "min",
                ),
                BUREAU_BALANCE_MONTH_MAX=(
                    "MONTHS_BALANCE",
                    "max",
                ),
                BUREAU_BALANCE_STATUS_COUNT=(
                    "STATUS",
                    "count",
                ),
                BUREAU_BALANCE_STATUS_NUNIQUE=(
                    "STATUS",
                    "nunique",
                ),
                BUREAU_BALANCE_OVERDUE_COUNT=(
                    "STATUS_OVERDUE",
                    "sum",
                ),
                BUREAU_BALANCE_SEVERE_OVERDUE_COUNT=(
                    "STATUS_SEVERE_OVERDUE",
                    "sum",
                ),
                BUREAU_BALANCE_STATUS_0_COUNT=(
                    "STATUS_0",
                    "sum",
                ),
                BUREAU_BALANCE_STATUS_1_COUNT=(
                    "STATUS_1",
                    "sum",
                ),
                BUREAU_BALANCE_STATUS_2_COUNT=(
                    "STATUS_2",
                    "sum",
                ),
                BUREAU_BALANCE_STATUS_3_COUNT=(
                    "STATUS_3",
                    "sum",
                ),
                BUREAU_BALANCE_STATUS_4_COUNT=(
                    "STATUS_4",
                    "sum",
                ),
                BUREAU_BALANCE_STATUS_5_COUNT=(
                    "STATUS_5",
                    "sum",
                ),
                BUREAU_BALANCE_STATUS_C_COUNT=(
                    "STATUS_C",
                    "sum",
                ),
                BUREAU_BALANCE_STATUS_X_COUNT=(
                    "STATUS_X",
                    "sum",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Behavioral rates
        # ----------------------------------------------------------

        features["BUREAU_BALANCE_OVERDUE_RATE"] = (
            features["BUREAU_BALANCE_OVERDUE_COUNT"]
            / features["BUREAU_BALANCE_MONTH_COUNT"]
        )

        features["BUREAU_BALANCE_SEVERE_OVERDUE_RATE"] = (
            features[
                "BUREAU_BALANCE_SEVERE_OVERDUE_COUNT"
            ]
            / features["BUREAU_BALANCE_MONTH_COUNT"]
        )

        return features