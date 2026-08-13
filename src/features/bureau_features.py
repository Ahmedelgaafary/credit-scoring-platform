from __future__ import annotations

import pandas as pd


class BureauFeatureEngineer:
    """Create applicant-level features from bureau history."""

    @staticmethod
    def create_features(
        bureau: pd.DataFrame,
        bureau_balance_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate bureau history by SK_ID_CURR.
        """

        data = bureau.copy()

        # ----------------------------------------------------------
        # Merge bureau balance features
        # ----------------------------------------------------------

        if bureau_balance_features is not None:
            data = data.merge(
                bureau_balance_features,
                on="SK_ID_BUREAU",
                how="left",
            )

        # ----------------------------------------------------------
        # Basic credit history
        # ----------------------------------------------------------

        basic_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                BUREAU_CREDIT_COUNT=(
                    "SK_ID_BUREAU",
                    "count",
                ),
                BUREAU_CREDIT_TYPE_COUNT=(
                    "CREDIT_TYPE",
                    "nunique",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Credit status
        # ----------------------------------------------------------

        data["ACTIVE_CREDIT"] = (
            data["CREDIT_ACTIVE"] == "Active"
        ).astype(int)

        data["CLOSED_CREDIT"] = (
            data["CREDIT_ACTIVE"] == "Closed"
        ).astype(int)

        data["BAD_DEBT_CREDIT"] = (
            data["CREDIT_ACTIVE"] == "Bad debt"
        ).astype(int)

        data["SOLD_CREDIT"] = (
            data["CREDIT_ACTIVE"] == "Sold"
        ).astype(int)

        status_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                BUREAU_ACTIVE_COUNT=(
                    "ACTIVE_CREDIT",
                    "sum",
                ),
                BUREAU_CLOSED_COUNT=(
                    "CLOSED_CREDIT",
                    "sum",
                ),
                BUREAU_BAD_DEBT_COUNT=(
                    "BAD_DEBT_CREDIT",
                    "sum",
                ),
                BUREAU_SOLD_COUNT=(
                    "SOLD_CREDIT",
                    "sum",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Credit amounts
        # ----------------------------------------------------------

        amount_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                BUREAU_CREDIT_SUM_TOTAL=(
                    "AMT_CREDIT_SUM",
                    "sum",
                ),
                BUREAU_CREDIT_SUM_MEAN=(
                    "AMT_CREDIT_SUM",
                    "mean",
                ),
                BUREAU_CREDIT_SUM_MAX=(
                    "AMT_CREDIT_SUM",
                    "max",
                ),
                BUREAU_DEBT_TOTAL=(
                    "AMT_CREDIT_SUM_DEBT",
                    "sum",
                ),
                BUREAU_DEBT_MEAN=(
                    "AMT_CREDIT_SUM_DEBT",
                    "mean",
                ),
                BUREAU_CREDIT_LIMIT_TOTAL=(
                    "AMT_CREDIT_SUM_LIMIT",
                    "sum",
                ),
                BUREAU_OVERDUE_AMOUNT_TOTAL=(
                    "AMT_CREDIT_SUM_OVERDUE",
                    "sum",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Overdue accounts
        # ----------------------------------------------------------

        data["HAS_OVERDUE"] = (
            data["AMT_CREDIT_SUM_OVERDUE"] > 0
        ).astype(int)

        overdue_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                BUREAU_OVERDUE_ACCOUNT_COUNT=(
                    "HAS_OVERDUE",
                    "sum",
                ),
                BUREAU_DAYS_OVERDUE_MEAN=(
                    "CREDIT_DAY_OVERDUE",
                    "mean",
                ),
                BUREAU_DAYS_OVERDUE_MAX=(
                    "CREDIT_DAY_OVERDUE",
                    "max",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Bureau-balance features
        # ----------------------------------------------------------

        if bureau_balance_features is not None:

            balance_columns = [
                column
                for column in bureau_balance_features.columns
                if column != "SK_ID_BUREAU"
            ]

            balance_features = (
                data.groupby("SK_ID_CURR")[
                    balance_columns
                ]
                .mean()
                .reset_index()
            )

            balance_features = balance_features.rename(
                columns={
                    column: f"BUREAU_HISTORY_{column}"
                    for column in balance_columns
                }
            )

        else:
            balance_features = None

        # ----------------------------------------------------------
        # Merge
        # ----------------------------------------------------------

        result = basic_features

        for feature_table in [
            status_features,
            amount_features,
            overdue_features,
            balance_features,
        ]:

            if feature_table is not None:
                result = result.merge(
                    feature_table,
                    on="SK_ID_CURR",
                    how="left",
                )

        return result