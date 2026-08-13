from __future__ import annotations

import numpy as np
import pandas as pd


class CreditCardFeatureEngineer:
    """Create applicant-level features from credit-card history."""

    @staticmethod
    def create_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate credit-card behavior by SK_ID_CURR.
        """

        data = df.copy()

        # ----------------------------------------------------------
        # Credit utilization
        # ----------------------------------------------------------

        data["CREDIT_UTILIZATION"] = (
            data["AMT_BALANCE"]
            / data["AMT_CREDIT_LIMIT_ACTUAL"].replace(
                0,
                np.nan,
            )
        )

        # ----------------------------------------------------------
        # Drawing utilization
        # ----------------------------------------------------------

        data["DRAWING_UTILIZATION"] = (
            data["AMT_DRAWINGS_CURRENT"]
            / data["AMT_CREDIT_LIMIT_ACTUAL"].replace(
                0,
                np.nan,
            )
        )

        # ----------------------------------------------------------
        # Payment ratio
        # ----------------------------------------------------------

        data["CREDIT_CARD_PAYMENT_RATIO"] = (
            data["AMT_PAYMENT_CURRENT"]
            / data["AMT_INST_MIN_REGULARITY"].replace(
                0,
                np.nan,
            )
        )

        
        # ----------------------------------------------------------
        # Aggregate credit-card behavior
        # ----------------------------------------------------------

        features = (
            data.groupby("SK_ID_CURR")
            .agg(
                CREDIT_CARD_COUNT=(
                    "SK_ID_PREV",
                    "nunique",
                ),
                CREDIT_CARD_RECORD_COUNT=(
                    "SK_ID_PREV",
                    "count",
                ),
                CREDIT_CARD_BALANCE_MEAN=(
                    "AMT_BALANCE",
                    "mean",
                ),
                CREDIT_CARD_BALANCE_MAX=(
                    "AMT_BALANCE",
                    "max",
                ),
                CREDIT_CARD_BALANCE_SUM=(
                    "AMT_BALANCE",
                    "sum",
                ),
                CREDIT_CARD_LIMIT_MEAN=(
                    "AMT_CREDIT_LIMIT_ACTUAL",
                    "mean",
                ),
                CREDIT_CARD_LIMIT_MAX=(
                    "AMT_CREDIT_LIMIT_ACTUAL",
                    "max",
                ),
                CREDIT_CARD_UTILIZATION_MEAN=(
                    "CREDIT_UTILIZATION",
                    "mean",
                ),
                CREDIT_CARD_UTILIZATION_MAX=(
                    "CREDIT_UTILIZATION",
                    "max",
                ),
                CREDIT_CARD_DRAWING_MEAN=(
                    "AMT_DRAWINGS_CURRENT",
                    "mean",
                ),
                CREDIT_CARD_DRAWING_MAX=(
                    "AMT_DRAWINGS_CURRENT",
                    "max",
                ),
                CREDIT_CARD_DRAWING_SUM=(
                    "AMT_DRAWINGS_CURRENT",
                    "sum",
                ),
                CREDIT_CARD_DRAWING_UTILIZATION_MEAN=(
                    "DRAWING_UTILIZATION",
                    "mean",
                ),
                CREDIT_CARD_PAYMENT_MEAN=(
                    "AMT_PAYMENT_CURRENT",
                    "mean",
                ),
                CREDIT_CARD_PAYMENT_SUM=(
                    "AMT_PAYMENT_CURRENT",
                    "sum",
                ),
                CREDIT_CARD_MIN_PAYMENT_MEAN=(
                    "AMT_INST_MIN_REGULARITY",
                    "mean",
                ),
                CREDIT_CARD_PAYMENT_RATIO_MEAN=(
                    "CREDIT_CARD_PAYMENT_RATIO",
                    "mean",
                ),
                CREDIT_CARD_PAYMENT_RATIO_MIN=(
                    "CREDIT_CARD_PAYMENT_RATIO",
                    "min",
                ),
                CREDIT_CARD_PAYMENT_RATIO_MAX=(
                    "CREDIT_CARD_PAYMENT_RATIO",
                    "max",
                ),
                CREDIT_CARD_RECEIVABLE_MEAN=(
                    "AMT_RECEIVABLE_PRINCIPAL",
                    "mean",
                ),
                CREDIT_CARD_RECEIVABLE_MAX=(
                    "AMT_RECEIVABLE_PRINCIPAL",
                    "max",
                ),
                CREDIT_CARD_INSTALMENT_COUNT_MEAN=(
                    "CNT_INSTALMENT_MATURE_CUM",
                    "mean",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # High-utilization indicators
        # ----------------------------------------------------------

        data["HIGH_UTILIZATION"] = (
            data["CREDIT_UTILIZATION"] > 0.80
        ).astype(int)

        utilization_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                CREDIT_CARD_HIGH_UTILIZATION_COUNT=(
                    "HIGH_UTILIZATION",
                    "sum",
                ),
            )
            .reset_index()
        )

        features = features.merge(
            utilization_features,
            on="SK_ID_CURR",
            how="left",
        )

        # ----------------------------------------------------------
        # Installment maturity
        # ----------------------------------------------------------

        if "CNT_INSTALMENT_MATURE_CUM" in data.columns:
            features["CREDIT_CARD_INSTALMENT_MATURE_MAX"] = (
                data.groupby("SK_ID_CURR")[
                    "CNT_INSTALMENT_MATURE_CUM"
                ].max().values
            )

        return features