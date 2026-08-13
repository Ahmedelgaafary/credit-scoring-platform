from __future__ import annotations

import numpy as np
import pandas as pd


class InstallmentFeatureEngineer:
    """Create applicant-level features from installment payments."""

    @staticmethod
    def create_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate installment payment behavior by SK_ID_CURR.
        """

        data = df.copy()

        # ----------------------------------------------------------
        # Payment delay
        # ----------------------------------------------------------

        data["PAYMENT_DELAY_DAYS"] = (
            data["DAYS_ENTRY_PAYMENT"]
            - data["DAYS_INSTALMENT"]
        )

        data["LATE_PAYMENT"] = (
            data["PAYMENT_DELAY_DAYS"] > 0
        ).astype(int)

        data["SEVERE_LATE_PAYMENT"] = (
            data["PAYMENT_DELAY_DAYS"] > 30
        ).astype(int)

        # ----------------------------------------------------------
        # Payment amount difference
        # ----------------------------------------------------------

        data["PAYMENT_DIFFERENCE"] = (
            data["AMT_PAYMENT"]
            - data["AMT_INSTALMENT"]
        )

        data["UNDERPAYMENT"] = (
            data["AMT_PAYMENT"]
            < data["AMT_INSTALMENT"]
        ).astype(int)

        # ----------------------------------------------------------
        # Payment ratio
        # ----------------------------------------------------------

        data["PAYMENT_RATIO"] = (
            data["AMT_PAYMENT"]
            / data["AMT_INSTALMENT"].replace(0, np.nan)
        )

        # ----------------------------------------------------------
        # Aggregate payment behavior
        # ----------------------------------------------------------

        features = (
            data.groupby("SK_ID_CURR")
            .agg(
                INSTALLMENT_COUNT=(
                    "SK_ID_PREV",
                    "count",
                ),
                INSTALLMENT_DELAY_MEAN=(
                    "PAYMENT_DELAY_DAYS",
                    "mean",
                ),
                INSTALLMENT_DELAY_MAX=(
                    "PAYMENT_DELAY_DAYS",
                    "max",
                ),
                INSTALLMENT_DELAY_MIN=(
                    "PAYMENT_DELAY_DAYS",
                    "min",
                ),
                INSTALLMENT_DELAY_STD=(
                    "PAYMENT_DELAY_DAYS",
                    "std",
                ),
                INSTALLMENT_LATE_COUNT=(
                    "LATE_PAYMENT",
                    "sum",
                ),
                INSTALLMENT_SEVERE_LATE_COUNT=(
                    "SEVERE_LATE_PAYMENT",
                    "sum",
                ),
                INSTALLMENT_UNDERPAYMENT_COUNT=(
                    "UNDERPAYMENT",
                    "sum",
                ),
                INSTALLMENT_PAYMENT_MEAN=(
                    "AMT_PAYMENT",
                    "mean",
                ),
                INSTALLMENT_AMOUNT_MEAN=(
                    "AMT_INSTALMENT",
                    "mean",
                ),
                INSTALLMENT_PAYMENT_SUM=(
                    "AMT_PAYMENT",
                    "sum",
                ),
                INSTALLMENT_AMOUNT_SUM=(
                    "AMT_INSTALMENT",
                    "sum",
                ),
                INSTALLMENT_PAYMENT_RATIO_MEAN=(
                    "PAYMENT_RATIO",
                    "mean",
                ),
                INSTALLMENT_PAYMENT_RATIO_MIN=(
                    "PAYMENT_RATIO",
                    "min",
                ),
                INSTALLMENT_PAYMENT_RATIO_MAX=(
                    "PAYMENT_RATIO",
                    "max",
                ),
                INSTALLMENT_DIFFERENCE_MEAN=(
                    "PAYMENT_DIFFERENCE",
                    "mean",
                ),
                INSTALLMENT_DIFFERENCE_MIN=(
                    "PAYMENT_DIFFERENCE",
                    "min",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Late-payment rate
        # ----------------------------------------------------------

        features["INSTALLMENT_LATE_RATE"] = (
            features["INSTALLMENT_LATE_COUNT"]
            / features["INSTALLMENT_COUNT"]
        )

        features["INSTALLMENT_SEVERE_LATE_RATE"] = (
            features["INSTALLMENT_SEVERE_LATE_COUNT"]
            / features["INSTALLMENT_COUNT"]
        )

        features["INSTALLMENT_UNDERPAYMENT_RATE"] = (
            features["INSTALLMENT_UNDERPAYMENT_COUNT"]
            / features["INSTALLMENT_COUNT"]
        )

        return features