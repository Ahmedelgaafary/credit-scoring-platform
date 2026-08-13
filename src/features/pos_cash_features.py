from __future__ import annotations

import pandas as pd


class POSCashFeatureEngineer:
    """Create applicant-level features from POS/CASH history."""

    @staticmethod
    def create_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate POS/CASH history by SK_ID_CURR.
        """

        data = df.copy()

        # ----------------------------------------------------------
        # Delinquency
        # ----------------------------------------------------------

        data["LATE_PAYMENT"] = (
            data["SK_DPD"] > 0
        ).astype(int)

        data["SEVERE_LATE_PAYMENT"] = (
            data["SK_DPD"] > 30
        ).astype(int)

        # ----------------------------------------------------------
        # Contract completion / status
        # ----------------------------------------------------------

        data["ACTIVE_CONTRACT"] = (
            data["NAME_CONTRACT_STATUS"] == "Active"
        ).astype(int)

        data["COMPLETED_CONTRACT"] = (
            data["NAME_CONTRACT_STATUS"] == "Completed"
        ).astype(int)

        data["SIGNED_CONTRACT"] = (
            data["NAME_CONTRACT_STATUS"] == "Signed"
        ).astype(int)

        # ----------------------------------------------------------
        # Aggregate POS/CASH behavior
        # ----------------------------------------------------------

        features = (
            data.groupby("SK_ID_CURR")
            .agg(
                POS_CASH_LOAN_COUNT=(
                    "SK_ID_PREV",
                    "nunique",
                ),
                POS_CASH_RECORD_COUNT=(
                    "SK_ID_PREV",
                    "count",
                ),
                POS_CASH_DPD_MEAN=(
                    "SK_DPD",
                    "mean",
                ),
                POS_CASH_DPD_MAX=(
                    "SK_DPD",
                    "max",
                ),
                POS_CASH_DPD_TOTAL=(
                    "SK_DPD",
                    "sum",
                ),
                POS_CASH_DPD_DEF_MEAN=(
                    "SK_DPD_DEF",
                    "mean",
                ),
                POS_CASH_DPD_DEF_MAX=(
                    "SK_DPD_DEF",
                    "max",
                ),
                POS_CASH_INSTALLMENTS_MEAN=(
                    "CNT_INSTALMENT",
                    "mean",
                ),
                POS_CASH_INSTALLMENTS_MAX=(
                    "CNT_INSTALMENT",
                    "max",
                ),
                POS_CASH_REMAINING_INSTALLMENTS_MEAN=(
                    "CNT_INSTALMENT_FUTURE",
                    "mean",
                ),
                POS_CASH_REMAINING_INSTALLMENTS_MIN=(
                    "CNT_INSTALMENT_FUTURE",
                    "min",
                ),
                POS_CASH_REMAINING_INSTALLMENTS_MAX=(
                    "CNT_INSTALMENT_FUTURE",
                    "max",
                ),
                POS_CASH_LATE_COUNT=(
                    "LATE_PAYMENT",
                    "sum",
                ),
                POS_CASH_SEVERE_LATE_COUNT=(
                    "SEVERE_LATE_PAYMENT",
                    "sum",
                ),
                POS_CASH_ACTIVE_COUNT=(
                    "ACTIVE_CONTRACT",
                    "sum",
                ),
                POS_CASH_COMPLETED_COUNT=(
                    "COMPLETED_CONTRACT",
                    "sum",
                ),
                POS_CASH_SIGNED_COUNT=(
                    "SIGNED_CONTRACT",
                    "sum",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Behavioral rates
        # ----------------------------------------------------------

        features["POS_CASH_LATE_RATE"] = (
            features["POS_CASH_LATE_COUNT"]
            / features["POS_CASH_RECORD_COUNT"]
        )

        features["POS_CASH_SEVERE_LATE_RATE"] = (
            features["POS_CASH_SEVERE_LATE_COUNT"]
            / features["POS_CASH_RECORD_COUNT"]
        )

        features["POS_CASH_COMPLETION_RATE"] = (
            features["POS_CASH_COMPLETED_COUNT"]
            / features["POS_CASH_LOAN_COUNT"]
        )

        return features