from __future__ import annotations

import numpy as np
import pandas as pd


class PreviousApplicationFeatureEngineer:
    """Create applicant-level features from previous applications."""

    @staticmethod
    def create_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate previous application history by SK_ID_CURR."""

        data = df.copy()

        # ----------------------------------------------------------
        # Basic application history
        # ----------------------------------------------------------

        basic_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                PREVIOUS_APPLICATION_COUNT=(
                    "SK_ID_PREV",
                    "count",
                ),
                PREVIOUS_CREDIT_TYPE_COUNT=(
                    "NAME_CONTRACT_TYPE",
                    "nunique",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Application status
        # ----------------------------------------------------------

        data["PREVIOUS_APPROVED"] = (
            data["NAME_CONTRACT_STATUS"]
            == "Approved"
        ).astype(int)

        data["PREVIOUS_REFUSED"] = (
            data["NAME_CONTRACT_STATUS"]
            == "Refused"
        ).astype(int)

        data["PREVIOUS_CANCELED"] = (
            data["NAME_CONTRACT_STATUS"]
            == "Canceled"
        ).astype(int)

        data["PREVIOUS_UNUSED"] = (
            data["NAME_CONTRACT_STATUS"]
            == "Unused offer"
        ).astype(int)

        status_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                PREVIOUS_APPROVED_COUNT=(
                    "PREVIOUS_APPROVED",
                    "sum",
                ),
                PREVIOUS_REFUSED_COUNT=(
                    "PREVIOUS_REFUSED",
                    "sum",
                ),
                PREVIOUS_CANCELED_COUNT=(
                    "PREVIOUS_CANCELED",
                    "sum",
                ),
                PREVIOUS_UNUSED_COUNT=(
                    "PREVIOUS_UNUSED",
                    "sum",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Approval / refusal rates
        # ----------------------------------------------------------

        status_features["PREVIOUS_APPROVAL_RATE"] = (
            status_features[
                "PREVIOUS_APPROVED_COUNT"
            ]
            / basic_features[
                "PREVIOUS_APPLICATION_COUNT"
            ]
        )

        status_features["PREVIOUS_REFUSAL_RATE"] = (
            status_features[
                "PREVIOUS_REFUSED_COUNT"
            ]
            / basic_features[
                "PREVIOUS_APPLICATION_COUNT"
            ]
        )

        # ----------------------------------------------------------
        # Credit amounts
        # ----------------------------------------------------------

        amount_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                PREVIOUS_CREDIT_AMOUNT_MEAN=(
                    "AMT_CREDIT",
                    "mean",
                ),
                PREVIOUS_CREDIT_AMOUNT_MAX=(
                    "AMT_CREDIT",
                    "max",
                ),
                PREVIOUS_CREDIT_AMOUNT_SUM=(
                    "AMT_CREDIT",
                    "sum",
                ),
                PREVIOUS_APPLICATION_AMOUNT_MEAN=(
                    "AMT_APPLICATION",
                    "mean",
                ),
                PREVIOUS_APPLICATION_AMOUNT_MAX=(
                    "AMT_APPLICATION",
                    "max",
                ),
                PREVIOUS_ANNUITY_MEAN=(
                    "AMT_ANNUITY",
                    "mean",
                ),
                PREVIOUS_DOWN_PAYMENT_MEAN=(
                    "AMT_DOWN_PAYMENT",
                    "mean",
                ),
                PREVIOUS_GOODS_PRICE_MEAN=(
                    "AMT_GOODS_PRICE",
                    "mean",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Requested vs granted credit
        # ----------------------------------------------------------

        data["CREDIT_APPLICATION_RATIO"] = (
            data["AMT_CREDIT"]
            / data["AMT_APPLICATION"].replace(
                0,
                np.nan,
            )
        )

        ratio_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                PREVIOUS_CREDIT_APPLICATION_RATIO_MEAN=(
                    "CREDIT_APPLICATION_RATIO",
                    "mean",
                ),
                PREVIOUS_CREDIT_APPLICATION_RATIO_MAX=(
                    "CREDIT_APPLICATION_RATIO",
                    "max",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Previous application timing
        # ----------------------------------------------------------

        timing_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                PREVIOUS_DAYS_DECISION_MEAN=(
                    "DAYS_DECISION",
                    "mean",
                ),
                PREVIOUS_DAYS_DECISION_MIN=(
                    "DAYS_DECISION",
                    "min",
                ),
                PREVIOUS_DAYS_DECISION_MAX=(
                    "DAYS_DECISION",
                    "max",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Contract / payment terms
        # ----------------------------------------------------------

        terms_features = (
            data.groupby("SK_ID_CURR")
            .agg(
                PREVIOUS_INSTALLMENT_COUNT_MEAN=(
                    "CNT_PAYMENT",
                    "mean",
                ),
                PREVIOUS_INSTALLMENT_COUNT_MAX=(
                    "CNT_PAYMENT",
                    "max",
                ),
                PREVIOUS_PRODUCT_COMBINATION_COUNT=(
                    "PRODUCT_COMBINATION",
                    "nunique",
                ),
            )
            .reset_index()
        )

        # ----------------------------------------------------------
        # Merge all feature groups
        # ----------------------------------------------------------

        result = basic_features

        feature_tables = [
            status_features,
            amount_features,
            ratio_features,
            timing_features,
            terms_features,
        ]

        for feature_table in feature_tables:
            result = result.merge(
                feature_table,
                on="SK_ID_CURR",
                how="left",
            )

        return result