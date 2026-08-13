from __future__ import annotations

import pandas as pd


class ApplicationFeatureEngineer:
    """Create applicant-level features from application data."""

    @staticmethod
    def create_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create application-level features."""

        data = df.copy()

        # ----------------------------------------------------------
        # Basic financial ratios
        # ----------------------------------------------------------

        data["CREDIT_INCOME_RATIO"] = (
            data["AMT_CREDIT"]
            / data["AMT_INCOME_TOTAL"].replace(0, pd.NA)
        )

        data["ANNUITY_INCOME_RATIO"] = (
            data["AMT_ANNUITY"]
            / data["AMT_INCOME_TOTAL"].replace(0, pd.NA)
        )

        data["CREDIT_ANNUITY_RATIO"] = (
            data["AMT_CREDIT"]
            / data["AMT_ANNUITY"].replace(0, pd.NA)
        )

        data["GOODS_PRICE_CREDIT_RATIO"] = (
            data["AMT_GOODS_PRICE"]
            / data["AMT_CREDIT"].replace(0, pd.NA)
        )

        # ----------------------------------------------------------
        # Employment / age
        # ----------------------------------------------------------

        data["AGE_YEARS"] = (
            -data["DAYS_BIRTH"] / 365.25
        )

        data["EMPLOYMENT_YEARS"] = (
            -data["DAYS_EMPLOYED"] / 365.25
        )

        # ----------------------------------------------------------
        # External credit scores
        # ----------------------------------------------------------

        external_sources = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ]

        available_sources = [
            column
            for column in external_sources
            if column in data.columns
        ]

        if available_sources:

            data["EXT_SOURCE_MEAN"] = (
                data[available_sources]
                .mean(axis=1)
            )

            data["EXT_SOURCE_MIN"] = (
                data[available_sources]
                .min(axis=1)
            )

            data["EXT_SOURCE_MAX"] = (
                data[available_sources]
                .max(axis=1)
            )

            data["EXT_SOURCE_STD"] = (
                data[available_sources]
                .std(axis=1)
            )

        # ----------------------------------------------------------
        # Family / income
        # ----------------------------------------------------------

        data["INCOME_PER_PERSON"] = (
            data["AMT_INCOME_TOTAL"]
            / data["CNT_FAM_MEMBERS"].replace(
                0,
                pd.NA,
            )
        )

        data["CHILDREN_RATIO"] = (
            data["CNT_CHILDREN"]
            / data["CNT_FAM_MEMBERS"].replace(
                0,
                pd.NA,
            )
        )

        # ----------------------------------------------------------
        # Age / employment relationship
        # ----------------------------------------------------------

        data["EMPLOYMENT_AGE_RATIO"] = (
            data["EMPLOYMENT_YEARS"]
            / data["AGE_YEARS"].replace(
                0,
                pd.NA,
            )
        )

        # ----------------------------------------------------------
        # Keep applicant identifier
        # ----------------------------------------------------------

        feature_columns = [
            "SK_ID_CURR",
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "CREDIT_ANNUITY_RATIO",
            "GOODS_PRICE_CREDIT_RATIO",
            "AGE_YEARS",
            "EMPLOYMENT_YEARS",
            "INCOME_PER_PERSON",
            "CHILDREN_RATIO",
            "EMPLOYMENT_AGE_RATIO",
        ]

        feature_columns.extend(
            [
                column
                for column in [
                    "EXT_SOURCE_MEAN",
                    "EXT_SOURCE_MIN",
                    "EXT_SOURCE_MAX",
                    "EXT_SOURCE_STD",
                ]
                if column in data.columns
            ]
        )

        return data[feature_columns].copy()