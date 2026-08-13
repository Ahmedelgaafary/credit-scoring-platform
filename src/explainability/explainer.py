"""
                        explainer.py

                        Part of Credit Scoring Platform.
                        """
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


class CreditSHAPExplainer:
    """SHAP explanations for the champion credit-risk model."""

    def __init__(
        self,
        model_path: str | Path,
        preprocessor_path: str | Path,
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(
            preprocessor_path
        )

        self.model = joblib.load(
            self.model_path
        )

        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

        self.feature_names = (
            self.preprocessor
            .feature_names_
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )

    # ------------------------------------------------------------------
    # Transform raw applicant data
    # ------------------------------------------------------------------

    def transform(
        self,
        X: pd.DataFrame,
    ):
        """Apply the saved preprocessing pipeline."""

        return self.preprocessor.transform(
            X
        )

    # ------------------------------------------------------------------
    # SHAP values
    # ------------------------------------------------------------------

    def shap_values(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Calculate SHAP values."""

        X_transformed = self.transform(X)

        shap_values = (
            self.explainer.shap_values(
                X_transformed
            )
        )

        if isinstance(
            shap_values,
            list,
        ):
            shap_values = shap_values[1]

        return np.asarray(
            shap_values
        )

    # ------------------------------------------------------------------
    # Global importance
    # ------------------------------------------------------------------

    def global_importance(
        self,
        X: pd.DataFrame,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Return globally important features."""

        values = self.shap_values(X)

        importance = np.abs(
            values
        ).mean(axis=0)

        result = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": importance,
            }
        )

        result = result.sort_values(
            "mean_abs_shap",
            ascending=False,
        )

        return result.head(
            top_n
        ).reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # Applicant explanation
    # ------------------------------------------------------------------

    def explain_applicant(
        self,
        X: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Explain one applicant."""

        if len(X) != 1:
            raise ValueError(
                "explain_applicant expects "
                "exactly one applicant."
            )

        values = self.shap_values(X)[0]

        result = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": values,
            }
        )

        result["abs_shap"] = (
            result["shap_value"]
            .abs()
        )

        result["direction"] = np.where(
            result["shap_value"] > 0,
            "increases_risk",
            "decreases_risk",
        )

        result = result.sort_values(
            "abs_shap",
            ascending=False,
        )

        return result.head(
            top_n
        ).reset_index(
            drop=True
        )