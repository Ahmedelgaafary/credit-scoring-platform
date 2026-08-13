from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class CreditPreprocessor:
    """Prepare credit-scoring features for machine learning."""

    def __init__(self):
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------
    # Build preprocessor
    # ------------------------------------------------------------------

    def build(
        self,
        X: pd.DataFrame,
    ) -> ColumnTransformer:
        """Build preprocessing pipeline from training features."""

        numeric_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

        return self.preprocessor

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
    ) -> "CreditPreprocessor":
        """Fit preprocessing using training data only."""

        self.build(X)

        self.preprocessor.fit(X)

        self._set_feature_names()

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        X: pd.DataFrame,
    ):
        """Transform features using fitted preprocessor."""

        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        return self.preprocessor.transform(X)

    # ------------------------------------------------------------------
    # Fit + transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: pd.DataFrame,
    ):
        """Fit and transform training features."""

        self.fit(X)

        return self.transform(X)

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def _set_feature_names(self) -> None:
        """Store transformed feature names."""

        if self.preprocessor is None:
            return

        try:
            self.feature_names_ = (
                self.preprocessor
                .get_feature_names_out()
                .tolist()
            )
        except AttributeError:
            self.feature_names_ = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save fitted preprocessor."""

        if self.preprocessor is None:
            raise RuntimeError(
                "Cannot save an unfitted preprocessor."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self,
            path,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "CreditPreprocessor":
        """Load fitted preprocessor."""

        return joblib.load(path)