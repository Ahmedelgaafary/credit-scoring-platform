from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


class ProbabilityCalibrator:
    """Calibrate model probabilities for credit-risk use."""

    def __init__(
        self,
        method: str = "isotonic",
    ):
        if method not in {
            "isotonic",
            "sigmoid",
        }:
            raise ValueError(
                "method must be 'isotonic' "
                "or 'sigmoid'"
            )

        self.method = method
        self.calibrator = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        probabilities,
        y_true,
    ) -> "ProbabilityCalibrator":

        probabilities = np.asarray(
            probabilities
        )

        y_true = np.asarray(
            y_true
        )

        if self.method == "isotonic":

            self.calibrator = (
                IsotonicRegression(
                    y_min=0.0,
                    y_max=1.0,
                    out_of_bounds="clip",
                )
            )

        else:

            from sklearn.linear_model import LogisticRegression

            self.calibrator = (
                LogisticRegression(
                    max_iter=1000
                )
            )

        self.calibrator.fit(
            probabilities.reshape(-1, 1)
            if self.method == "sigmoid"
            else probabilities,
            y_true,
        )

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def predict(
        self,
        probabilities,
    ) -> np.ndarray:

        if self.calibrator is None:
            raise RuntimeError(
                "Calibrator has not been fitted."
            )

        probabilities = np.asarray(
            probabilities
        )

        if self.method == "sigmoid":

            return self.calibrator.predict_proba(
                probabilities.reshape(-1, 1)
            )[:, 1]

        return self.calibrator.predict(
            probabilities
        )

    # ------------------------------------------------------------------
    # Brier score
    # ------------------------------------------------------------------

    @staticmethod
    def brier_score(
        y_true,
        probabilities,
    ) -> float:

        return float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:

        if self.calibrator is None:
            raise RuntimeError(
                "Cannot save an unfitted calibrator."
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
    ) -> "ProbabilityCalibrator":

        return joblib.load(path)