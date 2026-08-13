from __future__ import annotations

import numpy as np
import pandas as pd


class CreditScoreEngine:
    """
    Convert Probability of Default (PD) into a credit score.

    Score range:
        300 = highest risk
        850 = lowest risk
    """

    def __init__(
        self,
        min_score: int = 300,
        max_score: int = 850,
    ):
        self.min_score = min_score
        self.max_score = max_score

    # ------------------------------------------------------------------
    # PD -> Credit Score
    # ------------------------------------------------------------------

    def pd_to_score(
        self,
        probability_of_default: float,
    ) -> int:
        """
        Convert PD to a 300-850 score using log-odds.

        Reference:
            PD = 10% -> Score = 650

        PDO:
            20 points for doubling the good odds.
        """

        pd_value = float(
            np.clip(
                probability_of_default,
                0.0001,
                0.9999,
            )
        )

        good_odds = (
            (1.0 - pd_value)
            / pd_value
        )

        base_score = 650.0
        base_odds = 9.0
        points_to_double_odds = 20.0

        score = (
            base_score
            + points_to_double_odds
            * np.log2(
                good_odds / base_odds
            )
        )

        return int(
            np.clip(
                round(score),
                self.min_score,
                self.max_score,
            )
        )

    # ------------------------------------------------------------------
    # Credit Score -> Risk Grade
    # ------------------------------------------------------------------

    def score_to_grade(
        self,
        score: int,
    ) -> str:
        """
        Convert credit score into a risk grade.
        """

        if score >= 750:
            return "A"

        if score >= 700:
            return "B"

        if score >= 650:
            return "C"

        if score >= 600:
            return "D"

        if score >= 550:
            return "E"

        return "F"

    # ------------------------------------------------------------------
    # Complete assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        probability_of_default: float,
    ) -> dict:
        """
        Generate a complete credit-risk assessment.
        """

        score = self.pd_to_score(
            probability_of_default
        )

        grade = self.score_to_grade(
            score
        )

        return {
            "probability_of_default": float(
                probability_of_default
            ),
            "credit_score": score,
            "risk_grade": grade,
        }

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        probability_column: str = (
            "calibrated_probability_of_default"
        ),
    ) -> pd.DataFrame:
        """
        Add credit score and risk grade
        to a prediction DataFrame.
        """

        result = df.copy()

        result["credit_score"] = (
            result[
                probability_column
            ].apply(
                self.pd_to_score
            )
        )

        result["risk_grade"] = (
            result[
                "credit_score"
            ].apply(
                self.score_to_grade
            )
        )

        return result