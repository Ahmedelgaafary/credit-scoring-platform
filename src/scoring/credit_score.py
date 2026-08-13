from __future__ import annotations

import math


class CreditScoreCalculator:
    """
    Convert between Probability of Default (PD)
    and credit score.

    Score range:
        300 = highest risk
        850 = lowest risk

    Reference:
        PD = 10% -> Score = 650

    PDO:
        20 points for doubling the good odds.
    """

    def __init__(
        self,
        min_score: int = 300,
        max_score: int = 850,
    ):
        self.min_score = min_score
        self.max_score = max_score

        self.base_score = 650.0
        self.base_odds = 9.0
        self.points_to_double_odds = 20.0

    # --------------------------------------------------------------
    # PD -> Score
    # --------------------------------------------------------------

    def pd_to_score(
        self,
        probability_of_default: float,
    ) -> int:
        """
        Convert Probability of Default to credit score.
        """

        pd_value = float(
            max(
                0.0001,
                min(
                    probability_of_default,
                    0.9999,
                ),
            )
        )

        good_odds = (
            (1.0 - pd_value)
            / pd_value
        )

        score = (
            self.base_score
            + self.points_to_double_odds
            * math.log2(
                good_odds
                / self.base_odds
            )
        )

        return int(
            max(
                self.min_score,
                min(
                    round(score),
                    self.max_score,
                ),
            )
        )

    # --------------------------------------------------------------
    # Score -> PD
    # --------------------------------------------------------------

    def score_to_pd(
        self,
        score: int | float,
    ) -> float:
        """
        Convert credit score back to Probability of Default.
        """

        score_value = float(score)

        good_odds = (
            self.base_odds
            * 2 ** (
                (
                    score_value
                    - self.base_score
                )
                / self.points_to_double_odds
            )
        )

        probability_of_default = (
            1.0
            / (1.0 + good_odds)
        )

        return float(
            max(
                0.0001,
                min(
                    probability_of_default,
                    0.9999,
                ),
            )
        )

    # --------------------------------------------------------------
    # Score -> Grade
    # --------------------------------------------------------------

    def score_to_grade(
        self,
        score: int | float,
    ) -> str:
        """
        Convert credit score into risk grade.
        """

        score = float(score)

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

    # --------------------------------------------------------------
    # Complete calculation
    # --------------------------------------------------------------

    def calculate(
        self,
        probability_of_default: float,
    ) -> int:
        """
        Calculate credit score from PD.
        """

        return self.pd_to_score(
            probability_of_default
        )