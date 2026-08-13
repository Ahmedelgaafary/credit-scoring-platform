from __future__ import annotations


class RiskGradeCalculator:
    """
    Convert credit score into a risk grade.
    """

    def score_to_grade(
        self,
        score: int | float,
    ) -> str:
        """
        Convert credit score into a risk grade.
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

    def calculate(
        self,
        score: int | float,
    ) -> str:
        """
        Alias for score_to_grade().
        """

        return self.score_to_grade(score)