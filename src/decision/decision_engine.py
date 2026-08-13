"""
                        decision_engine.py

                        Part of Credit Scoring Platform.
                        """
from __future__ import annotations


class DecisionEngine:
    """
    Determine the final credit decision from
    probability of default and risk grade.
    """

    def __init__(
        self,
        approval_pd: float = 0.10,
        review_pd: float = 0.20,
    ):
        self.approval_pd = approval_pd
        self.review_pd = review_pd

    def decide(
        self,
        probability_of_default: float,
        risk_grade: str,
    ) -> dict:
        """
        Generate a credit decision.

        Rules:
            PD <= 10%  -> APPROVE
            PD <= 20%  -> REVIEW
            PD > 20%   -> DECLINE
        """

        if not 0.0 <= probability_of_default <= 1.0:
            raise ValueError(
                "Probability of default must be between 0 and 1."
            )

        risk_grade = str(risk_grade).upper()

        if probability_of_default <= self.approval_pd:
            decision = "APPROVE"
            reason = (
                "Applicant meets the automatic "
                "approval criteria."
            )

        elif probability_of_default <= self.review_pd:
            decision = "REVIEW"
            reason = (
                "Applicant falls within the "
                "manual-review risk range."
            )

        else:
            decision = "DECLINE"
            reason = (
                "Applicant exceeds the maximum "
                "acceptable credit-risk level."
            )

        return {
            "decision": decision,
            "reason": reason,
            "probability_of_default": probability_of_default,
            "risk_grade": risk_grade,
        }

    def assess(
        self,
        probability_of_default: float,
        risk_grade: str,
    ) -> dict:
        """
        Alias for decide().
        """

        return self.decide(
            probability_of_default,
            risk_grade,
        )