from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CreditDecision:
    probability_of_default: float
    credit_score: int
    risk_grade: str
    decision: str
    reason: str


class CreditDecisionEngine:
    """
    Business decision engine for credit applications.

    Decisions:
        APPROVE
        REVIEW
        DECLINE
    """

    def __init__(
        self,
        approve_max_pd: float = 0.10,
        review_max_pd: float = 0.20,
        approve_min_score: int = 650,
        review_min_score: int = 600,
    ):
        self.approve_max_pd = approve_max_pd
        self.review_max_pd = review_max_pd
        self.approve_min_score = approve_min_score
        self.review_min_score = review_min_score

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(
        self,
        probability_of_default: float,
        credit_score: int,
        risk_grade: str,
    ) -> CreditDecision:
        """
        Generate a credit decision.
        """

        pd_value = float(
            probability_of_default
        )

        # --------------------------------------------------------------
        # APPROVE
        # --------------------------------------------------------------

        if (
            pd_value <= self.approve_max_pd
            and credit_score >= self.approve_min_score
        ):
            return CreditDecision(
                probability_of_default=pd_value,
                credit_score=credit_score,
                risk_grade=risk_grade,
                decision="APPROVE",
                reason=(
                    "Applicant meets the "
                    "automatic approval criteria."
                ),
            )

        # --------------------------------------------------------------
        # DECLINE
        # --------------------------------------------------------------

        if (
            pd_value > self.review_max_pd
            or credit_score < self.review_min_score
        ):
            return CreditDecision(
                probability_of_default=pd_value,
                credit_score=credit_score,
                risk_grade=risk_grade,
                decision="DECLINE",
                reason=(
                    "Applicant exceeds the "
                    "maximum acceptable credit-risk level."
                ),
            )

        # --------------------------------------------------------------
        # MANUAL REVIEW
        # --------------------------------------------------------------

        return CreditDecision(
            probability_of_default=pd_value,
            credit_score=credit_score,
            risk_grade=risk_grade,
            decision="REVIEW",
            reason=(
                "Applicant falls within the "
                "manual-review risk range."
            ),
        )

    # ------------------------------------------------------------------
    # Dictionary output
    # ------------------------------------------------------------------

    def assess(
        self,
        probability_of_default: float,
        credit_score: int,
        risk_grade: str,
    ) -> dict:
        """
        Return decision as a dictionary.
        """

        result = self.decide(
            probability_of_default=(
                probability_of_default
            ),
            credit_score=credit_score,
            risk_grade=risk_grade,
        )

        return {
            "probability_of_default":
                result.probability_of_default,
            "credit_score":
                result.credit_score,
            "risk_grade":
                result.risk_grade,
            "decision":
                result.decision,
            "reason":
                result.reason,
        }
        