"""
                        exceptions.py

                        Part of Credit Scoring Platform.
                        """
class CreditScoringError(Exception):
    """Base exception for the credit scoring platform."""


class DataValidationError(CreditScoringError):
    """Raised when input data is invalid."""


class ModelError(CreditScoringError):
    """Raised when a model or model artifact fails."""


class ScoringError(CreditScoringError):
    """Raised when credit scoring fails."""


class DecisionError(CreditScoringError):
    """Raised when the decision engine fails."""


class ExplanationError(CreditScoringError):
    """Raised when SHAP explanation fails."""