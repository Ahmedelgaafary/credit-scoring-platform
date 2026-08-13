from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.explainability.explainer import CreditSHAPExplainer
from src.risk.calibration import ProbabilityCalibrator
from src.scoring.credit_score import CreditScoreCalculator
from src.scoring.risk_grade import RiskGradeCalculator
from src.decision.decision_engine import DecisionEngine


class CreditScoringService:
    """
    End-to-end credit scoring service.

    Pipeline:

        Applicant
            ↓
        Preprocessor
            ↓
        XGBoost
            ↓
        Raw PD
            ↓
        Probability Calibration
            ↓
        Calibrated PD
            ↓
        Credit Score
            ↓
        Risk Grade
            ↓
        Decision
            ↓
        SHAP Explanation
    """

    def __init__(
        self,
        model_path: str | Path = (
            "models/champion_xgboost.joblib"
        ),
        preprocessor_path: str | Path = (
            "models/champion_preprocessor.joblib"
        ),
        calibrator_path: str | Path = (
            "models/probability_calibrator.joblib"
        ),
    ):
        self.model_path = Path(model_path)

        self.preprocessor_path = Path(
            preprocessor_path
        )

        self.calibrator_path = Path(
            calibrator_path
        )

        # ----------------------------------------------------------
        # Load ML artifacts
        # ----------------------------------------------------------

        self.model = joblib.load(
            self.model_path
        )

        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

        self.calibrator = (
            ProbabilityCalibrator.load(
                self.calibrator_path
            )
        )

        # ----------------------------------------------------------
        # Scoring layer
        # ----------------------------------------------------------

        self.score_calculator = (
            CreditScoreCalculator()
        )

        self.grade_calculator = (
            RiskGradeCalculator()
        )

        # ----------------------------------------------------------
        # Decision layer
        # ----------------------------------------------------------

        self.decision_engine = (
            DecisionEngine()
        )

        # ----------------------------------------------------------
        # SHAP explainer
        # ----------------------------------------------------------

        self.shap_explainer = (
            CreditSHAPExplainer(
                model_path=self.model_path,
                preprocessor_path=(
                    self.preprocessor_path
                ),
            )
        )

    # ------------------------------------------------------------------
    # Validate applicant
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_applicant(
        applicant: pd.DataFrame,
    ) -> None:
        """
        Validate applicant input.
        """

        if not isinstance(
            applicant,
            pd.DataFrame,
        ):
            raise TypeError(
                "Applicant must be a pandas DataFrame."
            )

        if applicant.empty:
            raise ValueError(
                "Applicant data is empty."
            )

        if len(applicant) != 1:
            raise ValueError(
                "Exactly one applicant is required."
            )

    # ------------------------------------------------------------------
    # Prepare model input
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_features(
        applicant: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove non-model fields from applicant data.
        """

        X = applicant.drop(
            columns=["TARGET"],
            errors="ignore",
        )

        X = X.drop(
            columns=["SK_ID_CURR"],
            errors="ignore",
        )

        return X

    # ------------------------------------------------------------------
    # Generate SHAP explanation
    # ------------------------------------------------------------------

    def _generate_explanation(
        self,
        X: pd.DataFrame,
        top_n: int = 10,
    ) -> list[dict]:
        """
        Generate SHAP explanation for one applicant.
        """

        shap_result = (
            self.shap_explainer.explain_applicant(
                X,
                top_n=top_n,
            )
        )

        explanation = (
            shap_result[
                [
                    "feature",
                    "shap_value",
                    "direction",
                ]
            ]
            .to_dict(
                orient="records"
            )
        )

        return explanation

    # ------------------------------------------------------------------
    # Score applicant
    # ------------------------------------------------------------------

    def score_applicant(
        self,
        applicant: pd.DataFrame,
        include_explanation: bool = True,
        explanation_top_n: int = 10,
    ) -> dict:
        """
        Generate a complete credit assessment.

        Returns:

            probability_of_default
            credit_score
            risk_grade
            decision
            reason
            explanation
        """

        self._validate_applicant(
            applicant
        )

        # ----------------------------------------------------------
        # Prepare model features
        # ----------------------------------------------------------

        X = self._prepare_features(
            applicant
        )

        # ----------------------------------------------------------
        # Raw model prediction
        # ----------------------------------------------------------

        X_processed = (
            self.preprocessor.transform(
                X
            )
        )

        raw_probability = float(
            self.model.predict_proba(
                X_processed
            )[0, 1]
        )

        # ----------------------------------------------------------
        # Probability calibration
        # ----------------------------------------------------------

        calibrated_probability = float(
            self.calibrator.predict(
                [raw_probability]
            )[0]
        )

        # ----------------------------------------------------------
        # Credit score
        # ----------------------------------------------------------

        credit_score = (
            self.score_calculator.pd_to_score(
                calibrated_probability
            )
        )

        # ----------------------------------------------------------
        # Risk grade
        # ----------------------------------------------------------

        risk_grade = (
            self.grade_calculator.score_to_grade(
                credit_score
            )
        )

        # ----------------------------------------------------------
        # Business decision
        # ----------------------------------------------------------

        decision_result = (
            self.decision_engine.decide(
                probability_of_default=(
                    calibrated_probability
                ),
                risk_grade=risk_grade,
            )
        )

        # ----------------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------------

        if include_explanation:

            explanation = (
                self._generate_explanation(
                    X,
                    top_n=explanation_top_n,
                )
            )

        else:

            explanation = []

        # ----------------------------------------------------------
        # Final assessment
        # ----------------------------------------------------------

        return {
            "probability_of_default": (
                calibrated_probability
            ),
            "credit_score": (
                credit_score
            ),
            "risk_grade": (
                risk_grade
            ),
            "decision": (
                decision_result["decision"]
            ),
            "reason": (
                decision_result["reason"]
            ),
            "explanation": explanation,
        }