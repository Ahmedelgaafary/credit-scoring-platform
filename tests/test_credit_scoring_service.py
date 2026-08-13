from pathlib import Path

import pandas as pd

from src.services.credit_scoring_service import CreditScoringService


DATASET = Path(
    "data/processed/credit_scoring_features.parquet"
)


def test_credit_scoring_service():
    df = pd.read_parquet(DATASET)

    applicant = (
        df.drop(
            columns=["TARGET"],
            errors="ignore",
        )
        .iloc[[0]]
        .copy()
    )

    service = CreditScoringService()

    result = service.score_applicant(
        applicant
    )

    # Basic result validation
    assert isinstance(result, dict)

    assert "probability_of_default" in result
    assert "credit_score" in result
    assert "risk_grade" in result
    assert "decision" in result
    assert "reason" in result
    assert "explanation" in result

    # Validate values
    assert 0.0 <= result["probability_of_default"] <= 1.0

    assert isinstance(
        result["credit_score"],
        int,
    )

    assert 300 <= result["credit_score"] <= 850

    assert result["risk_grade"] in {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    }

    assert result["decision"] in {
        "APPROVE",
        "REVIEW",
        "DECLINE",
    }

    assert isinstance(
        result["reason"],
        str,
    )

    assert len(result["reason"]) > 0

    # SHAP explanation
    assert isinstance(
        result["explanation"],
        list,
    )

    assert len(result["explanation"]) > 0

    for item in result["explanation"]:
        assert "feature" in item
        assert "shap_value" in item
        assert "direction" in item