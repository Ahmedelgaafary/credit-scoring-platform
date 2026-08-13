from __future__ import annotations

import pandas as pd
import requests


API_URL = "http://127.0.0.1:8000/api/v1/score"

DATASET = (
    "data/processed/"
    "credit_scoring_features.parquet"
)


def test_credit_scoring_api():
    df = pd.read_parquet(DATASET)

    applicant = df.iloc[0].copy()

    applicant = applicant.drop(
        labels=["TARGET"],
        errors="ignore",
    )

    features = {}

    for column, value in applicant.items():

        if pd.isna(value):
            features[column] = None

        else:
            features[column] = (
                value.item()
                if hasattr(value, "item")
                else value
            )

    payload = {
        "features": features
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=60,
    )

    assert response.status_code == 200

    result = response.json()

    assert "assessment" in result

    assessment = result["assessment"]

    # --------------------------------------------------------------
    # Validate credit assessment
    # --------------------------------------------------------------

    assert "probability_of_default" in assessment
    assert "credit_score" in assessment
    assert "risk_grade" in assessment
    assert "decision" in assessment
    assert "reason" in assessment
    assert "explanation" in assessment

    pd_value = assessment[
        "probability_of_default"
    ]

    assert 0.0 <= pd_value <= 1.0

    assert 300 <= assessment["credit_score"] <= 850

    assert assessment["risk_grade"] in {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    }

    assert assessment["decision"] in {
        "APPROVE",
        "REVIEW",
        "DECLINE",
    }

    assert isinstance(
        assessment["reason"],
        str,
    )

    assert len(
        assessment["reason"]
    ) > 0

    # --------------------------------------------------------------
    # Validate SHAP explanation
    # --------------------------------------------------------------

    explanation = assessment[
        "explanation"
    ]

    assert isinstance(
        explanation,
        list,
    )

    assert len(
        explanation
    ) > 0

    for item in explanation:

        assert "feature" in item
        assert "shap_value" in item
        assert "direction" in item

        assert item["direction"] in {
            "increases_risk",
            "decreases_risk",
        }