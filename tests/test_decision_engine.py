from src.risk.decision_engine import CreditDecisionEngine


def test_credit_decision_engine():
    engine = CreditDecisionEngine()

    test_cases = [
        {
            "pd": 0.03,
            "score": 687,
            "grade": "C",
            "expected_decision": "APPROVE",
        },
        {
            "pd": 0.10,
            "score": 650,
            "grade": "C",
            "expected_decision": "APPROVE",
        },
        {
            "pd": 0.15,
            "score": 640,
            "grade": "D",
            "expected_decision": "REVIEW",
        },
        {
            "pd": 0.25,
            "score": 610,
            "grade": "D",
            "expected_decision": "DECLINE",
        },
        {
            "pd": 0.40,
            "score": 598,
            "grade": "E",
            "expected_decision": "DECLINE",
        },
    ]

    for case in test_cases:
        result = engine.assess(
            probability_of_default=case["pd"],
            credit_score=case["score"],
            risk_grade=case["grade"],
        )

        assert result["probability_of_default"] == case["pd"]
        assert result["credit_score"] == case["score"]
        assert result["risk_grade"] == case["grade"]
        assert result["decision"] == case["expected_decision"]
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0