from src.decision import DecisionEngine


def test_decision_engine():
    engine = DecisionEngine()

    test_cases = [
        (0.03, "C", "APPROVE"),
        (0.10, "C", "APPROVE"),
        (0.15, "D", "REVIEW"),
        (0.25, "D", "DECLINE"),
        (0.40, "E", "DECLINE"),
    ]

    for pd, grade, expected_decision in test_cases:
        result = engine.decide(pd, grade)

        assert result["decision"] == expected_decision
        assert "reason" in result
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0