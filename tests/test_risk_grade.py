from src.scoring import RiskGradeCalculator


def test_risk_grade_conversion():
    calculator = RiskGradeCalculator()

    expected_results = {
        719: "B",
        687: "C",
        672: "C",
        650: "C",
        627: "D",
        598: "E",
        575: "E",
    }

    for score, expected_grade in expected_results.items():
        grade = calculator.score_to_grade(score)

        assert grade == expected_grade