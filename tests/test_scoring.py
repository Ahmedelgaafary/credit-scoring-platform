from src.scoring import CreditScoreCalculator


def test_credit_score_conversion():
    calculator = CreditScoreCalculator()

    test_pds = [
        0.01,
        0.03,
        0.05,
        0.10,
        0.20,
        0.40,
        0.60,
    ]

    for pd in test_pds:
        score = calculator.pd_to_score(pd)

        recovered_pd = calculator.score_to_pd(score)

        assert 300 <= score <= 850
        assert 0.0 < recovered_pd < 1.0

        # Conversion should approximately recover the
        # original PD because the score is rounded.
        assert abs(recovered_pd - pd) < 0.01