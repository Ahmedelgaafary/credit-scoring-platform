from src.risk.credit_score import CreditScoreEngine


def main():

    engine = CreditScoreEngine()

    test_pds = [
        0.01,
        0.03,
        0.05,
        0.10,
        0.20,
        0.40,
        0.60,
    ]

    print("=" * 60)
    print("CREDIT SCORE TEST")
    print("=" * 60)

    for pd in test_pds:

        result = engine.assess(pd)

        print(
            f"PD: {pd:.2%} | "
            f"Score: {result['credit_score']} | "
            f"Grade: {result['risk_grade']}"
        )


if __name__ == "__main__":
    main()