from src.ml.model_comparison import ModelComparison


def main():

    comparison = ModelComparison(
        model_dir="models"
    )

    comparison.run(
        "data/processed/credit_scoring_features.parquet"
    )


if __name__ == "__main__":
    main()