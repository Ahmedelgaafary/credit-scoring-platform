from pathlib import Path

import pandas as pd

from src.explainability.shap_explainer import (
    CreditSHAPExplainer,
)


DATASET = (
    "data/processed/"
    "credit_scoring_features.parquet"
)

MODEL = (
    "models/"
    "champion_xgboost.joblib"
)

PREPROCESSOR = (
    "models/"
    "champion_preprocessor.joblib"
)


def main():

    print("=" * 70)
    print("CREDIT RISK SHAP EXPLANATION")
    print("=" * 70)

    df = pd.read_parquet(
        DATASET
    )

    # Remove target and identifier.
    X = df.drop(
        columns=[
            "TARGET",
            "SK_ID_CURR",
        ],
        errors="ignore",
    )

    # Use a sample for global SHAP.
    sample = X.sample(
        n=min(5000, len(X)),
        random_state=42,
    )

    explainer = CreditSHAPExplainer(
        model_path=MODEL,
        preprocessor_path=PREPROCESSOR,
    )

    print("\nCalculating global SHAP importance...")

    importance = (
        explainer.global_importance(
            sample,
            top_n=20,
        )
    )

    print("\nTop SHAP features:")
    print(
        importance.to_string(
            index=False
        )
    )

    output_dir = Path(
        "reports/shap"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        output_dir
        / "global_feature_importance.csv",
        index=False,
    )

    print(
        "\nSaved:"
        "\nreports/shap/"
        "global_feature_importance.csv"
    )


if __name__ == "__main__":
    main()