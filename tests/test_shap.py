import pandas as pd

from src.explainability import CreditSHAPExplainer


def test_shap_explanation():
    applicant = pd.read_parquet(
        "data/processed/credit_scoring_features.parquet"
    ).head(1)

    explainer = CreditSHAPExplainer(
        model_path="models/champion_xgboost.joblib",
        preprocessor_path="models/champion_preprocessor.joblib",
    )

    result = explainer.explain_applicant(
        applicant,
        top_n=10,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
    assert "feature" in result.columns
    assert "shap_value" in result.columns
    assert "abs_shap" in result.columns
    assert "direction" in result.columns