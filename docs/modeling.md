# Credit Scoring Platform — Modeling Documentation

## 1. Overview

The machine learning layer predicts the probability that a credit applicant will default.

The modeling pipeline is designed around:

* Multiple candidate models
* Consistent preprocessing
* Stratified train/validation splitting
* Model comparison
* Champion model selection
* Probability calibration
* Persistent model artifacts
* Production inference

The machine learning implementation is located under:

```text
src/ml/
```

---

## 2. Modeling Architecture

```text
Raw / Engineered Features
        ↓
Feature Preparation
        ↓
Train / Validation Split
        ↓
Preprocessing
        ↓
Candidate Models
        ↓
Model Comparison
        ↓
Champion Selection
        ↓
Probability Calibration
        ↓
Persisted Model Artifacts
        ↓
Production Prediction
```

---

## 3. ML Directory Structure

```text
src/ml/
│
├── __init__.py
├── champion.py
├── model_comparison.py
├── preprocessing.py
├── split.py
├── train.py
│
└── models/
    ├── __init__.py
    ├── lightgbm_model.py
    ├── logistic_regression.py
    ├── random_forest.py
    └── xgboost_model.py
```

---

## 4. Feature Dataset

The modeling pipeline consumes the processed feature dataset:

```text
data/processed/credit_scoring_features.parquet
```

The dataset contains the engineered applicant-level features generated from the Home Credit data.

The target variable is:

```text
TARGET
```

The applicant identifier is:

```text
SK_ID_CURR
```

`TARGET` is used during model training but is excluded from production scoring inputs.

---

## 5. Train / Validation Split

The dataset is divided into training and validation sets.

The configured values are:

```text
Random state: 42
Test size: 20%
Validation size: 20%
```

The split is stratified using the target variable to preserve the default/non-default distribution across the datasets.

This is important because credit-default datasets are highly imbalanced.

---

## 6. Preprocessing

The preprocessing pipeline prepares the feature matrix for machine learning.

The preprocessing stage is responsible for:

* Separating features from the target
* Removing identifier columns where appropriate
* Handling numerical features
* Handling categorical features
* Managing missing values
* Producing a consistent feature representation

The implementation is:

```text
src/ml/preprocessing.py
```

The preprocessing object is persisted with the champion model so that production inference uses the same transformations applied during training.

---

## 7. Candidate Models

The project evaluates several machine learning algorithms.

### Logistic Regression

Implementation:

```text
src/ml/models/logistic_regression.py
```

Logistic regression provides a simple and interpretable baseline.

It is useful for establishing a reference performance level against more complex models.

---

### Random Forest

Implementation:

```text
src/ml/models/random_forest.py
```

Random Forest is a tree-based ensemble model capable of capturing nonlinear relationships and feature interactions.

---

### XGBoost

Implementation:

```text
src/ml/models/xgboost_model.py
```

XGBoost is a gradient-boosting algorithm designed for structured/tabular data.

It is particularly suitable for credit-risk modeling because it can capture nonlinear relationships and complex interactions between applicant characteristics.

---

### LightGBM

Implementation:

```text
src/ml/models/lightgbm_model.py
```

LightGBM provides another gradient-boosting implementation optimized for efficient training on structured datasets.

---

## 8. Model Comparison

Candidate models are evaluated through:

```text
src/ml/model_comparison.py
```

The comparison process evaluates model performance using appropriate classification metrics.

The resulting comparison is saved as:

```text
models/model_comparison.csv
```

The comparison allows the project to select the strongest candidate based on validation performance rather than choosing a model arbitrarily.

---

## 9. Evaluation Metrics

Credit-risk modeling requires more than simple accuracy.

The primary evaluation metrics include:

### ROC-AUC

Measures the model's ability to rank defaulting applicants above non-defaulting applicants.

Higher values indicate better discrimination.

---

### PR-AUC

Precision-Recall AUC is particularly useful for imbalanced classification problems.

It focuses on the model's performance on the positive/default class.

---

### Precision

Measures the proportion of predicted defaults that are actually defaults.

---

### Recall

Measures the proportion of actual defaults correctly identified by the model.

---

### F1 Score

Combines precision and recall into a single metric.

---

## 10. Champion Model

After candidate comparison, the strongest model is selected as the champion.

The champion-model logic is implemented in:

```text
src/ml/champion.py
```

The production champion in the current project is the XGBoost model.

The persisted artifact is:

```text
models/champion_xgboost.joblib
```

---

## 11. Champion Preprocessor

The preprocessing pipeline used by the champion model is persisted separately:

```text
models/champion_preprocessor.joblib
```

Keeping the preprocessing object with the model is important because production inference must reproduce the exact transformations used during training.

The production flow is therefore:

```text
Applicant Features
        ↓
Champion Preprocessor
        ↓
Champion XGBoost
        ↓
Raw Probability
```

---

## 12. Probability Calibration

Raw machine-learning probabilities are not always reliable estimates of actual default probability.

For credit-risk applications, probability calibration is therefore applied after model prediction.

The calibration process is implemented in:

```text
src/risk/calibration.py
```

The trained calibrator is persisted as:

```text
models/probability_calibrator.joblib
```

The production prediction flow becomes:

```text
Applicant
   ↓
Preprocessor
   ↓
XGBoost
   ↓
Raw Probability
   ↓
Probability Calibrator
   ↓
Calibrated Probability of Default
```

---

## 13. Why Calibration Matters

Suppose a model predicts:

```text
PD = 0.20
```

This should ideally mean that applicants receiving approximately 20% predicted probability have a default rate close to 20% over an appropriate population.

A well-calibrated model therefore provides probabilities that are more meaningful for downstream credit decisions.

The calibrated probability is used by the risk layer rather than relying directly on the raw model probability.

---

## 14. Risk Layer Integration

The calibrated probability is passed to the scoring and decision components.

```text
Calibrated PD
      ↓
Credit Score
      ↓
Risk Grade
      ↓
Decision Engine
```

The credit-score transformation is implemented in:

```text
src/scoring/credit_score.py
```

The risk-grade transformation is implemented in:

```text
src/scoring/risk_grade.py
```

The final credit decision is implemented in:

```text
src/risk/decision_engine.py
```

---

## 15. Credit Score

The platform converts probability of default into a credit score on a:

```text
300–850
```

scale.

Higher scores represent lower estimated credit risk.

The score is calculated from the calibrated probability rather than directly from the model's raw output.

---

## 16. Risk Grades

The current risk-grade mapping is:

| Grade | Score Range |
| ----- | ----------: |
| A     |     750–850 |
| B     |     700–749 |
| C     |     650–699 |
| D     |     600–649 |
| E     |     300–599 |

This provides a categorical representation of the numerical credit score.

---

## 17. Model-to-Decision Pipeline

The complete modeling and risk pipeline is:

```text
Engineered Features
        ↓
Preprocessing
        ↓
Champion XGBoost
        ↓
Raw Default Probability
        ↓
Probability Calibration
        ↓
Calibrated PD
        ↓
Credit Score
        ↓
Risk Grade
        ↓
Decision
```

---

## 18. Explainability

The production prediction is accompanied by SHAP-based explanations.

The explainability implementation is:

```text
src/explainability/explainer.py
```

SHAP identifies features that contributed to the model's prediction.

The explanation distinguishes between:

```text
increases_risk
```

and:

```text
decreases_risk
```

This makes the model output easier to interpret for credit-risk analysis.

---

## 19. Model Artifacts

The production model artifacts are stored in:

```text
models/
```

Current artifacts include:

```text
champion_xgboost.joblib
champion_preprocessor.joblib
probability_calibrator.joblib
```

Additional generated evaluation artifacts may include:

```text
champion_metrics.json
validation_predictions.csv
calibrated_validation_predictions.csv
model_comparison.csv
```

Large generated artifacts are kept outside the source-code layer.

---

## 20. Training Scripts

The main training entry point is:

```text
train_champion.py
```

Supporting scripts include:

```text
compare_models.py
calibrate_model.py
explore_shap.py
```

### `train_champion.py`

Trains and persists the selected champion model.

### `compare_models.py`

Runs candidate model comparison.

### `calibrate_model.py`

Fits the probability calibration layer.

### `explore_shap.py`

Provides SHAP-based model explainability analysis.

---

## 21. Production Inference

Production inference does not retrain the model.

Instead, it loads the persisted artifacts:

```text
champion_preprocessor.joblib
champion_xgboost.joblib
probability_calibrator.joblib
```

Then:

```text
Input Features
      ↓
Load Preprocessor
      ↓
Transform Features
      ↓
Load Champion Model
      ↓
Predict Raw PD
      ↓
Load Calibrator
      ↓
Calculate Calibrated PD
```

The resulting PD is passed to the credit-risk layer.

---

## 22. Reproducibility

The modeling pipeline uses:

```text
RANDOM_STATE = 42
```

to make the main randomized operations reproducible.

The preprocessing pipeline and trained artifacts are persisted so that the same transformations can be reproduced during deployment.

---

## 23. Model Testing

The ML and risk layers are covered by automated tests.

Relevant tests include:

```text
tests/test_credit_score.py
tests/test_scoring.py
tests/test_risk_grade.py
tests/test_decision_engine.py
tests/test_credit_scoring_service.py
tests/test_shap.py
tests/test_api.py
```

The current project test suite passes successfully.

---

## 24. Model Governance

The modeling architecture separates:

```text
Training
Evaluation
Calibration
Persistence
Inference
Decision
Explainability
```

This separation makes it possible to replace or retrain the champion model without rewriting the entire credit-scoring application.

---

## 25. Summary

The machine learning system follows a complete production-oriented workflow:

```text
Home Credit Data
      ↓
Feature Engineering
      ↓
Train / Validation Split
      ↓
Preprocessing
      ↓
Model Comparison
      ↓
Champion XGBoost
      ↓
Probability Calibration
      ↓
Credit Score
      ↓
Risk Grade
      ↓
Credit Decision
      ↓
SHAP Explanation
      ↓
REST API
```

The result is an end-to-end credit-risk modeling pipeline that connects machine learning predictions with calibrated probability estimates, credit scoring, risk classification, automated decisions, and model explainability.
