# Model Development and Evaluation Report

## 1. Overview

This report documents the machine-learning development process used by the Credit Scoring Platform.

The objective is to build a reliable probability-of-default model and integrate it into the complete credit-risk pipeline.

The modeling workflow is:

```text
Processed Features
        ↓
Preprocessing
        ↓
Train / Validation Split
        ↓
Candidate Models
        ↓
Model Comparison
        ↓
Champion Selection
        ↓
Probability Calibration
        ↓
Production Model
```

---

## 2. Modeling Objective

The primary machine-learning task is binary classification.

The model predicts:

```text
TARGET
```

representing whether an applicant defaults.

The principal model output is:

```text
Probability of Default (PD)
```

This probability is subsequently used by the risk-scoring and decision layers.

---

## 3. Input Dataset

The modeling pipeline uses the engineered feature dataset:

```text
data/processed/credit_scoring_features.parquet
```

The dataset contains applicant-level features generated from the Home Credit source datasets.

The target column is:

```text
TARGET
```

The applicant identifier is:

```text
SK_ID_CURR
```

The target and identifier are handled separately from predictive features where appropriate.

---

## 4. Preprocessing

The preprocessing implementation is located at:

```text
src/ml/preprocessing.py
```

The preprocessing stage prepares the feature matrix for the candidate machine-learning algorithms.

The production preprocessing artifact is:

```text
models/champion_preprocessor.joblib
```

The same fitted preprocessing pipeline must be used for inference to ensure consistency between training and production.

---

## 5. Train / Validation Split

The splitting logic is implemented in:

```text
src/ml/split.py
```

The project configuration uses:

```text
Random state: 42
Test size: 20%
Validation size: 20%
```

The fixed random state provides reproducibility during model development.

---

## 6. Candidate Models

The platform evaluates multiple machine-learning algorithms.

Implementations are located under:

```text
src/ml/models/
```

The candidate models are:

### Logistic Regression

A linear baseline model used to provide a simple reference point.

```text
src/ml/models/logistic_regression.py
```

### Random Forest

A tree-ensemble model capable of capturing nonlinear relationships.

```text
src/ml/models/random_forest.py
```

### XGBoost

A gradient-boosted tree model used as one of the primary high-performance candidates.

```text
src/ml/models/xgboost_model.py
```

### LightGBM

Another gradient-boosting implementation used for model comparison.

```text
src/ml/models/lightgbm_model.py
```

---

## 7. Model Comparison

Model comparison is implemented in:

```text
src/ml/model_comparison.py
```

The purpose of the comparison stage is to evaluate candidate models consistently before selecting the production champion.

The comparison artifact is:

```text
models/model_comparison.csv
```

The evaluation process also produces validation predictions.

---

## 8. Champion Model

The selected champion model is:

```text
XGBoost
```

The trained artifact is:

```text
models/champion_xgboost.joblib
```

The champion-management logic is implemented in:

```text
src/ml/champion.py
```

The champion model is the model used by the production scoring pipeline.

---

## 9. Model Artifacts

The main model artifacts are:

```text
models/
├── champion_xgboost.joblib
├── champion_preprocessor.joblib
├── probability_calibrator.joblib
├── champion_metrics.json
├── model_comparison.csv
├── validation_predictions.csv
└── calibrated_validation_predictions.csv
```

These artifacts separate model training from production inference.

---

## 10. Model Performance

The project records the champion-model evaluation results in:

```text
models/champion_metrics.json
```

Candidate-model results are stored in:

```text
models/model_comparison.csv
```

The recorded metrics should be treated as the authoritative results of the completed training run rather than manually reproduced values.

The evaluation framework is designed to compare candidate models using classification-performance metrics relevant to credit-risk modeling.

---

## 11. ROC-AUC

ROC-AUC measures the model's ability to distinguish between defaulting and non-defaulting applicants across classification thresholds.

A higher ROC-AUC indicates stronger ranking ability.

ROC-AUC is useful for comparing candidate credit-risk models because the model ultimately produces a probability score rather than a single fixed classification.

---

## 12. Precision-Recall Evaluation

Because credit-default prediction can involve class imbalance, precision-recall analysis is also important.

Precision measures the proportion of predicted positive cases that are actually positive.

Recall measures the proportion of actual positive cases identified by the model.

Precision-recall metrics provide additional information about model behavior when the default class is relatively uncommon.

---

## 13. Probability Calibration

Raw model probabilities are passed through a calibration stage.

The calibration implementation is:

```text
src/risk/calibration.py
```

The calibration artifact is:

```text
models/probability_calibrator.joblib
```

The process is:

```text
XGBoost Prediction
        ↓
Raw Probability
        ↓
Probability Calibration
        ↓
Calibrated PD
```

This calibrated probability is used by the downstream credit-risk system.

---

## 14. Why Calibration Matters

A model can rank applicants correctly while still producing poorly calibrated probabilities.

For example, applicants predicted with a 10% probability should ideally represent a population with approximately that level of observed default frequency over the relevant evaluation horizon.

Calibration therefore improves the suitability of the model output for:

* Risk assessment
* Credit scoring
* Decision thresholds
* Portfolio analysis
* Risk reporting

---

## 15. Credit Score Transformation

The calibrated PD is converted into a credit score.

The scoring implementation is:

```text
src/scoring/credit_score.py
```

The configured score range is:

```text
300–850
```

The resulting score provides a user-friendly representation of the applicant's estimated credit risk.

---

## 16. Risk Grade

The credit score is converted into a risk grade.

The implementation is:

```text
src/scoring/risk_grade.py
```

The configured grades are:

| Grade | Score Range |
| ----- | ----------: |
| A     |     750–850 |
| B     |     700–749 |
| C     |     650–699 |
| D     |     600–649 |
| E     |     300–599 |

The grade provides a categorical representation of the applicant's risk level.

---

## 17. Credit Decision

The decision engine consumes the calibrated PD and risk information.

The implementation is:

```text
src/risk/decision_engine.py
```

The configured decision thresholds are:

| PD      | Decision |
| ------- | -------- |
| < 5%    | Approve  |
| 5%–<20% | Review   |
| ≥ 20%   | Reject   |

The engine also provides a textual reason for the decision.

---

## 18. Explainability

The model is integrated with SHAP-based explainability.

The implementation is:

```text
src/explainability/explainer.py
```

SHAP identifies features that contribute to the applicant's prediction.

The explanation distinguishes between:

```text
increases_risk
```

and:

```text
decreases_risk
```

This allows the final assessment to provide both a numerical risk estimate and an explanation.

---

## 19. End-to-End Model Pipeline

The complete modeling and risk pipeline is:

```text
Applicant Data
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
Champion XGBoost
      ↓
Raw Probability
      ↓
Probability Calibration
      ↓
Probability of Default
      ↓
Credit Score
      ↓
Risk Grade
      ↓
Decision Engine
      ↓
SHAP Explanation
```

---

## 20. Training Entry Point

The main champion-training script is:

```text
train_champion.py
```

The model-comparison script is:

```text
compare_models.py
```

The calibration script is:

```text
calibrate_model.py
```

The SHAP exploration script is:

```text
explore_shap.py
```

These scripts provide explicit entry points for the major modeling stages.

---

## 21. Model Testing

The project contains automated tests covering the main risk-modeling components.

Relevant tests include:

```text
tests/test_credit_score.py
tests/test_risk_grade.py
tests/test_scoring.py
tests/test_decision_engine.py
tests/test_decision_layer.py
tests/test_credit_scoring_service.py
tests/test_shap.py
```

The API integration test is located at:

```text
tests/test_api.py
```

The completed development test suite has passed successfully.

---

## 22. Model Monitoring

Model monitoring is implemented under:

```text
src/monitoring/
```

The monitoring layer includes:

```text
drift.py
metrics.py
performance.py
```

The system therefore supports monitoring beyond the initial model-development phase.

Monitoring should be used to detect:

* Data distribution changes
* Performance degradation
* Metric deterioration
* Potential retraining requirements

---

## 23. Model Limitations

The model is trained using historical credit data and therefore inherits limitations from the underlying dataset.

Important considerations include:

### Distribution Shift

Future applicants may differ from the historical training population.

### Data Quality

Missing or incorrect applicant information can affect predictions.

### Historical Bias

Historical lending behavior can contain biases that may be reflected in the training data.

### Probability Uncertainty

Predicted probabilities are estimates and should not be interpreted as guarantees.

### Explainability

SHAP values describe model-feature contributions but do not establish causal relationships.

---

## 24. Responsible Use

The model should be used as part of a broader credit-risk decision process.

Production deployment should include appropriate:

* Model governance
* Fairness testing
* Regulatory review
* Data-quality controls
* Security controls
* Human oversight
* Monitoring
* Model-version management

---

## 25. Reproducibility

The project uses a fixed random state:

```text
42
```

Configuration is centralized in:

```text
src/config.py
```

Model artifacts are explicitly stored under:

```text
models/
```

The training and calibration stages are separated into reproducible scripts.

This structure allows model development to be repeated and evaluated consistently.

---

## 26. Model Governance

Changes to the following components should trigger model validation:

* Training data
* Feature engineering
* Feature selection
* Preprocessing
* Model algorithm
* Hyperparameters
* Probability calibration
* Decision thresholds

A new model should only become the production champion after successful evaluation and testing.

---

## 27. Summary

The modeling layer provides the machine-learning foundation of the Credit Scoring Platform.

The final system combines:

```text
Feature Engineering
        +
Machine Learning
        +
Probability Calibration
        +
Credit Scoring
        +
Risk Grading
        +
Decisioning
        +
Explainability
        +
Monitoring
```

The current production-oriented champion is an **XGBoost model**, supported by a serialized preprocessing pipeline and probability calibrator.

The result is an end-to-end credit-risk modeling system designed to transform applicant data into a calibrated, explainable, and actionable credit assessment.
