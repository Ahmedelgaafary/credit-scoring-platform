# Credit Scoring Platform — Architecture

## 1. Overview

The Credit Scoring Platform is an end-to-end machine learning system for credit risk assessment.

The platform takes applicant and historical credit data, engineers predictive features, generates a probability of default (PD), converts the PD into a credit score and risk grade, makes a credit decision, and provides SHAP-based explanations.

### Main Flow

```text
Raw Credit Data
      ↓
Feature Engineering
      ↓
Feature Pipeline
      ↓
ML Preprocessing
      ↓
Champion XGBoost Model
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
      ↓
Credit Assessment API
```

---

## 2. Project Structure

```text
credit-scoring-platform/
│
├── app/
│   ├── __init__.py
│   └── api.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── models/
│   ├── champion_xgboost.joblib
│   ├── champion_preprocessor.joblib
│   └── probability_calibrator.joblib
│
├── src/
│   ├── config.py
│   │
│   ├── features/
│   │   ├── application_features.py
│   │   ├── bureau_features.py
│   │   ├── bureau_balance_features.py
│   │   ├── previous_application_features.py
│   │   ├── installment_features.py
│   │   ├── credit_card_features.py
│   │   ├── pos_cash_features.py
│   │   └── feature_pipeline.py
│   │
│   ├── ml/
│   │   ├── champion.py
│   │   ├── model_comparison.py
│   │   ├── preprocessing.py
│   │   ├── split.py
│   │   ├── train.py
│   │   └── models/
│   │
│   ├── risk/
│   │   ├── calibration.py
│   │   ├── credit_score.py
│   │   └── decision_engine.py
│   │
│   ├── scoring/
│   │   ├── credit_score.py
│   │   └── risk_grade.py
│   │
│   ├── decision/
│   │   └── decision_engine.py
│   │
│   ├── explainability/
│   │   └── explainer.py
│   │
│   ├── monitoring/
│   │   ├── drift.py
│   │   ├── metrics.py
│   │   └── performance.py
│   │
│   ├── services/
│   │   └── credit_scoring_service.py
│   │
│   └── utils/
│       ├── exceptions.py
│       └── logging.py
│
├── tests/
│
├── notebooks/
├── reports/
├── docs/
│
├── calibrate_model.py
├── compare_models.py
├── explore_shap.py
├── train_champion.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 3. Application Layer

The `app/` package contains the API interface.

### `app/api.py`

Provides the REST API endpoint for credit scoring.

The API receives applicant features and returns:

* Probability of default
* Credit score
* Risk grade
* Credit decision
* Decision reason
* SHAP explanation

The API communicates with the `CreditScoringService`, keeping business logic outside the API layer.

---

## 4. Feature Engineering Layer

The `src/features/` package transforms raw Home Credit data into applicant-level predictive features.

### Application Features

`application_features.py`

Processes the main application dataset and creates applicant-level variables.

### Bureau Features

`bureau_features.py`

Aggregates historical credit bureau information.

### Bureau Balance Features

`bureau_balance_features.py`

Aggregates monthly bureau balance information.

### Previous Application Features

`previous_application_features.py`

Creates features from previous loan applications.

### Installment Features

`installment_features.py`

Creates repayment and payment-behavior features from installment records.

### Credit Card Features

`credit_card_features.py`

Aggregates historical credit-card behavior.

### POS Cash Features

`pos_cash_features.py`

Aggregates POS cash-loan behavior.

### Feature Pipeline

`feature_pipeline.py`

Combines the feature groups into the final applicant-level dataset used by the ML pipeline.

---

## 5. Machine Learning Layer

The `src/ml/` package contains the model development and training pipeline.

### Model Candidates

The platform supports multiple candidate models:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

These are implemented under:

```text
src/ml/models/
```

### Model Comparison

`model_comparison.py`

Evaluates candidate models and compares their performance using classification metrics.

### Champion Model

`champion.py`

Handles champion-model selection and loading.

The current production model is the **XGBoost champion model**.

The trained artifacts are stored in:

```text
models/
```

---

## 6. Preprocessing

`src/ml/preprocessing.py`

Handles preprocessing required before model inference.

The preprocessing object is persisted as:

```text
models/champion_preprocessor.joblib
```

Using the same persisted preprocessing pipeline during inference ensures that production data receives the same transformations used during model training.

---

## 7. Probability Calibration

The raw model probability is calibrated before being used for credit-risk decisions.

```text
Raw XGBoost Probability
          ↓
Probability Calibrator
          ↓
Calibrated Probability of Default
```

The calibrator is stored as:

```text
models/probability_calibrator.joblib
```

This calibrated probability is the primary risk quantity used by the scoring and decision layers.

---

## 8. Risk Scoring

The `src/risk/` and `src/scoring/` packages convert model predictions into business-oriented credit-risk outputs.

### Probability of Default

The model produces:

```text
Probability of Default (PD)
```

For example:

```text
PD = 25.48%
```

### Credit Score

The probability of default is transformed into a credit score between:

```text
300 – 850
```

Higher scores represent lower estimated credit risk.

### Risk Grade

The credit score is mapped into risk grades:

| Grade | Score Range |
| ----- | ----------: |
| A     |     750–850 |
| B     |     700–749 |
| C     |     650–699 |
| D     |     600–649 |
| E     |     300–599 |

---

## 9. Decision Engine

The decision layer converts risk information into a lending decision.

The platform currently uses:

```text
PD < 5%       → APPROVE
5% ≤ PD < 20% → REVIEW
PD ≥ 20%      → DECLINE
```

The decision engine also considers the applicant's risk grade and produces a human-readable reason.

Example:

```text
Probability of Default: 25.48%
Credit Score: 618
Risk Grade: D
Decision: DECLINE
Reason: Applicant exceeds the maximum acceptable credit-risk level.
```

---

## 10. Explainability

The `src/explainability/` package uses SHAP to explain model predictions.

The system identifies the features that have the greatest influence on the applicant's predicted risk.

The explanation separates factors into:

```text
Risk-increasing factors
Protective factors
```

Example output:

```text
# SHAP EXPLANATION

Top factors influencing this applicant's risk:

Feature                                      SHAP Value
-------------------------------------------------------
EXT_SOURCE_2                                    0.4123
EXT_SOURCE_3                                    0.2871
DAYS_BIRTH                                      0.1542
AMT_CREDIT                                     -0.1024
```

This makes the system more transparent and suitable for analysis of individual credit decisions.

---

## 11. Credit Scoring Service

`src/services/credit_scoring_service.py`

The service layer connects the complete scoring pipeline.

Conceptually:

```text
Applicant Features
       ↓
Preprocessor
       ↓
Champion Model
       ↓
Probability Calibration
       ↓
Credit Score
       ↓
Risk Grade
       ↓
Decision Engine
       ↓
SHAP Explanation
       ↓
Final Assessment
```

The service provides one central interface for applicant scoring.

This prevents the API from directly managing individual ML and risk components.

---

## 12. Monitoring

The `src/monitoring/` package provides monitoring functionality.

### Performance Monitoring

`performance.py`

Tracks model performance metrics.

### Metrics

`metrics.py`

Provides evaluation metrics used to measure model quality.

### Drift Detection

`drift.py`

Detects changes in the input-data distribution that may indicate model degradation or population changes.

Monitoring is designed to support future production model governance.

---

## 13. Configuration

`src/config.py`

Centralizes important project configuration including:

* Project paths
* Dataset paths
* Model paths
* Target column
* Applicant ID column
* Credit-score limits
* Risk-grade ranges
* Decision thresholds
* Random seed
* Train/test split configuration

This prevents hard-coded configuration from being distributed throughout the project.

---

## 14. API Architecture

The production request flow is:

```text
Client
  ↓
POST /api/v1/score
  ↓
FastAPI Application
  ↓
CreditScoringService
  ↓
Feature/Model Pipeline
  ↓
Risk Scoring
  ↓
Decision Engine
  ↓
SHAP Explainer
  ↓
JSON Response
```

A successful response contains the complete credit assessment.

---

## 15. Testing Architecture

The `tests/` directory validates the major platform components.

Current test coverage includes:

```text
test_api.py
test_credit_score.py
test_credit_scoring_service.py
test_decision_engine.py
test_decision_layer.py
test_monitoring.py
test_monitoring_unit.py
test_risk_grade.py
test_scoring.py
test_shap.py
```

The current test suite has successfully reached:

```text
9 passed
```

The SHAP-related warnings originate from third-party library deprecation warnings and do not represent test failures.

---

## 16. Model Artifacts

The `models/` directory contains generated production artifacts:

```text
champion_xgboost.joblib
champion_preprocessor.joblib
probability_calibrator.joblib
```

These artifacts are required for model inference.

Generated model artifacts are excluded from normal source-control tracking according to `.gitignore`.

---

## 17. Training Workflow

Model development follows:

```text
Raw Home Credit Dataset
        ↓
Feature Engineering
        ↓
Processed Feature Dataset
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
XGBoost Champion
        ↓
Probability Calibration
        ↓
Persist Model Artifacts
```

Main scripts:

```text
compare_models.py
train_champion.py
calibrate_model.py
explore_shap.py
```

---

## 18. Deployment

The platform includes a `Dockerfile` for containerized deployment.

The intended deployment architecture is:

```text
Docker Container
      ↓
FastAPI
      ↓
Credit Scoring Service
      ↓
Persisted ML Artifacts
      ↓
Credit Assessment
```

This allows the model-serving application to be deployed independently from the training environment.

---

## 19. End-to-End Example

For an applicant:

```text
Applicant ID: 100002
```

The platform can produce:

```text
Probability of Default: 25.48%
Credit Score: 618
Risk Grade: D
Decision: DECLINE
```

The decision is generated automatically by the complete pipeline rather than by manually assigned rules.

---

## 20. Architecture Summary

The platform is organized into clear layers:

```text
┌───────────────────────────────┐
│          API Layer            │
│          app/api.py           │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│        Service Layer          │
│   CreditScoringService        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│      Feature Engineering      │
│          src/features         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Machine Learning        │
│           src/ml              │
│      Champion XGBoost         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Risk & Scoring          │
│       src/risk / scoring      │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Decision Engine         │
│        src/decision           │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Explainability          │
│          SHAP                 │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│      Final Credit Decision    │
└───────────────────────────────┘
```

The architecture separates **data processing, feature engineering, machine learning, risk scoring, decisioning, explainability, monitoring, and API serving**, providing a complete end-to-end credit scoring system.
