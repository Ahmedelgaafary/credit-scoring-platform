# Readme
# Credit Scoring Platform

An end-to-end machine-learning credit scoring platform built with Python, XGBoost, probability calibration, SHAP explainability, and a production-style REST API.

The platform transforms applicant and historical credit data into an explainable credit-risk assessment containing:

* Probability of Default (PD)
* Credit Score
* Risk Grade
* Credit Decision
* Decision Reason
* SHAP-based explanation

---

## 1. Project Overview

The system is designed as a modular credit-risk platform rather than a standalone machine-learning notebook.

The complete pipeline is:

```text
Raw Credit Data
       ↓
Feature Engineering
       ↓
Preprocessing
       ↓
Model Training
       ↓
Champion Model Selection
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
REST API
```

---

## 2. Key Features

### Machine Learning

* Multiple candidate models
* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Champion model selection
* Reproducible train/validation splitting
* Model preprocessing pipeline

### Risk Modeling

* Probability of Default prediction
* Probability calibration
* Credit-score conversion
* Risk-grade classification
* Rule-based credit decisioning

### Explainability

* SHAP-based explanations
* Risk-increasing factors
* Risk-decreasing factors
* Applicant-level explanations

### Production Components

* FastAPI-style REST API
* Modular service layer
* Configuration management
* Logging
* Custom exceptions
* Docker support
* Automated tests

### Monitoring

* Data drift monitoring
* Model metrics
* Performance monitoring

---

## 3. Dataset

The platform uses the Home Credit dataset.

The raw dataset contains:

```text
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
credit_card_balance.csv
POS_CASH_balance.csv
```

The feature-engineering pipeline combines information from these sources to construct applicant-level modeling features.

The processed feature dataset is:

```text
data/processed/credit_scoring_features.parquet
```

Raw and processed datasets are excluded from Git tracking.

---

## 4. Champion Model

The production champion model is an XGBoost classifier.

The main artifacts are:

```text
models/
├── champion_xgboost.joblib
├── champion_preprocessor.joblib
└── probability_calibrator.joblib
```

Additional model-evaluation artifacts include:

```text
champion_metrics.json
model_comparison.csv
validation_predictions.csv
calibrated_validation_predictions.csv
```

---

## 5. Credit Risk Pipeline

### Probability of Default

The machine-learning model first predicts the applicant's probability of default.

The probability is then passed through the calibration layer.

```text
Raw Model Probability
        ↓
Probability Calibrator
        ↓
Calibrated Probability of Default
```

### Credit Score

The calibrated PD is converted into a credit score on a:

```text
300–850
```

scale.

### Risk Grade

The platform maps the credit score to five risk grades:

| Grade |   Score |
| ----- | ------: |
| A     | 750–850 |
| B     | 700–749 |
| C     | 650–699 |
| D     | 600–649 |
| E     | 300–599 |

### Decision

The decision engine uses probability-of-default thresholds:

| PD      | Decision |
| ------- | -------- |
| < 5%    | Approve  |
| 5%–<20% | Review   |
| ≥ 20%   | Reject   |

The decision layer also generates a human-readable reason.

---

## 6. Explainability

The platform uses SHAP to explain individual predictions.

An assessment can contain:

```text
Feature
SHAP Value
Direction
```

where the direction identifies whether the feature:

```text
increases_risk
```

or:

```text
decreases_risk
```

This allows users to understand the main factors contributing to an applicant's risk assessment.

---

## 7. API

The application exposes a credit-scoring API.

The API implementation is located at:

```text
app/api.py
```

The main scoring endpoint is:

```text
POST /api/v1/score
```

The request contains applicant features.

The response contains the complete credit assessment.

Example response structure:

```json
{
  "assessment": {
    "probability_of_default": 0.10,
    "credit_score": 650,
    "risk_grade": "C",
    "decision": "REVIEW",
    "reason": "...",
    "explanation": []
  }
}
```

See:

```text
docs/api.md
```

for the complete API documentation.

---

## 8. Project Structure

```text
credit-scoring-platform/
│
├── app/
│   ├── api.py
│   └── __init__.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── docs/
│   ├── api.md
│   ├── architecture.md
│   ├── data_dictionary.md
│   ├── deployment.md
│   ├── methodology.md
│   ├── modeling.md
│   └── monitoring.md
│
├── models/
│   ├── champion_xgboost.joblib
│   ├── champion_preprocessor.joblib
│   └── probability_calibrator.joblib
│
├── src/
│   ├── decision/
│   ├── explainability/
│   ├── features/
│   ├── ml/
│   │   └── models/
│   ├── monitoring/
│   ├── risk/
│   ├── scoring/
│   ├── services/
│   └── utils/
│
├── tests/
│
├── calibrate_model.py
├── compare_models.py
├── explore_shap.py
├── train_champion.py
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 9. Source Code Organization

### `src/features/`

Feature engineering for the different Home Credit datasets.

### `src/ml/`

Machine-learning training, preprocessing, splitting, model comparison, and champion-model management.

### `src/risk/`

Probability calibration, risk scoring, and credit decisioning.

### `src/scoring/`

Credit-score and risk-grade conversion.

### `src/explainability/`

SHAP-based model explanations.

### `src/services/`

Application-level credit-scoring orchestration.

### `src/monitoring/`

Drift, metrics, and performance monitoring.

### `src/utils/`

Logging and custom application exceptions.

---

## 10. Training

The champion model can be trained using:

```bash
python train_champion.py
```

Model comparison can be performed using:

```bash
python compare_models.py
```

Probability calibration can be performed using:

```bash
python calibrate_model.py
```

---

## 11. Testing

The project uses `pytest`.

Run the complete test suite:

```bash
python -m pytest
```

The test suite covers:

```text
API integration
Credit score conversion
Risk-grade conversion
Credit scoring service
Decision engine
Decision layer
Monitoring
SHAP explainability
```

The complete development test suite currently passes successfully.

---

## 12. Docker

The project includes a Docker configuration:

```text
Dockerfile
```

The container can be used to package the API and its runtime dependencies into a reproducible environment.

Deployment instructions are documented in:

```text
docs/deployment.md
```

---

## 13. Documentation

Detailed technical documentation is available in:

| Document             | Description                                     |
| -------------------- | ----------------------------------------------- |
| `architecture.md`    | System architecture and component relationships |
| `methodology.md`     | Data and modeling methodology                   |
| `data_dictionary.md` | Dataset and feature definitions                 |
| `modeling.md`        | Model development and evaluation                |
| `api.md`             | API endpoints and usage                         |
| `deployment.md`      | Deployment instructions                         |
| `monitoring.md`      | Model and data monitoring                       |

---

## 14. Engineering Principles

The project follows several software-engineering principles:

* Modular architecture
* Separation of concerns
* Reusable components
* Configuration-driven behavior
* Automated testing
* Reproducible model training
* Explicit model artifacts
* Explainability
* Production-oriented API design

---

## 15. Limitations

This project is a portfolio and research-oriented implementation.

The model is trained on historical credit data and therefore may experience performance degradation when real-world data distributions change.

Before production lending use, the system would require additional:

* Fairness evaluation
* Regulatory validation
* Security controls
* Data-quality monitoring
* Model governance
* Bias assessment
* Production-scale infrastructure

---

## 16. Technology Stack

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
LightGBM
SHAP
FastAPI
Pytest
Joblib
PyArrow
Docker
Git
```

---

## 17. Project Goal

The goal of this project is to demonstrate how a machine-learning credit-risk model can be developed and integrated into a complete software system.

Rather than stopping at model training, the platform covers the full workflow:

```text
Data
→ Features
→ ML
→ Calibration
→ Risk Scoring
→ Decisioning
→ Explainability
→ API
→ Monitoring
```

This makes the project representative of a realistic **FinTech AI / Credit Risk ML Engineering** workflow.
