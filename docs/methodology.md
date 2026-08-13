# Credit Scoring Platform — Methodology

## 1. Objective

The objective of the Credit Scoring Platform is to estimate the probability that a credit applicant will default and convert that prediction into an interpretable credit-risk assessment.

The platform produces four main outputs:

* Probability of Default (PD)
* Credit Score
* Risk Grade
* Credit Decision

The system additionally provides SHAP-based explanations for individual predictions.

---

## 2. Dataset

The project uses the **Home Credit** dataset.

The dataset contains information about:

* Current loan applications
* Previous loan applications
* Credit bureau records
* Bureau balance history
* Installment payments
* Credit-card balances
* POS cash-loan balances

The primary target variable is:

```text
TARGET
```

where:

```text
TARGET = 1 → Default
TARGET = 0 → No Default
```

The applicant identifier is:

```text
SK_ID_CURR
```

---

## 3. Data Processing

The raw datasets are stored under:

```text
data/raw/
```

The feature-engineering pipeline combines information from the different Home Credit tables using the applicant identifier.

The resulting applicant-level dataset is stored under:

```text
data/processed/
```

The main processed dataset is:

```text
credit_scoring_features.parquet
```

Parquet is used because it provides efficient storage and faster loading compared with repeatedly processing the original CSV files.

---

## 4. Feature Engineering

Feature engineering converts raw financial records into applicant-level risk indicators.

### Application Features

Features are derived directly from the current application.

Examples include:

* Income
* Credit amount
* Annuity
* Employment information
* Age-related variables
* External credit scores
* Housing information

### Bureau Features

Historical bureau records are aggregated to describe the applicant's previous credit exposure.

Examples include:

* Number of previous bureau accounts
* Credit amounts
* Active accounts
* Outstanding balances
* Credit history statistics

### Previous Application Features

Previous loan applications are aggregated to capture historical borrowing behavior.

Examples include:

* Number of previous applications
* Approved applications
* Refused applications
* Previous credit amounts
* Previous loan amounts

### Installment Features

Installment-payment behavior is summarized to capture repayment characteristics.

Examples include:

* Payment delays
* Payment differences
* Number of installments
* Outstanding payment behavior

### Credit Card Features

Credit-card history is aggregated to capture:

* Balance behavior
* Credit utilization
* Payment behavior
* Credit-card activity

### POS Cash Features

POS cash-loan records are aggregated to capture:

* Contract history
* Remaining installments
* Payment behavior
* Account activity

---

## 5. Feature Aggregation

Most historical datasets contain multiple records for a single applicant.

Therefore, the data must be transformed from:

```text
Applicant
    ↓
Multiple historical records
```

into:

```text
Applicant
    ↓
Single feature vector
```

Typical aggregation operations include:

* Mean
* Maximum
* Minimum
* Sum
* Count
* Standard deviation

This produces one machine-learning observation per applicant.

---

## 6. Train / Validation Split

The processed applicant-level dataset is divided into training and validation subsets.

The project configuration uses:

```text
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42
```

The random seed ensures reproducibility.

The target variable is separated from the feature matrix before model training.

---

## 7. Preprocessing

Machine-learning preprocessing is performed using the preprocessing pipeline under:

```text
src/ml/preprocessing.py
```

The preprocessing pipeline handles the transformation of model inputs into a format suitable for machine learning.

The fitted preprocessing object is persisted as:

```text
models/champion_preprocessor.joblib
```

The same fitted preprocessing object is reused during inference.

This prevents training/inference preprocessing inconsistencies.

---

## 8. Candidate Models

Several classification algorithms are supported:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

The models are implemented under:

```text
src/ml/models/
```

The purpose of evaluating multiple models is to determine which algorithm provides the strongest predictive performance for the credit-risk problem.

---

## 9. Model Evaluation

Models are compared using classification metrics appropriate for imbalanced credit-risk data.

Important metrics include:

### ROC-AUC

Measures the ability of the model to rank defaulted applicants above non-defaulted applicants.

### PR-AUC

Provides additional insight into performance when the default class is relatively rare.

### Precision

Measures the proportion of predicted defaults that are actual defaults.

### Recall

Measures the proportion of actual defaults correctly identified by the model.

### F1 Score

Balances precision and recall.

The model comparison results are generated by:

```text
compare_models.py
```

---

## 10. Champion Model

After comparing candidate models, the best-performing model is selected as the champion model.

The current champion is:

```text
XGBoost
```

The trained model is persisted as:

```text
models/champion_xgboost.joblib
```

The champion model is used for production inference.

---

## 11. Probability Calibration

The raw probability produced by the machine-learning model is calibrated before being used as a business risk measure.

The calibration process is implemented in:

```text
src/risk/calibration.py
```

The fitted calibrator is stored as:

```text
models/probability_calibrator.joblib
```

The production prediction pipeline is therefore:

```text
Features
   ↓
Preprocessor
   ↓
XGBoost
   ↓
Raw Probability
   ↓
Probability Calibration
   ↓
Calibrated PD
```

Calibration improves the interpretation of predicted probabilities by making them more representative of observed default frequencies.

---

## 12. Probability of Default

The calibrated probability is interpreted as:

```text
Probability of Default (PD)
```

For example:

```text
PD = 0.03
```

means an estimated:

```text
3% probability of default
```

The PD is the central risk variable used by the downstream scoring and decision layers.

---

## 13. Credit Score

The calibrated PD is converted into a credit score.

The score is constrained to:

```text
300 – 850
```

The scoring system is implemented in:

```text
src/scoring/credit_score.py
```

Lower default probability results in a higher credit score.

Higher default probability results in a lower credit score.

The scoring transformation is designed to be reversible within the supported scoring range so that the corresponding PD can be recovered from the score.

---

## 14. Risk Grade

The credit score is converted into a categorical risk grade.

The current mapping is:

| Grade | Score Range | Risk Level |
| ----- | ----------: | ---------- |
| A     |     750–850 | Very Low   |
| B     |     700–749 | Low        |
| C     |     650–699 | Moderate   |
| D     |     600–649 | High       |
| E     |     300–599 | Very High  |

The grade calculation is implemented in:

```text
src/scoring/risk_grade.py
```

---

## 15. Credit Decision

The decision engine converts the applicant's estimated risk into a lending recommendation.

Current PD thresholds are:

|      PD | Decision |
| ------: | -------- |
|    < 5% | APPROVE  |
| 5%–<20% | REVIEW   |
|   ≥ 20% | DECLINE  |

The decision layer is implemented in:

```text
src/decision/decision_engine.py
```

The risk layer also contains:

```text
src/risk/decision_engine.py
```

which provides the credit-risk decision assessment.

---

## 16. Explainability

The platform uses SHAP to explain individual predictions.

The explainability module is located at:

```text
src/explainability/explainer.py
```

For each applicant, the system identifies influential features and determines whether they increase or decrease the predicted risk.

The explanation is divided into:

```text
Risk-increasing factors
Protective factors
```

This allows the final credit assessment to contain both a numerical decision and an explanation of the main factors behind that decision.

---

## 17. End-to-End Scoring Process

For a new applicant, the complete methodology is:

```text
1. Receive applicant data
        ↓
2. Validate input
        ↓
3. Apply feature transformations
        ↓
4. Apply persisted preprocessing
        ↓
5. Generate XGBoost prediction
        ↓
6. Calibrate probability
        ↓
7. Calculate credit score
        ↓
8. Assign risk grade
        ↓
9. Generate credit decision
        ↓
10. Generate SHAP explanation
        ↓
11. Return credit assessment
```

---

## 18. Final Credit Assessment

The final assessment contains:

```text
Probability of Default
Credit Score
Risk Grade
Decision
Decision Reason
SHAP Explanation
```

Example:

```text
Probability of Default: 25.48%
Credit Score: 618
Risk Grade: D
Decision: DECLINE
Reason: Applicant exceeds the maximum acceptable credit-risk level.
```

The numerical risk estimate, score, grade, and decision are therefore connected through a single reproducible pipeline.

---

## 19. Monitoring Methodology

The platform includes monitoring components for:

### Model Performance

Performance metrics can be tracked to identify deterioration in predictive quality.

### Data Drift

Input distributions can be compared against reference data to identify changes in the applicant population.

### Metrics

The monitoring layer provides reusable functions for calculating model-quality and monitoring metrics.

Monitoring modules are located under:

```text
src/monitoring/
```

---

## 20. Testing Methodology

The platform uses `pytest` for automated testing.

Tests cover:

* Credit-score conversion
* Risk-grade conversion
* Decision engine
* Credit scoring service
* API integration
* SHAP explainability
* Monitoring

The current test suite has reached:

```text
9 passed
```

Warnings reported by third-party dependencies do not represent failed tests.

---

## 21. Production Inference

The production inference architecture uses persisted artifacts rather than retraining the model for every request.

Required artifacts are:

```text
models/champion_xgboost.joblib
models/champion_preprocessor.joblib
models/probability_calibrator.joblib
```

The API loads these artifacts and performs inference on incoming applicants.

This separates:

```text
Training
```

from:

```text
Production Inference
```

and ensures that the deployed system uses the approved champion model.

---

## 22. Reproducibility

Reproducibility is supported through:

* Fixed random seed
* Persisted preprocessing pipeline
* Persisted trained model
* Persisted probability calibrator
* Centralized configuration
* Automated tests
* Version-controlled source code

The primary random seed is:

```text
42
```

---

## 23. Methodology Summary

The methodology can be summarized as:

```text
Home Credit Data
       ↓
Data Aggregation
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
Probability of Default
       ↓
Credit Score
       ↓
Risk Grade
       ↓
Decision
       ↓
SHAP Explanation
       ↓
API Response
       ↓
Monitoring
```

This methodology provides a complete machine-learning credit-risk workflow from raw financial data to an explainable lending decision.
