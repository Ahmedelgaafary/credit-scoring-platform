# Credit Scoring Platform — API Documentation

## 1. Overview

The Credit Scoring Platform exposes a REST API for evaluating credit applicants.

The API receives an applicant's engineered features and returns a complete credit-risk assessment.

The assessment includes:

* Probability of Default
* Credit Score
* Risk Grade
* Credit Decision
* Decision Reason
* SHAP Explanation

The API implementation is located at:

```text
app/api.py
```

---

## 2. Base URL

When running locally:

```text
http://127.0.0.1:8000
```

The primary scoring endpoint is:

```text
POST /api/v1/score
```

---

## 3. API Architecture

The request flows through the following components:

```text
Client
  ↓
FastAPI
  ↓
CreditScoringService
  ↓
Preprocessor
  ↓
Champion XGBoost
  ↓
Probability Calibration
  ↓
Credit Score
  ↓
Risk Grade
  ↓
Decision Engine
  ↓
SHAP Explainer
  ↓
JSON Response
```

---

## 4. Health Endpoint

The API provides a health endpoint for checking whether the application is running.

### Request

```text
GET /
```

### Example

```bash
curl http://127.0.0.1:8000/
```

The endpoint can be used as a basic service availability check.

---

## 5. Credit Scoring Endpoint

### Endpoint

```text
POST /api/v1/score
```

### Purpose

Scores a single applicant using the production credit-risk pipeline.

---

## 6. Request Format

The request body contains applicant features.

### Example

```json
{
  "features": {
    "SK_ID_CURR": 100002,
    "AMT_INCOME_TOTAL": 202500,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "AMT_GOODS_PRICE": 351000,
    "EXT_SOURCE_1": 0.083,
    "EXT_SOURCE_2": 0.262949,
    "EXT_SOURCE_3": 0.139376
  }
}
```

The actual production request should contain the complete feature set expected by the persisted preprocessing and model pipeline.

---

## 7. Request Fields

### `features`

Type:

```text
object
```

Contains the applicant's engineered features.

The feature names must correspond to the feature columns expected by the production preprocessing pipeline.

### `SK_ID_CURR`

Type:

```text
integer
```

Unique applicant identifier.

### Numerical Features

Numerical financial and behavioral features should be supplied as numeric values.

### Missing Values

Missing values may be represented as:

```json
null
```

when the corresponding feature supports missing values.

---

## 8. Example Python Request

```python
import requests


url = "http://127.0.0.1:8000/api/v1/score"


payload = {
    "features": {
        "SK_ID_CURR": 100002,
        "AMT_INCOME_TOTAL": 202500,
        "AMT_CREDIT": 406597.5,
        "AMT_ANNUITY": 24700.5,
        "AMT_GOODS_PRICE": 351000
    }
}


response = requests.post(
    url,
    json=payload,
    timeout=60,
)


print(response.status_code)
print(response.json())
```

---

## 9. Successful Response

A successful request returns HTTP status:

```text
200 OK
```

The response contains an assessment object.

### Example

```json
{
  "assessment": {
    "probability_of_default": 0.2548,
    "credit_score": 618,
    "risk_grade": "D",
    "decision": "DECLINE",
    "reason": "Applicant exceeds the maximum acceptable credit-risk level.",
    "explanation": [
      {
        "feature": "EXT_SOURCE_2",
        "shap_value": 0.4215,
        "direction": "increases_risk"
      },
      {
        "feature": "AMT_INCOME_TOTAL",
        "shap_value": -0.1832,
        "direction": "decreases_risk"
      }
    ]
  }
}
```

---

## 10. Response Fields

### `assessment`

Contains the complete credit assessment.

### `probability_of_default`

The calibrated probability that the applicant will default.

Example:

```text
0.2548
```

represents:

```text
25.48%
```

### `credit_score`

The applicant's credit score on the platform's:

```text
300–850
```

scale.

### `risk_grade`

Categorical risk classification.

Possible values:

```text
A
B
C
D
E
```

### `decision`

The lending recommendation.

Possible decisions are:

```text
APPROVE
REVIEW
DECLINE
```

### `reason`

Human-readable explanation of the decision.

### `explanation`

List of SHAP-based factors influencing the prediction.

---

## 11. SHAP Explanation

Each explanation item contains:

```json
{
  "feature": "EXT_SOURCE_2",
  "shap_value": 0.4215,
  "direction": "increases_risk"
}
```

### `feature`

The feature influencing the prediction.

### `shap_value`

The SHAP contribution of the feature.

### `direction`

Indicates whether the feature contribution:

```text
increases_risk
```

or:

```text
decreases_risk
```

---

## 12. Risk Interpretation

The probability of default is converted into the credit score and risk grade.

Current score ranges:

| Grade |   Score |
| ----- | ------: |
| A     | 750–850 |
| B     | 700–749 |
| C     | 650–699 |
| D     | 600–649 |
| E     | 300–599 |

---

## 13. Decision Rules

The current decision thresholds are:

|          PD | Decision |
| ----------: | -------- |
|      `< 5%` | APPROVE  |
| `5% – <20%` | REVIEW   |
|     `≥ 20%` | DECLINE  |

The decision engine also produces a reason describing the resulting decision.

---

## 14. Error Handling

The API should return an error response when the request cannot be processed.

Typical causes include:

* Invalid request structure
* Missing required features
* Invalid feature values
* Model loading failure
* Prediction failure
* Internal processing errors

A failed request should not be interpreted as a credit decision.

---

## 15. HTTP Status Codes

| Status | Meaning                    |
| ------ | -------------------------- |
| `200`  | Successful scoring request |
| `400`  | Invalid request            |
| `422`  | Validation error           |
| `500`  | Internal server error      |

---

## 16. Model Artifacts

The API uses persisted model artifacts from:

```text
models/
```

The main artifacts are:

```text
models/champion_xgboost.joblib
models/champion_preprocessor.joblib
models/probability_calibrator.joblib
```

The API does not retrain the model during a scoring request.

---

## 17. Production Scoring Flow

A scoring request follows:

```text
POST /api/v1/score
        ↓
Parse JSON
        ↓
Validate features
        ↓
Create applicant DataFrame
        ↓
Apply preprocessing
        ↓
XGBoost prediction
        ↓
Probability calibration
        ↓
Credit score calculation
        ↓
Risk grade calculation
        ↓
Decision engine
        ↓
SHAP explanation
        ↓
JSON response
```

---

## 18. API Integration Test

The API integration test is located at:

```text
tests/test_api.py
```

The test:

1. Loads a real applicant from the processed dataset.
2. Removes the training target.
3. Converts the applicant features into JSON-compatible values.
4. Sends the applicant to the API.
5. Verifies that the API returns HTTP 200.
6. Extracts the credit assessment.
7. Displays the credit-risk results.
8. Displays SHAP explanations.
9. Separates risk-increasing and protective factors.

The integration test currently passes successfully.

---

## 19. Running the API

From the project root:

```bash
uvicorn app.api:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

---

## 20. Interactive API Documentation

When the FastAPI application is running, interactive API documentation is available through the automatically generated documentation interfaces.

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

These interfaces allow the scoring endpoint to be tested without writing a separate client.

---

## 21. Example End-to-End Response

A complete assessment can look like:

```json
{
  "assessment": {
    "probability_of_default": 0.031,
    "credit_score": 687,
    "risk_grade": "C",
    "decision": "APPROVE",
    "reason": "Applicant is within the acceptable credit-risk threshold.",
    "explanation": [
      {
        "feature": "EXT_SOURCE_2",
        "shap_value": -0.31,
        "direction": "decreases_risk"
      },
      {
        "feature": "AMT_CREDIT",
        "shap_value": 0.18,
        "direction": "increases_risk"
      }
    ]
  }
}
```

---

## 22. Security Considerations

The API should be deployed behind appropriate production security controls.

Recommended production controls include:

* HTTPS
* Authentication
* Authorization
* Rate limiting
* Request validation
* Logging
* Monitoring
* Secure model artifact storage

Sensitive applicant information should not be unnecessarily written to application logs.

---

## 23. Deployment

The project includes a:

```text
Dockerfile
```

for containerized deployment.

The intended deployment architecture is:

```text
Client
   ↓
API Gateway / Load Balancer
   ↓
Credit Scoring API
   ↓
Credit Scoring Service
   ↓
Model Artifacts
```

---

## 24. API Testing

The API is tested independently from the internal scoring components.

The test suite includes:

```text
tests/test_api.py
```

and the broader test suite covers:

```text
Credit Score
Risk Grade
Decision Engine
Credit Scoring Service
Monitoring
SHAP Explainability
API Integration
```

The complete current test suite has passed successfully.

---

## 25. Summary

The Credit Scoring API provides a single production-facing endpoint:

```text
POST /api/v1/score
```

It transforms applicant features into an explainable credit assessment:

```text
Applicant Features
        ↓
Machine Learning Model
        ↓
Calibrated PD
        ↓
Credit Score
        ↓
Risk Grade
        ↓
Decision
        ↓
SHAP Explanation
```

The API therefore exposes the complete credit-risk workflow through a simple REST interface.
