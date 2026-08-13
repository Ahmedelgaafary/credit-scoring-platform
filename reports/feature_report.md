# Feature Engineering Report

## 1. Overview

This report describes the feature-engineering pipeline used by the Credit Scoring Platform.

The objective is to transform the raw Home Credit datasets into an applicant-level feature matrix suitable for machine-learning credit-risk prediction.

The feature-engineering pipeline combines information from:

* Application data
* Bureau records
* Bureau balance history
* Previous applications
* Installment payments
* Credit card balances
* POS cash balances

The final output is stored as:

```text id="y8k3vd"
data/processed/credit_scoring_features.parquet
```

---

## 2. Feature Engineering Architecture

The feature pipeline follows:

```text id="s9yq2k"
Raw Datasets
     ↓
Dataset-Specific Aggregation
     ↓
Applicant-Level Features
     ↓
Feature Joining
     ↓
Cleaning / Validation
     ↓
Final Feature Matrix
     ↓
Parquet Dataset
```

The implementation is located in:

```text id="3qzqmx"
src/features/
```

---

## 3. Source Datasets

### Application

```text id="v8z8u7"
application_train.csv
application_test.csv
```

Contains the main applicant-level information.

Examples include:

* Applicant identifier
* Target variable
* Income
* Credit amount
* Annuity
* Employment information
* Demographic and household information
* Housing information

The applicant identifier is:

```text id="b5f0pu"
SK_ID_CURR
```

---

### Bureau

```text id="7j4e1a"
bureau.csv
```

Contains historical credit information obtained from external credit records.

The bureau data is aggregated at the applicant level.

Implementation:

```text id="7v8n2h"
src/features/bureau_features.py
```

---

### Bureau Balance

```text id="p1ck6d"
bureau_balance.csv
```

Contains historical monthly status information for bureau credit accounts.

The data is aggregated before being joined with applicant-level bureau features.

Implementation:

```text id="1h7m0c"
src/features/bureau_balance_features.py
```

---

### Previous Applications

```text id="a4q6n0"
previous_application.csv
```

Contains applicants' previous credit applications.

Features are aggregated to capture historical application behavior.

Implementation:

```text id="9y5b2x"
src/features/previous_application_features.py
```

---

### Installments

```text id="2nq8rw"
installments_payments.csv
```

Contains historical installment-payment behavior.

The feature engineering process captures repayment-related characteristics from this dataset.

Implementation:

```text id="6t8v0s"
src/features/installment_features.py
```

---

### Credit Card Balance

```text id="q4x9ka"
credit_card_balance.csv
```

Contains historical credit-card account information.

Features are aggregated at the applicant level.

Implementation:

```text id="5h4p7v"
src/features/credit_card_features.py
```

---

### POS Cash Balance

```text id="d2x6vn"
POS_CASH_balance.csv
```

Contains historical POS/cash-loan information.

Features are aggregated before integration with the main applicant dataset.

Implementation:

```text id="3k7m1r"
src/features/pos_cash_features.py
```

---

## 4. Feature Modules

The feature-engineering package contains:

```text id="c6d8r1"
src/features/
├── application_features.py
├── bureau_balance_features.py
├── bureau_features.py
├── credit_card_features.py
├── feature_pipeline.py
├── installment_features.py
├── pos_cash_features.py
└── previous_application_features.py
```

Each module is responsible for transforming one logical source of information.

---

## 5. Application Features

Application features provide the primary applicant-level information.

The application feature module is:

```text id="z1k4qc"
src/features/application_features.py
```

Typical feature groups include:

### Financial Features

* Income
* Credit amount
* Annuity
* Goods price
* Income-to-credit relationships
* Credit-to-income relationships

### Employment Features

* Employment duration
* Registration duration
* Age-related variables

### Household Features

* Family size
* Number of children
* Household characteristics

### Housing Features

* Property characteristics
* Housing type
* Ownership information

These features provide the baseline applicant profile.

---

## 6. Bureau Features

Bureau information provides historical external credit information.

The bureau module aggregates multiple records for the same applicant.

Typical aggregation categories include:

```text id="0n6a6x"
Count of credit accounts
Credit amounts
Outstanding balances
Debt-related amounts
Account status
Historical credit activity
```

Aggregating bureau records prevents multiple rows per applicant from entering the final modeling dataset.

---

## 7. Bureau Balance Features

Bureau balance records contain monthly observations.

These records are transformed into applicant-level summaries.

The objective is to capture:

* Historical account status
* Number of monthly observations
* Account-history characteristics
* Credit-account activity

The resulting features are joined through the bureau account relationship.

---

## 8. Previous Application Features

Previous applications provide historical information about how applicants interacted with credit products.

The feature-engineering process summarizes:

* Number of previous applications
* Previous application amounts
* Previous approved applications
* Previous refused applications
* Historical credit activity

These features provide information about previous borrowing behavior.

---

## 9. Installment Features

Installment-payment data provides information about repayment behavior.

The feature engineering process captures characteristics such as:

```text id="4n6y7p"
Payment behavior
Installment amounts
Payment amounts
Payment differences
Historical repayment patterns
```

These features are useful because repayment behavior provides information about historical credit performance.

---

## 10. Credit Card Features

Credit-card history is transformed into applicant-level aggregates.

The resulting features describe historical credit-card behavior, including:

```text id="7d2p1v"
Credit utilization
Balance behavior
Payment behavior
Account activity
Historical balances
```

These features provide additional information about the applicant's existing credit exposure.

---

## 11. POS Cash Features

POS cash-loan history is aggregated to applicant level.

The features describe:

* Historical POS activity
* Outstanding balances
* Contract characteristics
* Payment history
* Account status

These features complement bureau and installment information.

---

## 12. Aggregation Strategy

Most historical datasets contain multiple records for one applicant.

The platform therefore uses aggregation to convert:

```text id="q3n8fa"
Multiple historical records
            ↓
One applicant-level feature vector
```

Common aggregation operations include:

```text id="2j8x0q"
count
mean
sum
min
max
```

Depending on the source dataset and feature definition.

---

## 13. Applicant-Level Join

After each dataset is transformed independently, the resulting feature tables are joined using the applicant identifier:

```text id="0q2m4k"
SK_ID_CURR
```

The final structure follows:

```text id="8p7v2n"
Application Features
        +
Bureau Features
        +
Bureau Balance Features
        +
Previous Application Features
        +
Installment Features
        +
Credit Card Features
        +
POS Cash Features
        ↓
Final Applicant Feature Matrix
```

---

## 14. Feature Pipeline

The central orchestration module is:

```text id="6k9c2m"
src/features/feature_pipeline.py
```

Its responsibility is to coordinate the individual feature modules and produce the final modeling dataset.

The pipeline ensures that feature generation remains centralized rather than being duplicated across training and inference workflows.

---

## 15. Feature Quality

Feature engineering must preserve the integrity of the applicant-level dataset.

Important checks include:

* Applicant identifier availability
* Duplicate applicant detection
* Missing-value handling
* Numeric feature consistency
* Correct joins between datasets
* Target separation
* Consistent feature names

The resulting dataset must contain one logical feature vector per applicant.

---

## 16. Target Handling

The target variable is:

```text id="z4w8s6"
TARGET
```

The target is used during model training but should not be included as an input feature during prediction.

Therefore:

```text id="h2f4qb"
Training:
Features + TARGET

Inference:
Features only
```

This separation prevents target leakage during prediction.

---

## 17. Identifier Handling

The applicant identifier is:

```text id="7g1q6a"
SK_ID_CURR
```

It is primarily used to identify and join applicant records.

Identifiers should not be treated as ordinary predictive variables unless there is a justified modeling reason.

---

## 18. Output Dataset

The processed dataset is stored in Parquet format:

```text id="n7v4cx"
data/processed/credit_scoring_features.parquet
```

Parquet provides an efficient format for storing the engineered feature matrix.

It is also convenient for loading the data during model training and testing.

---

## 19. Feature Reproducibility

Feature engineering is implemented as reusable Python modules rather than manually generated notebook transformations.

This allows the same feature definitions to be reused during:

```text id="8c3j4k"
Training
Validation
Testing
Inference
```

This is important for preventing discrepancies between training and production features.

---

## 20. Feature Leakage Prevention

Feature engineering must ensure that information unavailable at the time of credit assessment does not enter the model.

The platform separates:

```text id="6x9v3n"
Input Features
        ↓
Model
        ↓
Prediction
```

from:

```text id="4j7p1c"
TARGET
```

which is used only during supervised training and evaluation.

Historical datasets are aggregated to applicant level before being used for modeling.

---

## 21. Feature Processing Flow

The complete feature process is:

```text id="q8s5mn"
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
credit_card_balance.csv
POS_CASH_balance.csv
              ↓
       Dataset Processing
              ↓
       Feature Aggregation
              ↓
       Applicant-Level Joins
              ↓
       Feature Validation
              ↓
credit_scoring_features.parquet
```

---

## 22. Relationship With ML Pipeline

The feature-engineering pipeline feeds the machine-learning pipeline:

```text id="c8x1mz"
Feature Engineering
        ↓
Processed Feature Matrix
        ↓
ML Preprocessing
        ↓
Train / Validation Split
        ↓
Candidate Models
        ↓
Model Comparison
        ↓
Champion XGBoost
```

The ML components are implemented under:

```text id="t2r6hv"
src/ml/
```

---

## 23. Relationship With Risk Pipeline

After model prediction, the engineered features indirectly contribute to:

```text id="x3k9vb"
Probability of Default
        ↓
Credit Score
        ↓
Risk Grade
        ↓
Credit Decision
        ↓
SHAP Explanation
```

Therefore, feature engineering is the foundation of the complete credit-risk pipeline.

---

## 24. Summary

The feature-engineering layer converts multiple raw credit datasets into a unified applicant-level representation.

The main design principles are:

* Dataset-specific feature modules
* Applicant-level aggregation
* Consistent joining through `SK_ID_CURR`
* Separation of target and input features
* Reusable feature pipelines
* Leakage prevention
* Reproducibility
* Efficient Parquet storage

The resulting feature matrix provides the foundation for the platform's machine-learning and credit-risk decisioning components.
