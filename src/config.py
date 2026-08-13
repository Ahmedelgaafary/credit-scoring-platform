"""
                        config.py

                        Part of Credit Scoring Platform.
                        """
from pathlib import Path


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_NAME = "Credit Scoring Platform"
APP_VERSION = "1.0.0"
DEBUG = False


# ============================================================
# Data Directories
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"


# ============================================================
# Raw Data Files
# ============================================================

APPLICATION_TRAIN_FILE = (
    RAW_DATA_DIR / "application_train.csv"
)

APPLICATION_TEST_FILE = (
    RAW_DATA_DIR / "application_test.csv"
)

BUREAU_FILE = (
    RAW_DATA_DIR / "bureau.csv"
)

BUREAU_BALANCE_FILE = (
    RAW_DATA_DIR / "bureau_balance.csv"
)

PREVIOUS_APPLICATION_FILE = (
    RAW_DATA_DIR / "previous_application.csv"
)

INSTALLMENTS_FILE = (
    RAW_DATA_DIR / "installments_payments.csv"
)

CREDIT_CARD_FILE = (
    RAW_DATA_DIR / "credit_card_balance.csv"
)

POS_CASH_FILE = (
    RAW_DATA_DIR / "POS_CASH_balance.csv"
)


# ============================================================
# Processed Data
# ============================================================

PROCESSED_FEATURES_FILE = (
    PROCESSED_DATA_DIR / "credit_features.parquet"
)

TRAIN_FEATURES_FILE = (
    PROCESSED_DATA_DIR / "train_features.parquet"
)

TEST_FEATURES_FILE = (
    PROCESSED_DATA_DIR / "test_features.parquet"
)


# ============================================================
# Model
# ============================================================

MODEL_DIR = PROJECT_ROOT / "model"

MODEL_FILE = MODEL_DIR / "credit_model.pkl"

FEATURE_COLUMNS_FILE = (
    MODEL_DIR / "feature_columns.json"
)

MODEL_METADATA_FILE = (
    MODEL_DIR / "model_metadata.json"
)


# ============================================================
# Target
# ============================================================

TARGET_COLUMN = "TARGET"

ID_COLUMN = "SK_ID_CURR"


# ============================================================
# Credit Score
# ============================================================

MIN_CREDIT_SCORE = 300
MAX_CREDIT_SCORE = 850


# ============================================================
# Risk Grades
# ============================================================

RISK_GRADES = {
    "A": {
        "min_score": 750,
        "max_score": 850,
    },
    "B": {
        "min_score": 700,
        "max_score": 749,
    },
    "C": {
        "min_score": 650,
        "max_score": 699,
    },
    "D": {
        "min_score": 600,
        "max_score": 649,
    },
    "E": {
        "min_score": 300,
        "max_score": 599,
    },
}


# ============================================================
# Decision Thresholds
# ============================================================

DECISION_THRESHOLDS = {
    "approve": 0.05,
    "review": 0.20,
}


# ============================================================
# Model Training
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.20