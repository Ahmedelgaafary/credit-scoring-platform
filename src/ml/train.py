from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src.ml.preprocessing import CreditPreprocessor
from src.ml.split import split_data


class CreditRiskTrainer:
    """Train and evaluate the credit-risk model."""

    def __init__(
        self,
        model,
        model_dir: str | Path = "models",
    ):
        self.model = model
        self.model_dir = Path(model_dir)
        self.preprocessor = CreditPreprocessor()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------

    @staticmethod
    def load_dataset(
        path: str | Path,
    ) -> pd.DataFrame:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
    ):

        (
            X_train,
            X_valid,
            y_train,
            y_valid,
        ) = split_data(df)

        print(
            f"Training samples: {len(X_train):,}"
        )

        print(
            f"Validation samples: {len(X_valid):,}"
        )

        print(
            "\nTraining target distribution:"
        )

        print(
            y_train.value_counts(
                normalize=True
            )
        )

        print(
            "\nValidation target distribution:"
        )

        print(
            y_valid.value_counts(
                normalize=True
            )
        )

        return (
            X_train,
            X_valid,
            y_train,
            y_valid,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
    ):

        print(
            "\nFitting preprocessor..."
        )

        X_train_processed = (
            self.preprocessor.fit_transform(
                X_train
            )
        )

        print(
            "Transforming validation data..."
        )

        X_valid_processed = (
            self.preprocessor.transform(
                X_valid
            )
        )

        print(
            "Processed training shape:",
            X_train_processed.shape,
        )

        print(
            "Processed validation shape:",
            X_valid_processed.shape,
        )

        return (
            X_train_processed,
            X_valid_processed,
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        X_train,
        y_train,
    ):

        print(
            "\nTraining credit-risk model..."
        )

        self.model.fit(
            X_train,
            y_train,
        )

        print(
            "Model training complete."
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_valid,
        y_valid,
    ) -> dict:

        probabilities = (
            self.model.predict_proba(
                X_valid
            )[:, 1]
        )

        roc_auc = roc_auc_score(
            y_valid,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_valid,
            probabilities,
        )

        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

        print("\nModel evaluation")
        print("=" * 50)

        print(
            f"ROC-AUC : {roc_auc:.4f}"
        )

        print(
            f"PR-AUC  : {pr_auc:.4f}"
        )

        return metrics

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------

    def save(
        self,
    ) -> None:

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            self.model_dir
            / "credit_risk_model.joblib"
        )

        preprocessor_path = (
            self.model_dir
            / "credit_preprocessor.joblib"
        )

        joblib.dump(
            self.model,
            model_path,
        )

        self.preprocessor.save(
            preprocessor_path
        )

        print(
            f"\nModel saved to: {model_path}"
        )

        print(
            f"Preprocessor saved to: "
            f"{preprocessor_path}"
        )

    # ------------------------------------------------------------------
    # Complete training pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_path: str | Path,
    ) -> dict:

        print("=" * 70)
        print("CREDIT RISK MODEL TRAINING")
        print("=" * 70)

        df = self.load_dataset(
            dataset_path
        )

        print(
            f"\nDataset shape: {df.shape}"
        )

        (
            X_train,
            X_valid,
            y_train,
            y_valid,
        ) = self.prepare_data(df)

        (
            X_train_processed,
            X_valid_processed,
        ) = self.preprocess(
            X_train,
            X_valid,
        )

        self.train(
            X_train_processed,
            y_train,
        )

        metrics = self.evaluate(
            X_valid_processed,
            y_valid,
        )

        self.save()

        return metrics