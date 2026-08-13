from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from src.ml.models import create_xgboost
from src.ml.preprocessing import CreditPreprocessor
from src.ml.split import split_data


class ChampionModelTrainer:
    """Train and persist the selected champion credit-risk model."""

    def __init__(
        self,
        model_dir: str | Path = "models",
    ):
        self.model_dir = Path(model_dir)

        self.model = create_xgboost()

        self.preprocessor = CreditPreprocessor()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_ks(
        y_true: pd.Series,
        probabilities,
    ) -> float:

        fpr, tpr, _ = roc_curve(
            y_true,
            probabilities,
        )

        return float(
            (tpr - fpr).max()
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_path: str | Path,
    ) -> dict:

        print("=" * 70)
        print("CHAMPION CREDIT-RISK MODEL")
        print("=" * 70)

        dataset_path = Path(
            dataset_path
        )

        print(
            f"\nLoading dataset: {dataset_path}"
        )

        df = pd.read_parquet(
            dataset_path
        )

        print(
            f"Dataset shape: {df.shape}"
        )

        # --------------------------------------------------------------
        # Split
        # --------------------------------------------------------------

        (
            X_train,
            X_valid,
            y_train,
            y_valid,
        ) = split_data(df)

        print(
            f"\nTraining samples: "
            f"{len(X_train):,}"
        )

        print(
            f"Validation samples: "
            f"{len(X_valid):,}"
        )

        # --------------------------------------------------------------
        # Preprocessing
        # --------------------------------------------------------------

        print(
            "\nFitting preprocessor..."
        )

        X_train_processed = (
            self.preprocessor.fit_transform(
                X_train
            )
        )

        X_valid_processed = (
            self.preprocessor.transform(
                X_valid
            )
        )

        print(
            "Training matrix:",
            X_train_processed.shape,
        )

        print(
            "Validation matrix:",
            X_valid_processed.shape,
        )

        # --------------------------------------------------------------
        # Model
        # --------------------------------------------------------------

        print(
            "\nTraining XGBoost champion..."
        )

        self.model.fit(
            X_train_processed,
            y_train,
        )

        print(
            "Training complete."
        )

        # --------------------------------------------------------------
        # Validation predictions
        # --------------------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                X_valid_processed
            )[:, 1]
        )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        roc_auc = roc_auc_score(
            y_valid,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_valid,
            probabilities,
        )

        gini = (
            2 * roc_auc
        ) - 1

        ks = self.calculate_ks(
            y_valid,
            probabilities,
        )

        metrics = {
            "model": "XGBoost",
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "gini": float(gini),
            "ks": float(ks),
            "training_samples": int(
                len(X_train)
            ),
            "validation_samples": int(
                len(X_valid)
            ),
            "features": int(
                X_train_processed.shape[1]
            ),
        }

        print("\nChampion validation results")
        print("=" * 50)

        print(
            f"ROC-AUC : {roc_auc:.4f}"
        )

        print(
            f"PR-AUC  : {pr_auc:.4f}"
        )

        print(
            f"Gini    : {gini:.4f}"
        )

        print(
            f"KS      : {ks:.4f}"
        )

        # --------------------------------------------------------------
        # Save
        # --------------------------------------------------------------

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            self.model_dir
            / "champion_xgboost.joblib"
        )

        preprocessor_path = (
            self.model_dir
            / "champion_preprocessor.joblib"
        )

        predictions_path = (
            self.model_dir
            / "validation_predictions.csv"
        )

        metrics_path = (
            self.model_dir
            / "champion_metrics.json"
        )

        # Model
        joblib.dump(
            self.model,
            model_path,
        )

        # Preprocessor
        self.preprocessor.save(
            preprocessor_path
        )

        # Validation predictions
        predictions = pd.DataFrame(
            {
                "y_true": y_valid.values,
                "probability_of_default":
                    probabilities,
            }
        )

        predictions.to_csv(
            predictions_path,
            index=False,
        )

        # Metrics
        with open(
            metrics_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
            )

        print("\nSaved artifacts:")
        print(
            f"Model:         {model_path}"
        )
        print(
            f"Preprocessor:  {preprocessor_path}"
        )
        print(
            f"Predictions:   {predictions_path}"
        )
        print(
            f"Metrics:       {metrics_path}"
        )

        return metrics