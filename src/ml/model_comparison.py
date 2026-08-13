from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from src.ml.preprocessing import CreditPreprocessor
from src.ml.split import split_data
from src.ml.models import (
    create_lightgbm,
    create_logistic_regression,
    create_random_forest,
    create_xgboost,
)


class ModelComparison:
    """Train and compare multiple credit-risk models."""

    def __init__(
        self,
        model_dir: str | Path = "models",
    ):
        self.model_dir = Path(model_dir)

        self.models = {
            "Logistic Regression":
                create_logistic_regression(),

            "Random Forest":
                create_random_forest(),

            "XGBoost":
                create_xgboost(),

            "LightGBM":
                create_lightgbm(),
        }

        self.results: list[dict] = []

    # ------------------------------------------------------------------
    # KS statistic
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_ks(
        y_true: pd.Series,
        probabilities,
    ) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""

        fpr, tpr, _ = roc_curve(
            y_true,
            probabilities,
        )

        return float(
            max(tpr - fpr)
        )

    # ------------------------------------------------------------------
    # Gini
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_gini(
        roc_auc: float,
    ) -> float:
        """Calculate Gini coefficient."""

        return (
            2 * roc_auc
        ) - 1

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        name: str,
        model,
        X_valid,
        y_valid,
    ) -> dict:

        probabilities = (
            model.predict_proba(
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

        gini = self.calculate_gini(
            roc_auc
        )

        ks = self.calculate_ks(
            y_valid,
            probabilities,
        )

        result = {
            "model": name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "gini": gini,
            "ks": ks,
        }

        return result

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_path: str | Path,
    ) -> pd.DataFrame:

        print("=" * 70)
        print("CREDIT RISK MODEL COMPARISON")
        print("=" * 70)

        # --------------------------------------------------------------
        # Load dataset
        # --------------------------------------------------------------

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
            "\nFitting preprocessing..."
        )

        preprocessor = (
            CreditPreprocessor()
        )

        X_train_processed = (
            preprocessor.fit_transform(
                X_train
            )
        )

        X_valid_processed = (
            preprocessor.transform(
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

        # --------------------------------------------------------------
        # Train models
        # --------------------------------------------------------------

        for name, model in self.models.items():

            print("\n" + "=" * 70)
            print(f"Training: {name}")
            print("=" * 70)

            model.fit(
                X_train_processed,
                y_train,
            )

            result = self.evaluate(
                name,
                model,
                X_valid_processed,
                y_valid,
            )

            self.results.append(
                result
            )

            print(
                f"ROC-AUC: {result['roc_auc']:.4f}"
            )

            print(
                f"PR-AUC : {result['pr_auc']:.4f}"
            )

            print(
                f"Gini   : {result['gini']:.4f}"
            )

            print(
                f"KS     : {result['ks']:.4f}"
            )

        # --------------------------------------------------------------
        # Results
        # --------------------------------------------------------------

        results_df = pd.DataFrame(
            self.results
        )

        results_df = results_df.sort_values(
            by="roc_auc",
            ascending=False,
        ).reset_index(
            drop=True
        )

        print("\n")
        print("=" * 70)
        print("MODEL COMPARISON RESULTS")
        print("=" * 70)

        print(
            results_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        # --------------------------------------------------------------
        # Save comparison
        # --------------------------------------------------------------

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_path = (
            self.model_dir
            / "model_comparison.csv"
        )

        results_df.to_csv(
            results_path,
            index=False,
        )

        print(
            f"\nResults saved to: "
            f"{results_path}"
        )

        return results_df