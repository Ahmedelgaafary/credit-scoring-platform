from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from src.data.loader import DataLoader

from src.features.application_features import (
    ApplicationFeatureEngineer,
)

from src.features.bureau_balance_features import (
    BureauBalanceFeatureEngineer,
)

from src.features.bureau_features import (
    BureauFeatureEngineer,
)

from src.features.previous_application_features import (
    PreviousApplicationFeatureEngineer,
)

from src.features.installment_features import (
    InstallmentFeatureEngineer,
)

from src.features.credit_card_features import (
    CreditCardFeatureEngineer,
)

from src.features.pos_cash_features import (
    POSCashFeatureEngineer,
)


class FeaturePipeline:
    """Build the complete applicant-level feature dataset."""

    def __init__(
        self,
        data_dir: str | Path,
    ):
        self.loader = DataLoader(data_dir)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        base: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge applicant-level features."""

        return base.merge(
            features,
            on="SK_ID_CURR",
            how="left",
        )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def _build_application(
        self,
    ) -> pd.DataFrame:

        application = (
            self.loader.load_application_train()
        )

        target = application[
            [
                "SK_ID_CURR",
                "TARGET",
            ]
        ].copy()

        features = (
            ApplicationFeatureEngineer
            .create_features(application)
        )

        result = features.merge(
            target,
            on="SK_ID_CURR",
            how="left",
        )

        del application
        del features
        del target

        gc.collect()

        return result

    # ------------------------------------------------------------------
    # Bureau + Bureau Balance
    # ------------------------------------------------------------------

    def _add_bureau_features(
        self,
        base: pd.DataFrame,
    ) -> pd.DataFrame:

        print("Loading bureau_balance...")

        bureau_balance = (
            self.loader.load_bureau_balance(
                usecols=[
                    "SK_ID_BUREAU",
                    "MONTHS_BALANCE",
                    "STATUS",
                ]
            )
        )

        print(
            "Aggregating bureau_balance..."
        )

        bureau_balance_features = (
            BureauBalanceFeatureEngineer
            .create_features(
                bureau_balance
            )
        )

        del bureau_balance
        gc.collect()

        print("Loading bureau...")

        bureau = self.loader.load_bureau()

        print("Creating bureau features...")

        bureau_features = (
            BureauFeatureEngineer
            .create_features(
                bureau,
                bureau_balance_features,
            )
        )

        del bureau
        del bureau_balance_features

        gc.collect()

        print("Merging bureau features...")

        return self._merge(
            base,
            bureau_features,
        )

    # ------------------------------------------------------------------
    # Previous Applications
    # ------------------------------------------------------------------

    def _add_previous_application_features(
        self,
        base: pd.DataFrame,
    ) -> pd.DataFrame:

        print(
            "Loading previous applications..."
        )

        previous_application = (
            self.loader.load_previous_application()
        )

        print(
            "Creating previous application features..."
        )

        features = (
            PreviousApplicationFeatureEngineer
            .create_features(
                previous_application
            )
        )

        del previous_application
        gc.collect()

        return self._merge(
            base,
            features,
        )

    # ------------------------------------------------------------------
    # Installments
    # ------------------------------------------------------------------

    def _add_installment_features(
        self,
        base: pd.DataFrame,
    ) -> pd.DataFrame:

        print("Loading installments...")

        installments = (
            self.loader.load_installments()
        )

        print(
            "Creating installment features..."
        )

        features = (
            InstallmentFeatureEngineer
            .create_features(
                installments
            )
        )

        del installments
        gc.collect()

        return self._merge(
            base,
            features,
        )

    # ------------------------------------------------------------------
    # Credit Card
    # ------------------------------------------------------------------

    def _add_credit_card_features(
        self,
        base: pd.DataFrame,
    ) -> pd.DataFrame:

        print("Loading credit card...")

        credit_card = (
            self.loader.load_credit_card()
        )

        print(
            "Creating credit-card features..."
        )

        features = (
            CreditCardFeatureEngineer
            .create_features(
                credit_card
            )
        )

        del credit_card
        gc.collect()

        return self._merge(
            base,
            features,
        )

    # ------------------------------------------------------------------
    # POS/CASH
    # ------------------------------------------------------------------

    def _add_pos_cash_features(
        self,
        base: pd.DataFrame,
    ) -> pd.DataFrame:

        print("Loading POS/CASH...")

        pos_cash = (
            self.loader.load_pos_cash()
        )

        print(
            "Creating POS/CASH features..."
        )

        features = (
            POSCashFeatureEngineer
            .create_features(
                pos_cash
            )
        )

        del pos_cash
        gc.collect()

        return self._merge(
            base,
            features,
        )

    # ------------------------------------------------------------------
    # Complete pipeline
    # ------------------------------------------------------------------

    def build_training_dataset(
        self,
    ) -> pd.DataFrame:

        print("=" * 70)
        print("BUILDING CREDIT SCORING DATASET")
        print("=" * 70)

        # --------------------------------------------------------------
        # Application
        # --------------------------------------------------------------

        print("\n[1/6] Application features")

        dataset = self._build_application()

        print(
            f"Current shape: {dataset.shape}"
        )

        # --------------------------------------------------------------
        # Bureau
        # --------------------------------------------------------------

        print("\n[2/6] Bureau features")

        dataset = self._add_bureau_features(
            dataset
        )

        print(
            f"Current shape: {dataset.shape}"
        )

        # --------------------------------------------------------------
        # Previous applications
        # --------------------------------------------------------------

        print(
            "\n[3/6] Previous application features"
        )

        dataset = (
            self._add_previous_application_features(
                dataset
            )
        )

        print(
            f"Current shape: {dataset.shape}"
        )

        # --------------------------------------------------------------
        # Installments
        # --------------------------------------------------------------

        print(
            "\n[4/6] Installment features"
        )

        dataset = (
            self._add_installment_features(
                dataset
            )
        )

        print(
            f"Current shape: {dataset.shape}"
        )

        # --------------------------------------------------------------
        # Credit cards
        # --------------------------------------------------------------

        print(
            "\n[5/6] Credit-card features"
        )

        dataset = (
            self._add_credit_card_features(
                dataset
            )
        )

        print(
            f"Current shape: {dataset.shape}"
        )

        # --------------------------------------------------------------
        # POS/CASH
        # --------------------------------------------------------------

        print(
            "\n[6/6] POS/CASH features"
        )

        dataset = (
            self._add_pos_cash_features(
                dataset
            )
        )

        print(
            f"Current shape: {dataset.shape}"
        )

        print("\n" + "=" * 70)
        print("FEATURE BUILD COMPLETE")
        print("=" * 70)

        return dataset

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_training_dataset(
        self,
        output_path: str | Path,
    ) -> pd.DataFrame:

        dataset = (
            self.build_training_dataset()
        )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"\nSaving dataset to: {output_path}"
        )

        dataset.to_parquet(
            output_path,
            index=False,
        )

        print("Dataset saved.")

        return dataset