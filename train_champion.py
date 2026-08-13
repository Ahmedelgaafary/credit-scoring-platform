from src.ml.champion import ChampionModelTrainer


def main():

    trainer = ChampionModelTrainer(
        model_dir="models"
    )

    trainer.train(
        "data/processed/credit_scoring_features.parquet"
    )


if __name__ == "__main__":
    main()