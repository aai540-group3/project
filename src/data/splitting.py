import os

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_paths = cfg.data.path
    split_params = cfg.training.split

    interim_data_path = to_absolute_path(f"{data_paths.interim}/data.csv")
    train_data_path = to_absolute_path(f"{data_paths.processed}/train.csv")
    test_data_path = to_absolute_path(f"{data_paths.processed}/test.csv")

    print("Loading data...")
    df = pd.read_csv(interim_data_path)

    print("Splitting data...")
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_params.test_size, random_state=split_params.random_state
    )

    # Save train and test sets
    os.makedirs(to_absolute_path(data_paths.processed), exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)

    print(f"Train data saved to {train_data_path}")
    print(f"Test data saved to {test_data_path}")


if __name__ == "__main__":
    main()
