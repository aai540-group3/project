"""
.. module:: models.autogluon.preprocessing
   :synopsis: Preprocess data for training the AutoGluon model.

This script performs preprocessing steps on the training data for the AutoGluon model, including:

- Cleaning feature names to replace invalid characters.
- Identifying numerical and boolean features.
- Imputing missing values in numerical features using the median.
- Scaling numerical features using StandardScaler.
- Converting boolean features to integers.
- Imputing missing values in boolean features using the most frequent value.

The preprocessed data is then saved to a CSV file.
"""

import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bool_to_int(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert boolean features to integers.

    :param X: Input DataFrame.
    :type X: pd.DataFrame
    :return: DataFrame with boolean features converted to integers.
    :rtype: pd.DataFrame
    """
    return X.astype(int)


def clean_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature names by replacing invalid characters.

    This function replaces square brackets, closing brackets, and less than signs
    with underscores in the column names of the input DataFrame.

    :param X: Input DataFrame.
    :type X: pd.DataFrame
    :return: DataFrame with cleaned feature names.
    :rtype: pd.DataFrame
    """
    X.columns = (
        X.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
    )
    return X


@hydra.main(
        config_path=os.getenv("CONFIG_PATH"),
        config_name=os.getenv("CONFIG_NAME"),
        version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Preprocess data for training the AutoGluon model.

    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :raises Exception: If an error occurs during preprocessing.
    """
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Use Hydra-managed paths
        train_data_path = Path(to_absolute_path(cfg.paths.data.processed.train_file))
        output_dir = Path(to_absolute_path(cfg.paths.data.processed.base)) / cfg.model.name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preprocessing data for {cfg.model.name}...")

        train_df = pd.read_csv(train_data_path)
        logger.info(f"Loaded data shape: {train_df.shape}")
        logger.info(f"Columns: {train_df.columns}")

        y = train_df["readmitted"]
        X = train_df.drop(columns=["readmitted"])

        # Clean feature names
        X = clean_feature_names(X)

        # Define preprocessing steps
        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        boolean_features = X.select_dtypes(include=["bool"]).columns.tolist()

        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Boolean features: {boolean_features}")

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        boolean_transformer = Pipeline(
            steps=[
                ("bool_to_int", FunctionTransformer(bool_to_int)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("bool", boolean_transformer, boolean_features),
            ]
        )

        # Fit and transform the data
        X_preprocessed = preprocessor.fit_transform(X)

        # Get feature names
        feature_names = numeric_features + boolean_features

        # Convert to DataFrame
        X_preprocessed_df = pd.DataFrame(
            X_preprocessed, columns=feature_names
        )

        # Add the target variable back
        X_preprocessed_df["readmitted"] = y

        logger.info(f"Preprocessed data shape: {X_preprocessed_df.shape}")

        # Save preprocessed data
        preprocessed_train_data_path = output_dir / "train_preprocessed.csv"
        X_preprocessed_df.to_csv(preprocessed_train_data_path, index=False)

        logger.info(f"Preprocessed data saved to {preprocessed_train_data_path}")

    except Exception as e:
        logger.error(
            f"An error occurred during preprocessing: {str(e)}", exc_info=True
        )
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


if __name__ == "__main__":
    main()
