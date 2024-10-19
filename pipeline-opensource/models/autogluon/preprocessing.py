"""
.. module:: models.autogluon.preprocessing
   :synopsis: Preprocess data for training and evaluating the AutoGluon model.

This script performs preprocessing steps on both the training and testing data for the AutoGluon model, including:

- Cleaning feature names to replace invalid characters.
- Identifying numerical and boolean features.
- Imputing missing values in numerical features using the median.
- Scaling numerical features using StandardScaler.
- Converting boolean features to integers.
- Imputing missing values in boolean features using the most frequent value.

The preprocessed data is then saved to CSV files in the model's artifacts directory.
"""

import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import joblib
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
    Preprocess data for training and evaluating the AutoGluon model.

    :param cfg: Hydra configuration object.
    :type cfg: DictConfig
    :raises Exception: If an error occurs during preprocessing.
    """
    try:
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Use Hydra-managed paths
        train_data_path = Path(to_absolute_path(cfg.paths.data.processed.train_file))
        test_data_path = Path(to_absolute_path(cfg.paths.data.processed.test_file))
        artifacts_dir = Path(to_absolute_path(cfg.paths.models.autogluon.artifacts))
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Preprocessing data for {cfg.model.name}...")

        # Load training data
        train_df = pd.read_csv(train_data_path)
        logger.info(f"Loaded training data shape: {train_df.shape}")
        logger.info(f"Training data columns: {train_df.columns.tolist()}")

        # Load test data
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded test data shape: {test_df.shape}")
        logger.info(f"Test data columns: {test_df.columns.tolist()}")

        # Combine training and test data for consistent preprocessing
        combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

        # Split features and target
        y_combined = combined_df["readmitted"]
        X_combined = combined_df.drop(columns=["readmitted"])

        # Clean feature names
        X_combined = clean_feature_names(X_combined)

        # Define preprocessing steps
        numeric_features = X_combined.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        boolean_features = X_combined.select_dtypes(include=["bool"]).columns.tolist()

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
            ],
            remainder="passthrough",  # Include any remaining columns as is
        )

        # Fit and transform the data
        X_preprocessed_combined = preprocessor.fit_transform(X_combined)

        # Get all feature names after preprocessing
        processed_feature_names = (
            numeric_features
            + boolean_features
            + [
                col
                for col in X_combined.columns
                if col not in numeric_features + boolean_features
            ]
        )

        # Convert to DataFrame
        X_preprocessed_combined_df = pd.DataFrame(
            X_preprocessed_combined, columns=processed_feature_names
        )

        # Add the target variable back
        X_preprocessed_combined_df["readmitted"] = y_combined

        logger.info(
            f"Preprocessed combined data shape: {X_preprocessed_combined_df.shape}"
        )

        # Split back into training and test sets
        train_rows = len(train_df)
        X_preprocessed_train_df = X_preprocessed_combined_df.iloc[
            :train_rows
        ].reset_index(drop=True)
        X_preprocessed_test_df = X_preprocessed_combined_df.iloc[
            train_rows:
        ].reset_index(drop=True)

        # Save preprocessed training data
        preprocessed_train_data_path = Path(
            to_absolute_path(cfg.paths.models.autogluon.preprocessed_data_train)
        )
        X_preprocessed_train_df.to_csv(preprocessed_train_data_path, index=False)
        logger.info(
            f"Preprocessed training data saved to {preprocessed_train_data_path}"
        )

        # Save preprocessed test data
        preprocessed_test_data_path = Path(
            to_absolute_path(cfg.paths.models.autogluon.preprocessed_data_test)
        )
        X_preprocessed_test_df.to_csv(preprocessed_test_data_path, index=False)
        logger.info(f"Preprocessed test data saved to {preprocessed_test_data_path}")

        # Save the preprocessor
        preprocessor_path = Path(
            to_absolute_path(cfg.paths.models.autogluon.preprocessor)
        )
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {str(e)}", exc_info=True)
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


if __name__ == "__main__":
    main()
