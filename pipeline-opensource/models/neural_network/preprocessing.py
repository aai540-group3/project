"""
.. module:: models.neural_network.preprocessing
   :synopsis: Preprocess data for training the neural network model.

This script performs preprocessing steps on the data for the neural network model, including:

- Handling missing values separately for numerical and binary features.
- Converting boolean features to integers.
- Scaling numerical features using StandardScaler.
- Splitting the data into training, validation, and testing sets.
- Saving the preprocessed data and preprocessing objects.
"""

import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bool_to_int(X):
    """
    Converts boolean features to integers (True -> 1, False -> 0).
    """
    return X.astype(int)


@hydra.main(
    config_path=os.getenv("CONFIG_PATH"),
    config_name=os.getenv("CONFIG_NAME"),
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Main function to preprocess data for the neural network model.

    :param cfg: Configuration object provided by Hydra.
    """
    try:
        # Ensure model.name is set to 'neural_network'
        cfg.model.name = "neural_network"

        logger.info("Starting preprocessing for neural network model...")
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Resolve paths using cfg.paths
        train_data_file = Path(to_absolute_path(cfg.paths.data.processed.train_file))
        test_data_file = Path(to_absolute_path(cfg.paths.data.processed.test_file))
        output_dir = Path(to_absolute_path(cfg.paths.models.neural_network.artifacts))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info(f"Loading training data from {train_data_file}...")
        train_df = pd.read_csv(train_data_file)
        logger.info(f"Loading testing data from {test_data_file}...")
        test_df = pd.read_csv(test_data_file)

        # Combine train and test data for consistent preprocessing
        combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

        # Ensure all data is numeric
        logger.info("Converting all data to numeric types, coercing errors to NaN...")
        combined_df = combined_df.apply(pd.to_numeric, errors="coerce")

        # Replace infinite values with NaN
        logger.info("Replacing infinite values with NaN...")
        combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Identify feature types
        logger.info("Identifying numerical and binary features...")
        # Exclude 'readmitted' from features
        features = combined_df.columns.drop("readmitted")

        # Separate numerical and binary features
        numerical_features = (
            combined_df[features]
            .select_dtypes(include=["int64", "float64"])
            .columns.tolist()
        )
        binary_features = (
            combined_df[features].select_dtypes(include=["bool"]).columns.tolist()
        )

        # Convert object type columns (if any) to categorical and then to numerical codes
        object_features = (
            combined_df[features].select_dtypes(include=["object"]).columns.tolist()
        )
        if object_features:
            logger.info(
                f"Converting object type features to categorical codes: {object_features}"
            )
            for col in object_features:
                combined_df[col] = combined_df[col].astype("category").cat.codes
            numerical_features.extend(object_features)

        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Binary features: {binary_features}")

        # Split features and target
        logger.info("Splitting features and target variable...")
        X = combined_df.drop(columns=["readmitted"])
        y = combined_df["readmitted"].values  # Ensure y is a NumPy array

        # Define transformers
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        binary_transformer = Pipeline(
            steps=[
                ("bool_to_int", FunctionTransformer(bool_to_int)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("bin", binary_transformer, binary_features),
            ]
        )

        # Fit the preprocessor and transform the data
        logger.info("Fitting preprocessor and transforming data...")
        X_processed = preprocessor.fit_transform(X)

        # Get feature names
        try:
            feature_names_num = (
                preprocessor.named_transformers_["num"]
                .named_steps["scaler"]
                .get_feature_names_out(numerical_features)
            )
        except AttributeError:
            feature_names_num = numerical_features

        feature_names_bin = binary_features  # Binary features remain unchanged

        feature_names = np.concatenate([feature_names_num, feature_names_bin])

        # Convert processed data back to DataFrame
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

        # Save the preprocessor
        preprocessor_path = Path(
            to_absolute_path(cfg.paths.models.neural_network.preprocessor)
        )
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        # Split back into train and test sets
        X_train_processed = X_processed_df.iloc[: len(train_df), :]
        X_test_processed = X_processed_df.iloc[len(train_df) :, :]
        y_train = y[: len(train_df)]
        y_test = y[len(train_df) :]

        # Further split training data into training and validation sets
        logger.info("Splitting training data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_processed,
            y_train,
            test_size=0.2,
            random_state=cfg.training.split.random_state,
            stratify=y_train,
        )

        # Save processed data as NumPy arrays
        logger.info("Saving processed data as NumPy arrays...")
        np.save(output_dir / "X_train.npy", X_train.values)
        np.save(output_dir / "X_val.npy", X_val.values)
        np.save(output_dir / "X_test.npy", X_test_processed.values)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_val.npy", y_val)
        np.save(output_dir / "y_test.npy", y_test)
        logger.info("Processed data saved as NumPy arrays.")

        # Save the combined preprocessed data for DVC pipeline
        logger.info("Saving combined preprocessed data to preprocessed_data.csv...")
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = np.concatenate([y_train, y_val])

        preprocessed_df = X_train_val.copy()
        preprocessed_df["readmitted"] = y_train_val

        preprocessed_data_path = Path(
            to_absolute_path(cfg.paths.models.neural_network.preprocessed_data)
        )
        preprocessed_df.to_csv(preprocessed_data_path, index=False)
        logger.info(f"Combined preprocessed data saved to {preprocessed_data_path}")

        logger.info("Preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {str(e)}", exc_info=True)
        logger.error(f"Configuration dump: {OmegaConf.to_yaml(cfg)}")
        raise


if __name__ == "__main__":
    main()
