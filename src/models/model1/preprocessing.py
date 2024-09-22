import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   PolynomialFeatures, StandardScaler)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_int(X):
    return X.astype(int)

def create_preprocessor(categorical_cols: list, numerical_cols: list, boolean_cols: list, feature_params: dict):
    """Creates a preprocessing pipeline."""
    transformers = []

    if numerical_cols:
        numerical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("polynomial", PolynomialFeatures(
                degree=feature_params.get("poly_degree", 2),
                interaction_only=True,
                include_bias=False,
            ) if feature_params.get("add_polynomial_features", False) else "passthrough"),
        ])
        transformers.append(("num", numerical_transformer, numerical_cols))

    if categorical_cols:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if boolean_cols:
        boolean_transformer = FunctionTransformer(convert_to_int)
        transformers.append(("bool", boolean_transformer, boolean_cols))

    return ColumnTransformer(transformers)

def get_feature_names(preprocessor, input_features):
    feature_names = []

    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            if isinstance(trans.named_steps.get('polynomial'), PolynomialFeatures):
                feature_names.extend(trans.named_steps['polynomial'].get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        elif name == 'cat':
            feature_names.extend(trans.named_steps['onehot'].get_feature_names_out(cols))
        elif name == 'bool':
            feature_names.extend(cols)

    return feature_names

@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        data_paths = cfg.data.path
        feature_params = cfg.feature_engineering

        train_data_path = Path(to_absolute_path(f"{data_paths.processed}/train.csv"))
        preprocessed_train_data_path = Path(to_absolute_path(f"{data_paths.processed}/train_preprocessed.csv"))
        preprocessor_path = Path(to_absolute_path(f"{data_paths.processed}/preprocessor.joblib"))
        feature_names_path = Path(to_absolute_path(f"{data_paths.processed}/feature_names.joblib"))

        logger.info("Loading processed training data...")
        train_df = pd.read_csv(train_data_path)

        logger.info("Identifying feature types...")
        X = train_df.drop(columns=["readmitted"])
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Boolean columns: {boolean_cols}")
        logger.info(f"Numerical columns: {numerical_cols}")

        logger.info("Creating preprocessing pipeline...")
        preprocessor = create_preprocessor(categorical_cols, numerical_cols, boolean_cols, feature_params)

        logger.info("Preprocessing training data...")
        X_train_preprocessed = preprocessor.fit_transform(X)

        feature_names = get_feature_names(preprocessor, X.columns)

        preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
        preprocessed_df["readmitted"] = train_df["readmitted"]

        logger.info("Saving preprocessed training data...")
        preprocessed_train_data_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_df.to_csv(preprocessed_train_data_path, index=False)
        logger.info(f"Preprocessed train data saved to {preprocessed_train_data_path}")

        logger.info("Saving preprocessor...")
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        logger.info("Saving feature names...")
        joblib.dump(feature_names, feature_names_path)
        logger.info(f"Feature names saved to {feature_names_path}")

        logger.info(f"Original data shape: {train_df.shape}")
        logger.info(f"Preprocessed data shape: {preprocessed_df.shape}")
        logger.info(f"Number of features: {len(feature_names)}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()