import os

import hydra
import joblib
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from dvclive import Live


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load configurations
    data_paths = cfg.data.path
    model_params = cfg.model.params
    feature_params = cfg.feature_engineering

    train_data_path = to_absolute_path(f"{data_paths.processed}/train.csv")
    model_output_path = to_absolute_path(cfg.model.model_output_path)

    print("Loading training data...")
    train_df = pd.read_csv(train_data_path)

    X_train = train_df.drop(columns=["readmitted"])
    y_train = train_df["readmitted"]

    # Identify categorical and numerical columns dynamically
    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Preprocessing steps
    numerical_steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]

    if feature_params.add_polynomial_features:
        numerical_steps.append(
            (
                "polynomial",
                PolynomialFeatures(
                    degree=feature_params.poly_degree,
                    interaction_only=True,
                    include_bias=False,
                ),
            )
        )

    numerical_transformer = Pipeline(steps=numerical_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Logistic Regression model
    model = LogisticRegression(**model_params)

    # Pipeline
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

    # Initialize DVCLive
    with Live() as live:
        # Fit the model
        print("Training the model...")
        clf.fit(X_train, y_train)

        # Log metrics to DVCLive
        live.log_metric("training_completed", 1)

        # Perform cross-validation
        print("Performing cross-validation...")
        cross_val_scores = cross_val_score(
            clf, X_train, y_train, cv=5, scoring="accuracy"
        )
        cv_accuracy = cross_val_scores.mean()
        print(f"Cross-Validation Accuracy: {cv_accuracy * 100:.2f}%")

        # Log cross-validation metrics
        live.log_metric("cv_accuracy", cv_accuracy)
        for idx, score in enumerate(cross_val_scores):
            live.log_metric(f"cv_fold_{idx+1}_accuracy", score)

        # Signal DVCLive to move to the next step (for plotting purposes)
        live.next_step()

    # Save the model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(clf, model_output_path)
    print(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    main()
