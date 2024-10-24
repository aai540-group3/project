import hashlib
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import shap
import json
from dvclive import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
DATA_DIR = "data/processed"
MODELS_DIR = "models"

# Global variable to store feature names for SHAP
feature_names = None

def create_model(optimizer="adam", dropout_rate=0.5, units=64):
    """
    Creates the neural network model.
    """
    global feature_names
    model = Sequential()
    input_dim = X_train.shape[1]  # Assuming X_train is already defined

    model.add(
        Dense(
            units=units,
            activation="relu",
            input_dim=input_dim,
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(
        Dense(units=units, activation="relu")
    )
    model.add(Dropout(dropout_rate))
    model.add(
        Dense(units=units // 2, activation="relu")
    )
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    """
    Main function to train and evaluate all models.
    """
    try:
        global feature_names
        # Ensure the output directories exist
        Path("models/autogluon").mkdir(parents=True, exist_ok=True)
        Path("models/logistic_regression").mkdir(parents=True, exist_ok=True)
        Path("models/neural_network").mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info("Loading preprocessed data...")
        train_data_path = "data/processed/train.csv"
        test_data_path = "data/processed/test.csv"
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        # Split into train/test/validate
        X = train_data.drop(columns=["readmitted"])
        y = train_data["readmitted"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )

        # Train AutoGluon
        logger.info("Training AutoGluon model...")

        # Calculate hash of training data
        with open(train_data_path, "rb") as f:
            input_hash = hashlib.md5(f.read(), usedforsecurity=False).hexdigest()

        live = Live(dir="models/autogluon", dvcyaml=False)

        # Check if model exists and hash matches
        skip_training = False
        if os.path.exists("models/autogluon/model.pkl") and os.path.exists("models/autogluon/input_hash.txt"):
            with open("models/autogluon/input_hash.txt", "r") as f:
                stored_hash = f.read().strip()
            if stored_hash == input_hash:
                logger.info("AutoGluon model already exists. Skipping training.")
                predictor = joblib.load("models/autogluon/model.pkl")
                skip_training = True

        if not skip_training:
            predictor = TabularPredictor(
                label="readmitted", path="models/autogluon", problem_type="binary"
            )
            predictor.fit(
                train_data=X_train,
                y_train=y_train,
                time_limit=3600,
                presets="best_quality",
                hyperparameters={"GBM": {"num_boost_round": 100}},
                verbosity=2,
            )
            # Save the model
            joblib.dump(predictor, "models/autogluon/model.pkl")
            logger.info(f"AutoGluon model saved to models/autogluon/model.pkl")
            # Save the hash
            with open("models/autogluon/input_hash.txt", "w") as f:
                f.write(input_hash)

        # Evaluate AutoGluon
        logger.info("Evaluating AutoGluon model...")
        y_pred = predictor.predict(X_test)
        y_pred_proba = predictor.predict_proba(X_test)
        positive_class = predictor.class_labels[-1]
        y_pred_proba_positive = y_pred_proba[positive_class]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba_positive, average="weighted"),
            "f1_score": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        with open("models/autogluon/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        for metric_name, metric_value in metrics.items():
            live.log_metric(metric_name, metric_value)

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(predictor.class_labels))
        plt.xticks(tick_marks, predictor.class_labels)
        plt.yticks(tick_marks, predictor.class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("models/autogluon/confusion_matrix.png")
        plt.close()
        live.log_image("confusion_matrix", "models/autogluon/confusion_matrix.png")

        # Log ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_positive)
        roc_auc_value = auc(fpr, tpr)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc_value:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig("models/autogluon/roc_curve.png")
        plt.close()
        live.log_image("roc_curve", "models/autogluon/roc_curve.png")

        # Compute and plot feature importances using SHAP
        logger.info("Calculating feature importances using SHAP...")
        explainer = shap.Explainer(predictor.predict, X_train)
        shap_values = explainer(X_test)

        # Summary plot (bar plot of mean absolute SHAP values)
        shap.summary_plot(
            shap_values,
            features=X_test,
            feature_names=X_test.columns,
            show=False
        )
        plt.savefig("models/autogluon/feature_importances.png")
        plt.close()
        live.log_image("feature_importances", "models/autogluon/feature_importances.png")
        live.end()

        # Train Logistic Regression
        logger.info("Training Logistic Regression model...")
        live = Live(dir="models/logistic_regression", dvcyaml=False)

        # Check if model exists and hash matches
        skip_training = False
        if os.path.exists("models/logistic_regression/model.pkl") and os.path.exists("models/logistic_regression/input_hash.txt"):
            with open("models/logistic_regression/input_hash.txt", "r") as f:
                stored_hash = f.read().strip()
            if stored_hash == input_hash:
                logger.info("Logistic Regression model already exists. Skipping training.")
                model = joblib.load("models/logistic_regression/model.pkl")
                skip_training = True

        if not skip_training:
            model = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42
            )

            # Define parameter grid for GridSearchCV
            param_grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}
            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grid, scoring="roc_auc", cv=5
            )
            grid_search.fit(X_train, y_train)

            # Get the best model from GridSearchCV
            model = grid_search.best_estimator_

            # Save the model
            joblib.dump(model, "models/logistic_regression/model.pkl")
            logger.info(f"Logistic Regression model saved to models/logistic_regression/model.pkl")
            # Save the hash
            with open("models/logistic_regression/input_hash.txt", "w") as f:
                f.write(input_hash)

        # Evaluate Logistic Regression
        logger.info("Evaluating Logistic Regression model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        with open("models/logistic_regression/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        for metric_name, metric_value in metrics.items():
            live.log_metric(metric_name, metric_value)

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(model.classes_))
        plt.xticks(tick_marks, model.classes_)
        plt.yticks(tick_marks, model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig("models/logistic_regression/confusion_matrix.png")
        plt.close()
        live.log_image("confusion_matrix", "models/logistic_regression/confusion_matrix.png")

        # Log ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc_value = auc(fpr, tpr)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc_value:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig("models/logistic_regression/roc_curve.png")
        plt.close()
        live.log_image("roc_curve", "models/logistic_regression/roc_curve.png")

        # Compute and plot feature importances using SHAP
        logger.info("Calculating feature importances using SHAP...")
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_test)

        # Summary plot (bar plot of mean absolute SHAP values)
        shap.summary_plot(
            shap_values,
            features=X_test,
            feature_names=X_test.columns,
            show=False
        )
        plt.savefig("models/logistic_regression/feature_importances.png")
        plt.close()
        live.log_image("feature_importances", "models/logistic_regression/feature_importances.png")
        live.end()

        # Train Neural Network
        logger.info("Training Neural Network model...")
        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model_path = "models/neural_network/model.h5"
        history_path = "models/neural_network/history.pkl"

        # Calculate hash of input data
        with open(train_data_path, "rb") as f:
            input_hash = hashlib.md5(f.read(), usedforsecurity=False).hexdigest()

        # Check if model file exists before checking hashes
        if os.path.exists(model_path) and os.path.exists("models/neural_network/input_hash.txt"):
            with open("models/neural_network/input_hash.txt", "r") as f:
                stored_hash = f.read().strip()
            if stored_hash == input_hash:
                logger.info("Neural Network model already exists with the same input hash. Skipping training.")
                return

        # Store feature names for SHAP
        feature_names = X_train.columns

        # Create KerasClassifier for GridSearchCV
        model = KerasClassifier(build_fn=create_model, verbose=0)

        # Define parameter grid for GridSearchCV
        param_grid = {
            "optimizer": ["adam", "adamw"],
            "dropout_rate": [0.2, 0.5],
            "units": [32, 64],
            "batch_size": [16, 32],
            "epochs": [50, 100],
        }
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, scoring="roc_auc", cv=5
        )
        grid_search.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # Get the best model from GridSearchCV
        best_model = grid_search.best_estimator_

        # Train the best model
        logger.info("Starting model training...")
        history = best_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping]
        )

        # Save the model
        logger.info(f"Saving the model to {model_path}...")
        best_model.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save training history
        joblib.dump(history.history, history_path)
        logger.info(f"Training history saved to {history_path}")

        # Save the input hash
        with open("models/neural_network/input_hash.txt", "w") as f:
            f.write(input_hash)
        logger.info(f"Input hash saved to models/neural_network/input_hash.txt")

        live = Live(dir="models/neural_network", dvcyaml=False)

        # Evaluate Neural Network
        logger.info("Evaluating Neural Network model...")
        y_pred_proba = best_model.predict(X_test).ravel()
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        with open("models/neural_network/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        for metric_name, metric_value in metrics.items():
            live.log_metric(metric_name, metric_value)

        # Generate and log Confusion Matrix plot
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Add labels to each cell
        thresh = conf_matrix.max() / 2.0
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(
                j,
                i,
                format(conf_matrix[i, j], "d"),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.savefig("models/neural_network/confusion_matrix.png")
        plt.close()
        live.log_image("confusion_matrix", "models/neural_network/confusion_matrix.png")
        logger.info(f"Confusion matrix plot saved to models/neural_network/confusion_matrix.png")

        # Generate and log ROC Curve plot
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc_value = auc(fpr, tpr)

        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc_value:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig("models/neural_network/roc_curve.png")
        plt.close()
        live.log_image("roc_curve", "models/neural_network/roc_curve.png")
        logger.info(f"ROC curve plot saved to models/neural_network/roc_curve.png")

        # Compute and plot Feature Importances using SHAP
        logger.info("Calculating feature importances using SHAP...")

        # Due to computational constraints, use a subset of the data
        X_sample = X_test[:100]
        background = X_test[:100]

        # Create SHAP explainer
        explainer = shap.DeepExplainer(best_model.model, background)
        shap_values = explainer.shap_values(X_sample)

        # Debugging output
        logger.info(f"Type of shap_values: {type(shap_values)}")
        logger.info(f"shap_values shape: {np.array(shap_values).shape}")
        logger.info(f"X_sample shape: {X_sample.shape}")
        logger.info(f"Feature names length: {len(feature_names)}")

        # Adjust shap_values for plotting
        shap_values_to_plot = np.squeeze(shap_values)
        logger.info(f"After squeezing, shap_values shape: {shap_values_to_plot.shape}")

        # Ensure shapes match
        assert (
            shap_values_to_plot.shape == X_sample.shape
        ), f"Mismatch in shapes between shap_values ({shap_values_to_plot.shape}) and X_sample ({X_sample.shape})"

        # Plot feature importances
        shap.summary_plot(
            shap_values_to_plot,
            features=X_sample,
            feature_names=feature_names,
            show=False,
        )
        plt.savefig("models/neural_network/feature_importances.png")
        plt.close()
        live.log_image("feature_importances", "models/neural_network/feature_importances.png")
        logger.info(f"Feature importances plot saved to models/neural_network/feature_importances.png")

        live.end()
        logger.info("Evaluation completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()