import logging
import os

import tensorflow as tf
import yaml
from dvclive import Live
from src.utils import (apply_smote, calculate_metrics, load_data,
                       log_class_distribution, plot_confusion_matrix,
                       plot_roc_curve, preprocess_data, save_metrics,
                       scale_features, setup_artifacts, split_data)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(mode: str) -> dict:
    """Load configuration for the specified mode (quick/full)."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)[mode]


def create_model(input_dim: int, config: dict) -> tuple:
    """Create a Keras model based on the hyperparameters from the configuration.

    Args:
        input_dim (int): The number of input features.
        config (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing the compiled model and the batch size.
    """
    model = Sequential()
    n_layers = config["optimization"]["param_space"]["n_layers"]["high"]
    units_first = config["optimization"]["param_space"]["units_first"]["high"]
    units_factor = config["optimization"]["param_space"]["units_factor"]
    dropout = config["optimization"]["param_space"]["dropout"]
    learning_rate = config["optimization"]["param_space"]["learning_rate"]
    activation = config["optimization"]["param_space"]["activation"]
    optimizer_name = config["optimization"]["param_space"]["optimizer"]

    model.add(Dense(units_first, activation=activation, input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    units = units_first
    for _ in range(n_layers - 1):
        units = max(int(units * units_factor), 1)
        model.add(Dense(units, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = (
        Adam(learning_rate=learning_rate)
        if optimizer_name == "adam"
        else SGD(learning_rate=learning_rate)
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model, config["training"]["batch_size"]


def train_neural_network(mode: str) -> None:
    """Train a neural network model based on the provided configuration."""
    config = load_config(mode)
    artifacts_path = config["paths"]["artifacts"]
    setup_artifacts(artifacts_path, config["paths"]["subdirs"])

    live = Live(dir=os.path.join(artifacts_path, "metrics"), dvcyaml=False)

    try:
        df, _ = load_data(config["paths"]["data"])
        X, y = preprocess_data(df, config["model"]["target"])
        log_class_distribution(live, y)

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X,
            y,
            config["splits"]["test_size"],
            config["splits"]["random_state"],
        )

        X_train_balanced, y_train_balanced = apply_smote(
            X_train, y_train, config["model"]["random_state"]
        )
        log_class_distribution(live, y_train_balanced, prefix="after")

        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train_balanced,
            X_val,
            X_test,
        )

        # Log split sizes
        live.log_params(
            {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "train_size_after_smote": len(X_train_balanced),
            }
        )

        # Create and train the model
        model, batch_size = create_model(X_train_scaled.shape[1], config)
        early_stopping = EarlyStopping(
            monitor="val_auc",
            patience=10,
            mode="max",
            restore_best_weights=True,
        )

        logger.info("Training the model...")
        history = model.fit(
            X_train_scaled,
            y_train_balanced,
            validation_data=(X_val_scaled, y_val),
            epochs=config["training"]["epochs"],
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1,
        )

        # Evaluate the model
        val_pred = model.predict(X_val_scaled).flatten()
        test_pred = model.predict(X_test_scaled).flatten()

        metrics = calculate_metrics(y_val, val_pred, y_test, test_pred)
        save_metrics(metrics, artifacts_path)

        # Generate plots
        plot_confusion_matrix(
            y_test,
            (test_pred > 0.5).astype(int),
            artifacts_path,
            config["plots"],
        )
        plot_roc_curve(
            y_val,
            val_pred,
            y_test,
            test_pred,
            metrics,
            artifacts_path,
            config["plots"],
        )

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if live:
            live.end()


if __name__ == "__main__":
    mode = os.getenv("MODE", "quick").lower()
    train_neural_network(mode)
