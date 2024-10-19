"""
.. module:: models.neural_network.train
   :synopsis: Training script for the neural network model.

This script trains the neural network model using the preprocessed data. It includes:

- Loading preprocessed data.
- Building the neural network architecture based on configuration.
- Compiling the model with specified optimizer and loss.
- Training the model with early stopping.
- Saving the trained model and training history.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
from pathlib import Path

import numpy as np
import joblib

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=os.getenv("CONFIG_PATH"),
    config_name=os.getenv("CONFIG_NAME"),
    version_base=None,
)
def main(cfg: DictConfig):
    """
    Main function to train the neural network model.

    :param cfg: Configuration object provided by Hydra.
    """
    # Ensure model.name is set to 'neural_network'
    cfg.model.name = "neural_network"

    logger.info("Starting training for neural network model...")
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Paths
    data_dir = Path(to_absolute_path(cfg.paths.models.neural_network.artifacts))
    model_path = Path(to_absolute_path(cfg.paths.models.neural_network.model_file))
    history_path = Path(to_absolute_path(cfg.paths.models.neural_network.history_file))

    # Ensure the output directories exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading preprocessed data...")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    # Build the neural network
    logger.info("Building the neural network model...")
    model = Sequential()
    input_dim = X_train.shape[1]

    # Add layers based on configuration
    for i, layer_config in enumerate(cfg.model.params.layers):
        if i == 0:
            model.add(
                Dense(
                    units=layer_config.units,
                    activation=layer_config.activation,
                    input_dim=input_dim,
                )
            )
        else:
            model.add(
                Dense(units=layer_config.units, activation=layer_config.activation)
            )
        if "dropout" in layer_config and layer_config.dropout > 0:
            model.add(Dropout(layer_config.dropout))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    # Optimizer
    optimizer_config = cfg.model.params.optimizer
    if optimizer_config.type.lower() == "adamw":
        optimizer = AdamW(learning_rate=optimizer_config.learning_rate)
    elif optimizer_config.type.lower() == "adam":
        optimizer = Adam(learning_rate=optimizer_config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config.type}")

    # Convert Hydra config metrics to a Python list with error handling
    try:
        metrics = list(cfg.model.params.metrics)
    except AttributeError:
        logger.warning(
            "Metrics not properly defined in config. Using default 'accuracy'."
        )
        metrics = ["accuracy"]

    # Compile the model
    logger.info("Compiling the model...")
    model.compile(optimizer=optimizer, loss=cfg.model.params.loss, metrics=metrics)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg.model.params.patience,
        restore_best_weights=True,
    )

    # Train the model
    logger.info("Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.model.params.epochs,
        batch_size=cfg.model.params.batch_size,
        callbacks=[early_stopping],
    )

    # Save the model
    logger.info(f"Saving the model to {model_path}...")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save training history
    joblib.dump(history.history, history_path)
    logger.info(f"Training history saved to {history_path}")
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
