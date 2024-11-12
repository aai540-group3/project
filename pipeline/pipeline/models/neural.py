"""
Neural Network Model Implementation
===============================

.. module:: pipeline.models.neural
   :synopsis: Configuration-driven neural network model

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .base import Model


class NeuralNetworkModel(Model):
    """Neural network model with flexible architecture.

    :param cfg: Model configuration
    :type cfg: DictConfig
    :raises ImportError: If TensorFlow dependencies are not installed
    """

    def __init__(self, cfg: DictConfig):
        """Initialize neural network model.

        :param cfg: Model configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.history: Optional[tf.keras.callbacks.History] = None

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """Build neural network from configuration.

        :param input_dim: Input dimension
        :type input_dim: int
        :return: Compiled Keras model
        :rtype: tf.keras.Model
        """
        # Get architecture config based on mode
        arch_cfg = self.cfg.quick_mode.architecture if self.cfg.experiment.name == "quick" else self.cfg.architecture

        # Build layers
        layers = []

        # Input layer
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        x = input_layer

        if arch_cfg.input_layer.get("batch_norm", False):
            x = tf.keras.layers.BatchNormalization()(x)
        if arch_cfg.input_layer.get("dropout", 0) > 0:
            x = tf.keras.layers.Dropout(arch_cfg.input_layer.dropout)(x)

        # Hidden layers
        for layer_cfg in arch_cfg.hidden_layers:
            x = tf.keras.layers.Dense(units=layer_cfg.units, activation=layer_cfg.activation)(x)

            if layer_cfg.get("batch_norm", False):
                x = tf.keras.layers.BatchNormalization()(x)
            if layer_cfg.get("dropout", 0) > 0:
                x = tf.keras.layers.Dropout(layer_cfg.dropout)(x)

        # Output layer
        output = tf.keras.layers.Dense(
            units=arch_cfg.output_layer.units,
            activation=arch_cfg.output_layer.activation,
        )(x)

        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        # Compile model
        model.compile(
            optimizer=self._get_optimizer(),
            loss=self._get_loss(),
            metrics=self._get_metrics(),
        )

        return model

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create optimizer from configuration.

        :return: Configured optimizer
        :rtype: tf.keras.optimizers.Optimizer
        """
        opt_cfg = (
            self.cfg.quick_mode.training.optimizer
            if self.cfg.experiment.name == "quick"
            else self.cfg.training.optimizer
        )

        return tf.keras.optimizers.get(opt_cfg.name).from_config(
            {
                "learning_rate": opt_cfg.learning_rate,
                "beta_1": opt_cfg.get("beta_1", 0.9),
                "beta_2": opt_cfg.get("beta_2", 0.999),
                "epsilon": opt_cfg.get("epsilon", 1e-07),
            }
        )

    def _get_loss(self) -> Any:
        """Get loss function from configuration.

        :return: Loss function
        :rtype: Any
        """
        train_cfg = self.cfg.quick_mode.training if self.cfg.experiment.name == "quick" else self.cfg.training
        return train_cfg.loss

    def _get_metrics(self) -> List[str]:
        """Get metrics from configuration.

        :return: List of metrics
        :rtype: List[str]
        """
        train_cfg = self.cfg.quick_mode.training if self.cfg.experiment.name == "quick" else self.cfg.training
        return [tf.keras.metrics.get(metric) if isinstance(metric, str) else metric for metric in train_cfg.metrics]

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Get callbacks from configuration.

        :return: List of callbacks
        :rtype: List[tf.keras.callbacks.Callback]
        """
        train_cfg = self.cfg.quick_mode.training if self.cfg.experiment.name == "quick" else self.cfg.training

        callbacks = []

        # Early stopping
        if train_cfg.get("early_stopping"):
            callbacks.append(tf.keras.callbacks.EarlyStopping(**train_cfg.early_stopping))

        # Model checkpoint
        if train_cfg.get("model_checkpoint"):
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(Path(self.cfg.model_path) / "best_model.h5"),
                    **train_cfg.model_checkpoint,
                )
            )

        return callbacks

    def _get_class_weights(self, y: pd.Series) -> Optional[Dict[int, float]]:
        """Calculate class weights if enabled in configuration.

        :param y: Target labels
        :type y: pd.Series
        :return: Class weights dictionary or None
        :rtype: Optional[Dict[int, float]]
        """
        train_cfg = self.cfg.quick_mode.training if self.cfg.experiment.name == "quick" else self.cfg.training

        if train_cfg.get("class_weight") == "balanced":
            classes = np.unique(y)
            weights = compute_class_weight("balanced", classes=classes, y=y)
            return dict(zip(classes, weights))
        return None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> None:
        """Train neural network model.

        :param X_train: Training features
        :type X_train: pd.DataFrame
        :param y_train: Training labels
        :type y_train: pd.Series
        :param X_val: Validation features
        :type X_val: Optional[pd.DataFrame]
        :param y_val: Validation labels
        :type y_val: Optional[pd.Series]
        :param kwargs: Additional training parameters
        :raises RuntimeError: If training fails
        """
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None

            # Build model
            self.model = self._build_model(X_train.shape[1])

            # Get training configuration
            train_cfg = self.cfg.quick_mode.training if self.cfg.experiment.name == "quick" else self.cfg.training

            # Calculate class weights if enabled
            class_weights = self._get_class_weights(y_train)

            # Train model
            self.history = self.model.fit(
                X_train_scaled,
                y_train,
                validation_data=(X_val_scaled, y_val) if X_val is not None else None,
                batch_size=train_cfg.batch_size,
                epochs=train_cfg.epochs,
                callbacks=self._get_callbacks(),
                class_weight=class_weights,
                **kwargs,
            )

            self._is_fitted = True
            logger.info("Neural network training completed successfully")

        except Exception as e:
            logger.error(f"Neural network training failed: {str(e)}")
            raise RuntimeError(f"Neural network training failed: {str(e)}")
