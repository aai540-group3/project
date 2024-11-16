"""
Neural Network Model Implementation
===================================

This module provides a neural network model implementation with flexible architecture
and configuration-driven parameters.

.. module:: pipeline.models.neural_network
   :synopsis: Configuration-driven neural network model

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import shap

from .model import Model
from .metrics import Metrics


class NeuralNetwork(Model):
    """Neural network model with flexible architecture."""

    def __init__(self):
        """Initialize the neural network model."""
        super().__init__()
        self.model: Optional[tf.keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.history: Optional[tf.keras.callbacks.History] = None

        # Extract configuration parameters
        self.label_column: str = self.cfg.models.base.get("label", "target")
        self.model_params: Dict[str, Any] = self.model_config.get("model_params", {})
        self.training_params: Dict[str, Any] = self.model_config.get("training", {})
        self.architecture_params: Dict[str, Any] = self.model_config.get("architecture", {})
        logger.info(f"NeuralNetwork initialized with label column '{self.label_column}'.")

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        """Build the neural network model from the configuration.

        :param input_dim: The number of input features.
        :type input_dim: int
        :return: A compiled Keras model.
        :rtype: tf.keras.Model
        """
        try:
            # Input layer
            inputs = tf.keras.Input(shape=(input_dim,))
            x = inputs

            # Apply input layer configurations
            input_layer_cfg = self.architecture_params.get("input_layer", {})
            if input_layer_cfg.get("batch_norm", False):
                x = tf.keras.layers.BatchNormalization()(x)
            if input_layer_cfg.get("dropout", 0) > 0:
                x = tf.keras.layers.Dropout(rate=input_layer_cfg["dropout"])(x)

            # Hidden layers
            for layer_cfg in self.architecture_params.get("hidden_layers", []):
                x = tf.keras.layers.Dense(
                    units=layer_cfg["units"],
                    activation=layer_cfg["activation"],
                    kernel_regularizer=self._get_regularizer(layer_cfg.get("kernel_regularizer", {})),
                )(x)
                if layer_cfg.get("batch_norm", False):
                    x = tf.keras.layers.BatchNormalization()(x)
                if layer_cfg.get("dropout", 0) > 0:
                    x = tf.keras.layers.Dropout(rate=layer_cfg["dropout"])(x)

            # Output layer
            output_layer_cfg = self.architecture_params.get("output_layer", {})
            outputs = tf.keras.layers.Dense(
                units=output_layer_cfg.get("units", 1),
                activation=output_layer_cfg.get("activation", "sigmoid"),
            )(x)

            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Compile model
            model.compile(
                optimizer=self._get_optimizer(),
                loss=self._get_loss(),
                metrics=self._get_metrics(),
            )

            logger.info("Model built and compiled successfully.")
            return model
        except Exception as e:
            logger.error(f"Error building the model: {e}")
            raise

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create an optimizer from the configuration.

        :return: Configured optimizer.
        :rtype: tf.keras.optimizers.Optimizer
        """
        optimizer_cfg = self.training_params.get("optimizer", {"name": "adam"})
        optimizer_name = optimizer_cfg.get("name", "adam")
        optimizer_params = optimizer_cfg.get("params", {})
        optimizer = tf.keras.optimizers.get(optimizer_name)
        optimizer = optimizer(**optimizer_params)
        return optimizer

    def _get_loss(self) -> Union[str, Callable]:
        """Get the loss function from the configuration.

        :return: Configured loss function.
        :rtype: Union[str, Callable]
        """
        loss = self.training_params.get("loss", "binary_crossentropy")
        return loss

    def _get_metrics(self) -> List[Union[str, Callable]]:
        """Get the list of metrics from the configuration.

        :return: List of metrics.
        :rtype: List[Union[str, Callable]]
        """
        metrics_cfg = self.training_params.get("metrics", ["accuracy"])
        metrics = [tf.keras.metrics.get(metric) if isinstance(metric, str) else metric for metric in metrics_cfg]
        return metrics

    def _get_regularizer(self, reg_cfg: Dict[str, Any]) -> Optional[tf.keras.regularizers.Regularizer]:
        """Create a regularizer from the configuration.

        :param reg_cfg: Regularizer configuration.
        :type reg_cfg: Dict[str, Any]
        :return: Configured regularizer or None.
        :rtype: Optional[tf.keras.regularizers.Regularizer]
        """
        if not reg_cfg:
            return None
        reg_name = reg_cfg.get("name")
        reg_params = reg_cfg.get("params", {})
        regularizer = tf.keras.regularizers.get(reg_name)
        return regularizer(**reg_params)

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Get the list of callbacks from the configuration.

        :return: List of callbacks.
        :rtype: List[tf.keras.callbacks.Callback]
        """
        callbacks_cfg = self.training_params.get("callbacks", {})
        callbacks = []

        # Early Stopping
        if callbacks_cfg.get("early_stopping", {}).get("enabled", False):
            early_stopping_params = callbacks_cfg["early_stopping"].get("params", {})
            callbacks.append(tf.keras.callbacks.EarlyStopping(**early_stopping_params))

        # Model Checkpoint
        if callbacks_cfg.get("model_checkpoint", {}).get("enabled", False):
            checkpoint_params = callbacks_cfg["model_checkpoint"].get("params", {})
            checkpoint_params["filepath"] = str(self.models_dir / "best_model.h5")
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(**checkpoint_params))

        # TensorBoard
        if callbacks_cfg.get("tensorboard", {}).get("enabled", False):
            tensorboard_params = callbacks_cfg["tensorboard"].get("params", {})
            tensorboard_params["log_dir"] = str(self.models_dir / "logs")
            callbacks.append(tf.keras.callbacks.TensorBoard(**tensorboard_params))

        logger.info(f"Configured callbacks: {[type(cb).__name__ for cb in callbacks]}")
        return callbacks

    def _get_class_weights(self, y: pd.Series) -> Optional[Dict[int, float]]:
        """Calculate class weights if enabled in the configuration.

        :param y: Target labels.
        :type y: pd.Series
        :return: Class weights dictionary or None.
        :rtype: Optional[Dict[int, float]]
        """
        class_weight_option = self.training_params.get("class_weight")
        if class_weight_option == "balanced":
            classes = np.unique(y)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
            class_weights = dict(zip(classes, weights))
            logger.info(f"Computed class weights: {class_weights}")
            return class_weights
        return None

    def train(self) -> Any:
        """Train the neural network model.

        :return: Trained Keras model.
        :rtype: Any
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Please run prepare_data() first.")

        try:
            # Data Preprocessing
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_val_scaled = self.scaler.transform(self.X_val) if self.X_val is not None else None

            # Build and compile the model
            self.model = self._build_model(input_dim=X_train_scaled.shape[1])

            # Get class weights if needed
            class_weights = self._get_class_weights(self.y_train)

            # Training parameters
            batch_size = self.training_params.get("batch_size", 32)
            epochs = self.training_params.get("epochs", 10)

            # Train the model
            self.history = self.model.fit(
                X_train_scaled,
                self.y_train,
                validation_data=(X_val_scaled, self.y_val) if self.X_val is not None else None,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=self._get_callbacks(),
                class_weight=class_weights,
                verbose=1,
            )

            logger.info("Neural network training completed successfully.")
            return self.model
        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model.

        :param X: Features for prediction.
        :type X: pd.DataFrame
        :return: Predicted labels.
        :rtype: np.ndarray
        """
        if not self.model or not self.scaler:
            raise ValueError("Model has not been trained or loaded.")
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            # Convert probabilities to class labels
            predicted_labels = (predictions > 0.5).astype(int).flatten()
            return predicted_labels
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate prediction probabilities using the trained model.

        :param X: Features for probability prediction.
        :type X: pd.DataFrame
        :return: Predicted probabilities.
        :rtype: pd.DataFrame
        """
        if not self.model or not self.scaler:
            raise ValueError("Model has not been trained or loaded.")
        try:
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict(X_scaled)
            return pd.DataFrame(probabilities, columns=["probability"])
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            raise

    def get_estimator(self) -> Any:
        """Retrieve the trained Keras model.

        :return: The trained Keras model.
        :rtype: Any
        """
        if not self.model:
            logger.error("Model has not been trained or loaded.")
            return None
        return self.model

    def get_prediction_function(self) -> Optional[Callable]:
        """Get a prediction function suitable for SHAP analysis.

        :return: A callable that takes a NumPy array and returns predictions.
        :rtype: Optional[Callable]
        """
        if not self.model:
            logger.error("Model has not been trained or loaded.")
            return None
        try:

            def predict_fn(x: np.ndarray) -> np.ndarray:
                return self.model.predict(x)

            return predict_fn
        except Exception as e:
            logger.error(f"Failed to create prediction function: {e}")
            return None

    def save_model(self, model_artifact: Any, filename: str = "model.h5") -> Path:
        """Save the trained model to disk.

        :param model_artifact: The model artifact to save.
        :type model_artifact: Any
        :param filename: The filename to save the model under.
        :type filename: str
        :return: Path to the saved model file.
        :rtype: Path
        """
        model_path = self.models_dir / filename
        try:
            with self._model_lock:
                model_artifact.save(model_path)
            # Save the scaler
            scaler_path = self.models_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Model and scaler saved successfully at '{self.models_dir}'.")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filename: str = "model.h5") -> Any:
        """Load a trained model from disk.

        :param filename: The filename of the saved model.
        :type filename: str
        :return: The loaded model.
        :rtype: Any
        """
        model_path = self.models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at '{model_path}'")
        try:
            with self._model_lock:
                self.model = tf.keras.models.load_model(model_path)
            # Load the scaler
            scaler_path = self.models_dir / "scaler.joblib"
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Model and scaler loaded successfully from '{self.models_dir}'.")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores using weights.

        :return: Dictionary mapping feature names to importance scores, or None if unavailable.
        :rtype: Optional[Dict[str, float]]
        """
        logger.warning("Feature importance is not directly available for neural networks.")
        return None

    def generate_plots(self, metrics: Metrics) -> None:
        """Generate and save evaluation plots specific to neural networks.

        :param metrics: Metrics object containing evaluation data.
        :type metrics: Metrics
        """
        super().generate_plots(metrics)

    def compute_shap_values(self, background: Optional[pd.DataFrame] = None) -> Optional[shap.Explanation]:
        """Compute SHAP values for the model and feature data.

        :param background: Background dataset for SHAP.
        :type background: Optional[pd.DataFrame]
        :return: SHAP values explanation object.
        :rtype: Optional[shap.Explanation]
        """
        if not self.model or not self.X_test is not None:
            logger.warning("Model or feature data not provided. Cannot compute SHAP values.")
            return None

        try:
            # Use a smaller sample in quick mode
            if self.mode == "quick":
                sample_size = min(1000, len(self.X_test))
                X_sample = self.X_test.sample(n=sample_size, random_state=42)
                logger.info(f"Using a sample size of {X_sample.shape[0]} for SHAP analysis in quick mode.")
            else:
                X_sample = self.X_test

            X_sample_scaled = self.scaler.transform(X_sample)

            # Use DeepExplainer for neural networks
            if background is not None:
                background_scaled = self.scaler.transform(background)
            else:
                background_scaled = shap.sample(X_sample_scaled, 100, random_state=42)

            explainer = shap.DeepExplainer(self.model, background_scaled)
            shap_values = explainer.shap_values(X_sample_scaled)

            # Create SHAP explanation object
            shap_explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value[0],
                data=X_sample_scaled,
                feature_names=self.X_test.columns.tolist(),
            )

            logger.info("Successfully computed SHAP values.")
            return shap_explanation
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
