"""
Neural Network Model
====================

This module provides a neural network model implementation with flexible architecture
and configuration-driven parameters using PyTorch.

.. module:: pipeline.models.neural_network
   :synopsis: Configuration-driven neural network model

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .model import Model


class NeuralNetwork(Model):
    """Neural network model with flexible architecture using PyTorch."""

    def __init__(self):
        """Initialize the neural network model."""
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.history: List[Dict[str, float]] = []

        # Extract configuration parameters
        self.label_column: str = self.cfg.models.base.get("label", "target")
        self.model_params: Dict[str, Any] = self.model_config.get("model_params", {})
        self.training_params: Dict[str, Any] = self.model_config.get("training", {})
        self.architecture_params: Dict[str, Any] = self.model_config.get("architecture", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"NeuralNetwork initialized with label column '{self.label_column}' on device '{self.device}'.")

    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the neural network model from the configuration.

        :param input_dim: The number of input features.
        :type input_dim: int
        :return: A PyTorch neural network model.
        :rtype: nn.Module
        """
        try:
            layers = []
            # Input Layer
            input_layer_cfg = self.architecture_params.get("input_layer", {})
            if input_layer_cfg.get("batch_norm", False):
                layers.append(nn.BatchNorm1d(input_dim))
            if input_layer_cfg.get("dropout", 0) > 0:
                layers.append(nn.Dropout(p=input_layer_cfg["dropout"]))

            # Hidden Layers
            prev_units = input_dim
            for layer_cfg in self.architecture_params.get("hidden_layers", []):
                layers.append(nn.Linear(prev_units, layer_cfg["units"]))
                layers.append(self._get_activation(layer_cfg["activation"]))
                if layer_cfg.get("batch_norm", False):
                    layers.append(nn.BatchNorm1d(layer_cfg["units"]))
                if layer_cfg.get("dropout", 0) > 0:
                    layers.append(nn.Dropout(p=layer_cfg["dropout"]))
                prev_units = layer_cfg["units"]

            # Output Layer
            output_layer_cfg = self.architecture_params.get("output_layer", {})
            layers.append(nn.Linear(prev_units, output_layer_cfg.get("units", 1)))
            layers.append(self._get_activation(output_layer_cfg.get("activation", "sigmoid")))

            # Create the model
            model = nn.Sequential(*layers).to(self.device)
            logger.info("Model built successfully.")
            return model
        except Exception as e:
            logger.error(f"Error building the model: {e}")
            raise

    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function by name.

        :param activation_name: Name of the activation function.
        :type activation_name: str
        :return: Activation function module.
        :rtype: nn.Module
        """
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "softmax": nn.Softmax(dim=1),
            # Add more activations if needed
        }
        return activations.get(activation_name.lower(), nn.ReLU())

    def _get_optimizer(self) -> optim.Optimizer:
        """Create an optimizer from the configuration.

        :return: Configured optimizer.
        :rtype: optim.Optimizer
        """
        optimizer_cfg = self.training_params.get("optimizer", {"name": "adam"})
        optimizer_name = optimizer_cfg.get("name", "adam").lower()
        optimizer_params = optimizer_cfg.get("params", {})
        optimizer_class = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
            # Add more optimizers if needed
        }.get(optimizer_name, optim.Adam)
        optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        return optimizer

    def _get_loss_function(self) -> Callable:
        """Get the loss function from the configuration.

        :return: Configured loss function.
        :rtype: Callable
        """
        loss_name = self.training_params.get("loss", "binary_cross_entropy")
        loss_functions = {
            "binary_cross_entropy": nn.BCELoss(),
            "cross_entropy": nn.CrossEntropyLoss(),
            # Add more loss functions if needed
        }
        return loss_functions.get(loss_name.lower(), nn.BCELoss())

    def _get_class_weights(self, y: pd.Series) -> Optional[torch.Tensor]:
        """Calculate class weights if enabled in the configuration.

        :param y: Target labels.
        :type y: pd.Series
        :return: Class weights tensor or None.
        :rtype: Optional[torch.Tensor]
        """
        class_weight_option = self.training_params.get("class_weight")
        if class_weight_option == "balanced":
            classes = np.unique(y)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
            class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)
            logger.info(f"Computed class weights: {class_weights}")
            return class_weights
        return None

    def train(self) -> Any:
        """Train the neural network model.

        :return: Trained PyTorch model.
        :rtype: Any
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Please run prepare_data() first.")

        try:
            # Data Preprocessing
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            y_train = self.y_train.values

            X_val_scaled = self.scaler.transform(self.X_val) if self.X_val is not None else None
            y_val = self.y_val.values if self.y_val is not None else None

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

            if X_val_scaled is not None:
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)
            else:
                X_val_tensor = y_val_tensor = None

            # Build the model
            self.model = self._build_model(input_dim=X_train_scaled.shape[1])

            # Get optimizer and loss function
            optimizer = self._get_optimizer()
            loss_fn = self._get_loss_function()

            # Get class weights if needed
            class_weights = self._get_class_weights(self.y_train)

            # Training parameters
            batch_size = self.training_params.get("batch_size", 32)
            epochs = self.training_params.get("epochs", 10)

            # Training loop
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            if X_val_tensor is not None:
                val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            else:
                val_loader = None

            logger.info(f"Starting training for {epochs} epochs.")
            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    if class_weights is not None:
                        loss *= class_weights[y_batch.long()]
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * X_batch.size(0)

                avg_loss = epoch_loss / len(train_loader.dataset)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

                # Validation
                if val_loader is not None:
                    self.model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            outputs = self.model(X_batch)
                            loss = loss_fn(outputs, y_batch)
                            val_loss += loss.item() * X_batch.size(0)
                    avg_val_loss = val_loss / len(val_loader.dataset)
                    logger.info(f"Validation Loss: {avg_val_loss:.4f}")

                # Early Stopping (optional)
                # Implement early stopping logic if needed

                # Save training history
                self.history.append({"epoch": epoch + 1, "loss": avg_loss})

            logger.info("Training completed successfully.")
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
            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predicted_labels = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
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
            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                probabilities = self.model(X_tensor).cpu().numpy()
            return pd.DataFrame(probabilities, columns=["probability"])
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            raise

    def get_estimator(self) -> Any:
        """Retrieve the trained PyTorch model.

        :return: The trained PyTorch model.
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
                self.model.eval()
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    return self.model(x_tensor).cpu().numpy()

            return predict_fn
        except Exception as e:
            logger.error(f"Failed to create prediction function: {e}")
            return None

    def save_model(self, model_artifact: Any, filename: str = "model.pth") -> Path:
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
                torch.save(model_artifact.state_dict(), model_path)
            # Save the scaler
            scaler_path = self.models_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Model and scaler saved successfully at '{self.models_dir}'.")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filename: str = "model.pth") -> Any:
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
            # Build the model architecture
            self.model = self._build_model(input_dim=self.X_train.shape[1])
            with self._model_lock:
                self.model.load_state_dict(torch.load(model_path))
            self.model.to(self.device)
            # Load the scaler
            scaler_path = self.models_dir / "scaler.joblib"
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Model and scaler loaded successfully from '{self.models_dir}'.")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores using SHAP.

        :return: Dictionary mapping feature names to importance scores, or None if unavailable.
        :rtype: Optional[Dict[str, float]]
        """
        logger.warning("Feature importance will be computed using SHAP values.")
        return None

    def generate_plots(self, metrics: Any) -> None:
        """Generate and save evaluation plots specific to neural networks.

        :param metrics: Metrics object containing evaluation data.
        :type metrics: Any
        """
        super().generate_plots(metrics)
