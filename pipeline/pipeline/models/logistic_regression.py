"""
Logistic Regression Model
=========================

This module provides a logistic regression model implementation
inheriting from the abstract base Model class.

.. module:: pipeline.models.logistic_regression
   :synopsis: Logistic Regression model using PyTorch

.. moduleauthor:: aai540-group3
"""

import pathlib
from typing import Any, Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.preprocessing import StandardScaler

from .model import Model


class LogisticRegression(Model):
    """Concrete implementation of the Model abstract base class using PyTorch."""

    def __init__(self):
        """Initialize the Logistic Regression model."""
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_params: Dict[str, Any] = self.model_config.get("model_params", {})
        self.training_params: Dict[str, Any] = self.model_config.get("training", {})
        self.label_column: str = self.cfg.models.base.get("label", "target")
        logger.info(
            f"LogisticRegression initialized with label column '{self.label_column}' on device '{self.device}'."
        )

    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the logistic regression model.

        :param input_dim: Number of input features.
        :type input_dim: int
        :return: A PyTorch logistic regression model.
        :rtype: nn.Module
        """
        model = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid()).to(self.device)
        logger.info("Model built successfully.")
        return model

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
            # Add more loss functions if needed
        }
        return loss_functions.get(loss_name.lower(), nn.BCELoss())

    def train(self) -> nn.Module:
        """Train the logistic regression model.

        :return: The trained PyTorch model.
        :rtype: nn.Module
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Please run prepare_data() first.")

        try:
            # Data Preprocessing
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            y_train = self.y_train.values.astype(np.float32).reshape(-1, 1)

            X_val_scaled = self.scaler.transform(self.X_val) if self.X_val is not None else None
            y_val = self.y_val.values.astype(np.float32).reshape(-1, 1) if self.y_val is not None else None

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

            if X_val_scaled is not None:
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            else:
                X_val_tensor = y_val_tensor = None

            # Build the model
            self.model = self._build_model(input_dim=X_train_scaled.shape[1])

            # Get optimizer and loss function
            optimizer = self._get_optimizer()
            loss_fn = self._get_loss_function()

            # Training parameters
            batch_size = self.training_params.get("batch_size", 32)
            epochs = self.training_params.get("epochs", 10)

            # Adjust epochs for quick mode
            if self.mode == "quick":
                epochs = min(epochs, 1)
                logger.info(f"Quick mode: epochs set to {epochs}.")

            # Training loop
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            logger.info(f"Starting training for {epochs} epochs.")
            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * X_batch.size(0)

                avg_loss = epoch_loss / len(train_loader.dataset)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

                # Validation
                if X_val_tensor is not None:
                    self.model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        outputs = self.model(X_val_tensor)
                        val_loss = loss_fn(outputs, y_val_tensor).item()
                    logger.info(f"Validation Loss: {val_loss:.4f}")

            logger.info("Training completed successfully.")
            return self.model
        except Exception as e:
            logger.error(f"Logistic Regression training failed: {e}")
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

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from the model weights.

        :return: Dictionary mapping feature names to their importance scores.
        :rtype: Optional[Dict[str, float]]
        """
        if not self.model:
            logger.error("Model has not been trained or loaded.")
            return None
        try:
            weights = self.model[0].weight.cpu().detach().numpy().flatten()
            feature_importance = dict(zip(self.X_train.columns, weights))
            logger.info("Feature importance calculated successfully.")
            return feature_importance
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return None

    def get_prediction_function(self) -> Optional[Callable]:
        """Get a prediction function suitable for SHAP analysis.

        :return: A callable that takes a NumPy array and returns predictions.
        :rtype: Optional[Callable]
        """
        if not self.model or not self.scaler:
            logger.error("Model has not been trained or loaded.")
            return None
        try:

            def predict_fn(x: np.ndarray) -> np.ndarray:
                x_scaled = self.scaler.transform(x)
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    return self.model(x_tensor).cpu().numpy()

            return predict_fn
        except Exception as e:
            logger.error(f"Failed to create prediction function: {e}")
            return None

    def save_model(self, model_artifact: Any, filename: str = "model.pth") -> pathlib.Path:
        """Save the trained logistic regression model.

        :param model_artifact: The model artifact to save.
        :type model_artifact: Any
        :param filename: The filename to save the model under.
        :type filename: str
        :return: Path to the saved model file.
        :rtype: pathlib.Path
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
        """Load a trained logistic regression model from disk.

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
            if self.X_train is None:
                raise ValueError("Training data not available. Cannot determine input dimension.")
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
