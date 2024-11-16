"""
Logistic Regression Model
=========================

This module provides a logistic regression model implementation
inheriting from the abstract base Model class.

.. module:: pipeline.models.logisticregression
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
from sklearn.exceptions import NotFittedError
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
        self.feature_names: Optional[np.ndarray] = None  # To store feature names
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
        }.get(optimizer_name, optim.Adam)
        optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        logger.info(f"Optimizer '{optimizer_class.__name__}' created with parameters {optimizer_params}.")
        return optimizer

    def _get_loss_function(self) -> Callable:
        """Get the loss function from the configuration.

        :return: Configured loss function.
        :rtype: Callable
        """
        loss_name = self.training_params.get("loss", "binary_cross_entropy")
        loss_functions = {
            "binary_cross_entropy": nn.BCELoss(),
            "bce": nn.BCELoss(),
            "binary_cross_entropy_with_logits": nn.BCEWithLogitsLoss(),
        }
        loss_fn = loss_functions.get(loss_name.lower(), nn.BCELoss())
        logger.info(f"Loss function '{loss_fn.__class__.__name__}' selected.")
        return loss_fn

    def train(self) -> nn.Module:
        """Train the logistic regression model.

        :return: The trained PyTorch model.
        :rtype: nn.Module
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Please run prepare_data() first.")

        try:
            # Ensure X_train is a DataFrame
            if not isinstance(self.X_train, pd.DataFrame):
                self.X_train = pd.DataFrame(self.X_train)
                logger.warning("Converted X_train to pandas DataFrame.")

            # Data Preprocessing
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.feature_names = self.scaler.feature_names_in_
            logger.info("Scaler fitted on training data.")

            y_train = self.y_train.values.astype(np.float32).reshape(-1, 1)

            # Handle validation data if present
            if self.X_val is not None and self.y_val is not None:
                if not isinstance(self.X_val, pd.DataFrame):
                    self.X_val = pd.DataFrame(self.X_val)
                    logger.warning("Converted X_val to pandas DataFrame.")
                X_val_scaled = self.scaler.transform(self.X_val)
                y_val = self.y_val.values.astype(np.float32).reshape(-1, 1)
                logger.info("Validation data scaled.")
            else:
                X_val_scaled = y_val = None
                logger.info("No validation data provided.")

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

            logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}.")
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
                logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

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
            # Ensure X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
                logger.warning("Converted input X to pandas DataFrame.")

            # Validate feature names
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Input data is missing features: {missing_features}")

            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                logger.warning(f"Input data has extra features not used in training: {extra_features}")

            # Reorder and select only the required features
            X = X[self.feature_names]

            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                predicted_labels = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
            logger.info("Predictions generated successfully.")
            return predicted_labels
        except NotFittedError:
            logger.error("Scaler has not been fitted yet.")
            raise
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities using the trained model.

        :param X: Features for probability prediction.
        :type X: pd.DataFrame
        :return: Predicted probabilities.
        :rtype: np.ndarray
        """
        if not self.model or not self.scaler:
            raise ValueError("Model has not been trained or loaded.")
        try:
            # Ensure X is a DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
                logger.warning("Converted input X to pandas DataFrame.")

            # Validate feature names
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Input data is missing features: {missing_features}")

            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                logger.warning(f"Input data has extra features not used in training: {extra_features}")

            # Reorder and select only the required features
            X = X[self.feature_names]

            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                probabilities = self.model(X_tensor).cpu().numpy()
            logger.info("Prediction probabilities generated successfully.")
            return probabilities.flatten()
        except NotFittedError:
            logger.error("Scaler has not been fitted yet.")
            raise
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            raise

    def get_estimator(self) -> Optional[Any]:
        """Retrieve the trained PyTorch model.

        :return: The trained PyTorch model.
        :rtype: Optional[Any]
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
            feature_importance = dict(zip(self.feature_names, weights))
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
                # Convert to DataFrame with the correct columns
                if not isinstance(x, pd.DataFrame):
                    x_df = pd.DataFrame(x, columns=self.feature_names)
                    logger.debug("Converted input to DataFrame within prediction function.")
                else:
                    x_df = x

                # Validate feature names
                missing_features = set(self.feature_names) - set(x_df.columns)
                if missing_features:
                    raise ValueError(f"Input data is missing features: {missing_features}")

                extra_features = set(x_df.columns) - set(self.feature_names)
                if extra_features:
                    logger.warning(f"Input data has extra features not used in training: {extra_features}")

                # Reorder and select only the required features
                x_df = x_df[self.feature_names]

                x_scaled = self.scaler.transform(x_df)
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    predictions = self.model(x_tensor).cpu().numpy()
                return predictions

            logger.info("Prediction function for SHAP created successfully.")
            return predict_fn
        except Exception as e:
            logger.error(f"Failed to create prediction function: {e}")
            return None

    def save_model(self, filename: str = "model.pth") -> pathlib.Path:
        """Save the trained logistic regression model and scaler.

        :param filename: The filename to save the model under.
        :type filename: str
        :return: Path to the saved model file.
        :rtype: pathlib.Path
        """
        model_path = self.models_dir / filename
        try:
            with self._model_lock:
                torch.save(self.model.state_dict(), model_path)
            # Save the scaler
            scaler_path = self.models_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Model saved successfully at '{model_path}'.")
            logger.info(f"Scaler saved successfully at '{scaler_path}'.")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model and scaler: {e}")
            raise

    def load_model(self, filename: str = "model.pth") -> Optional[Any]:
        """Load a trained logistic regression model and scaler from disk.

        :param filename: The filename of the saved model.
        :type filename: str
        :return: The loaded model.
        :rtype: Optional[Any]
        """
        model_path = self.models_dir / filename
        scaler_path = self.models_dir / "scaler.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at '{model_path}'")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at '{scaler_path}'")

        try:
            # Load the scaler first to retrieve feature names
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded successfully from '{scaler_path}'.")

            # Retrieve feature names from the scaler
            if hasattr(self.scaler, "feature_names_in_"):
                self.feature_names = self.scaler.feature_names_in_
                logger.info(f"Feature names retrieved: {self.feature_names}")
            else:
                raise AttributeError("Scaler does not have 'feature_names_in_' attribute.")

            # Build the model architecture
            if self.feature_names is None:
                raise ValueError("Feature names are not available. Cannot build the model.")
            self.model = self._build_model(input_dim=len(self.feature_names))

            with self._model_lock:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            logger.info(f"Model loaded successfully from '{model_path}'.")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model and scaler: {e}")
            raise
