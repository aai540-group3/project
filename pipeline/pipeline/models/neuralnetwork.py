"""
Neural Network Model
===================

This module provides a neural network model implementation
inheriting from the abstract base Model class.

.. module:: pipeline.models.neuralnetwork
   :synopsis: Neural Network model using PyTorch

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import accuracy_score
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
        self.class_weights: Optional[torch.Tensor] = None

        # Extract configuration parameters
        self.label_column: str = self.cfg.models.base.get("label", "target")
        self.model_params: Dict[str, Any] = self.model_config.get("model_params", {})
        self.training_params: Dict[str, Any] = self.model_config.get("training", {})
        self.architecture_params: Dict[str, Any] = self.model_config.get("architecture", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plots_dir = Path(self.cfg.paths.plots) / self.name
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
            # Remove activation function for output layer if using BCEWithLogitsLoss
            if output_layer_cfg.get("activation") and not self.using_bce_with_logits():
                layers.append(self._get_activation(output_layer_cfg["activation"]))

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
        }.get(optimizer_name, optim.Adam)

        logger.info(f"Initializing optimizer '{optimizer_name}' with parameters: {optimizer_params}")

        optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        return optimizer

    def _get_loss_function(self) -> Callable:
        """Get the loss function from the configuration.

        :return: Configured loss function.
        :rtype: Callable
        """
        loss_name = self.training_params.get("loss", "binary_cross_entropy")
        if loss_name.lower() == "binary_cross_entropy":
            if self.using_bce_with_logits():
                # Use BCEWithLogitsLoss for better numerical stability
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
            else:
                # Use BCELoss with reduction 'none' to get per-sample losses
                loss_fn = nn.BCELoss(reduction="none")
        elif loss_name.lower() == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss(reduction="none")
        else:
            logger.error(f"Unsupported loss function: {loss_name}")
            raise ValueError(f"Unsupported loss function: {loss_name}")
        return loss_fn

    def using_bce_with_logits(self) -> bool:
        """Check if BCEWithLogitsLoss is used based on the output layer activation."""
        output_layer_cfg = self.architecture_params.get("output_layer", {})
        activation = output_layer_cfg.get("activation", "").lower()
        return activation != "sigmoid"

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
            class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            logger.info(f"Computed class weights: {class_weights}")

            if self.using_bce_with_logits():
                pos_weight = class_weights[1] / class_weights[0]
                pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
                logger.info(f"Computed pos_weight for BCEWithLogitsLoss: {pos_weight}")
                return pos_weight
            else:
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
            self.class_weights = self._get_class_weights(self.y_train)
            self.model = self._build_model(input_dim=X_train_scaled.shape[1])

            # Get optimizer and loss function
            optimizer = self._get_optimizer()
            loss_fn = self._get_loss_function()

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
                all_train_preds = []
                all_train_targets = []

                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)

                    # Adjust outputs and targets if using BCEWithLogitsLoss
                    if self.using_bce_with_logits():
                        outputs = outputs.view(-1)
                        y_batch = y_batch.view(-1)
                    else:
                        outputs = torch.sigmoid(outputs).view(-1)
                        y_batch = y_batch.view(-1)

                    loss = loss_fn(outputs, y_batch)
                    if self.class_weights is not None and not self.using_bce_with_logits():
                        y_batch_indices = y_batch.long()
                        sample_weights = self.class_weights[y_batch_indices]
                        loss = loss * sample_weights
                    # Compute mean loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * X_batch.size(0)

                    # Collect predictions and targets for accuracy
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    all_train_preds.extend(preds.cpu().numpy())
                    all_train_targets.extend(y_batch.cpu().numpy())

                avg_loss = epoch_loss / len(train_loader.dataset)
                train_accuracy = accuracy_score(all_train_targets, all_train_preds)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")

                # Validation
                if val_loader is not None:
                    self.model.eval()
                    val_loss = 0.0
                    all_val_preds = []
                    all_val_targets = []
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            outputs = self.model(X_batch)
                            if self.using_bce_with_logits():
                                outputs = outputs.view(-1)
                                y_batch = y_batch.view(-1)
                            else:
                                outputs = torch.sigmoid(outputs).view(-1)
                                y_batch = y_batch.view(-1)
                            loss = loss_fn(outputs, y_batch)
                            val_loss += loss.mean().item() * X_batch.size(0)

                            # Collect predictions and targets for accuracy
                            preds = (torch.sigmoid(outputs) > 0.5).float()
                            all_val_preds.extend(preds.cpu().numpy())
                            all_val_targets.extend(y_batch.cpu().numpy())
                    avg_val_loss = val_loss / len(val_loader.dataset)
                    val_accuracy = accuracy_score(all_val_targets, all_val_preds)
                    logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                else:
                    avg_val_loss = None
                    val_accuracy = None

                # Save training history
                history_entry = {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "accuracy": train_accuracy,
                }
                if avg_val_loss is not None:
                    history_entry["val_loss"] = avg_val_loss
                if val_accuracy is not None:
                    history_entry["val_accuracy"] = val_accuracy
                self.history.append(history_entry)

            logger.info("Training completed successfully.")

            # Generate and save training plots
            self._save_training_plots()

            return self.model
        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            raise

    def _save_training_plots(self) -> None:
        """Generate and save training loss and accuracy plots."""
        import matplotlib.pyplot as plt

        # Ensure the plots directory exists
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Extract loss and accuracy from history
        epochs = [h["epoch"] for h in self.history]
        losses = [h["loss"] for h in self.history]
        val_losses = [h.get("val_loss") for h in self.history]
        accuracies = [h.get("accuracy") for h in self.history]
        val_accuracies = [h.get("val_accuracy") for h in self.history]

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, label="Training Loss")
        if any(val_losses):
            plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_plot_path = self.plots_dir / "training_loss.png"
        plt.savefig(loss_plot_path)
        plt.close()
        logger.info(f"Training loss plot saved at '{loss_plot_path}'.")

        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, accuracies, label="Training Accuracy")
        if any(val_accuracies):
            plt.plot(epochs, val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        acc_plot_path = self.plots_dir / "training_accuracy.png"
        plt.savefig(acc_plot_path)
        plt.close()
        logger.info(f"Training accuracy plot saved at '{acc_plot_path}'.")

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
                logits = self.model(X_tensor)
                if self.using_bce_with_logits():
                    probabilities = torch.sigmoid(logits)
                else:
                    probabilities = logits
                predicted_labels = (probabilities.cpu().numpy() > 0.5).astype(int).flatten()
            return predicted_labels
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
            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(X_tensor)
                if self.using_bce_with_logits():
                    probabilities = torch.sigmoid(logits)
                else:
                    probabilities = logits
                probabilities = probabilities.cpu().numpy().flatten()
            return probabilities  # Return as NumPy array
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
                    logits = self.model(x_tensor)
                    if self.using_bce_with_logits():
                        probabilities = torch.sigmoid(logits)
                    else:
                        probabilities = logits
                    return probabilities.cpu().numpy()

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
