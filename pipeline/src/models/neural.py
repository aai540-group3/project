import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model, load_model

from .base import BaseModel

logger = logging.getLogger(__name__)

class NeuralNetworkModel(BaseModel):
    """Neural Network model implementation."""

    def __init__(self, cfg: DictConfig):
        """Initialize Neural Network model.

        Args:
            cfg: Model configuration
        """
        super().__init__(cfg)
        self.model: Optional[Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_importance_: Optional[Dict[str, float]] = None

    def _build_model(self, input_dim: int) -> Model:
        """Build neural network architecture.

        Args:
            input_dim: Input dimension

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(input_dim,))
        x = inputs

        # Hidden layers
        for units in self.cfg.architecture.hidden_layers:
            x = Dense(units, activation=self.cfg.architecture.activation)(x)
            x = BatchNormalization()(x)
            x = Dropout(self.cfg.architecture.dropout)(x)

        # Output layer
        outputs = Dense(1, activation="sigmoid")(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.cfg.training.learning_rate
            ),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )

        return model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> None:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional arguments passed to fit
        """
        logger.info("Training Neural Network model...")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Build model
        self.model = self._build_model(X_train.shape[1])

        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_auc" if X_val is not None else "auc",
                patience=self.cfg.training.early_stopping.patience,
                mode="max",
                restore_best_weights=True
            ),
            ModelCheckpoint(
                str(Path(self.cfg.model_path) / "best_model.h5"),
                monitor="val_auc" if X_val is not None else "auc",
                mode="max",
                save_best_only=True
            )
        ]

        # Train model
        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val) if X_val is not None else None,
            epochs=self.cfg.training.epochs,
            batch_size=self.cfg.training.batch_size,
            callbacks=callbacks,
            verbose=1,
            **kwargs
        )

        # Calculate feature importance using permutation importance
        self.feature_importance_ = self._calculate_feature_importance(
            X_train,
            y_train
        )

        # Store metrics
        if X_val is not None:
            val_metrics = self.model.evaluate(X_val_scaled, y_val, verbose=0)
            self.metrics = dict(zip(self.model.metrics_names, val_metrics))

        self._is_fitted = True
        logger.info("Neural Network model training completed")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features

        Returns:
            Predicted labels
        """
        self._check_is_fitted()
        X_scaled = self.scaler.transform(X)
        return (self.model.predict(X_scaled) > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions.

        Args:
            X: Features

        Returns:
            Predicted probabilities
        """
        self._check_is_fitted()
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled)
        return np.column_stack([1 - proba, proba])

    def _calculate_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """Calculate feature importance using permutation importance.

        Args:
            X: Features
            y: Labels
            n_repeats: Number of times to repeat permutation

        Returns:
            Feature importance scores
        """
        importance = {}
        X_scaled = self.scaler.transform(X)
        baseline_score = self.model.evaluate(X_scaled, y, verbose=0)[1]  # accuracy

        for column in X.columns:
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[column] = np.random.permutation(X_permuted[column])
                X_permuted_scaled = self.scaler.transform(X_permuted)
                score = self.model.evaluate(
                    X_permuted_scaled,
                    y,
                    verbose=0
                )[1]  # accuracy
                scores.append(baseline_score - score)
            importance[column] = np.mean(scores)

        # Normalize importance scores
        max_importance = max(importance.values())
        return {k: v / max_importance for k, v in importance.items()}

    def save(self, path: Union[str, Path]) -> None:
        """Save model.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        if self.model is not None:
            self.model.save(str(path / "model.h5"))

        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, path / "scaler.joblib")

        # Save additional metadata
        metadata = {
            "metrics": self.metrics,
            "feature_importance": self.feature_importance_,
            "is_fitted": self._is_fitted,
            "config": self.cfg
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load model.

        Args:
            path: Path to load model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        # Load Keras model
        self.model = load_model(str(path / "model.h5"))

        # Load scaler
        self.scaler = joblib.load(path / "scaler.joblib")

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.metrics = metadata["metrics"]
        self.feature_importance_ = metadata["feature_importance"]
        self._is_fitted = metadata["is_fitted"]
        self.cfg = metadata["config"]
