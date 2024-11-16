import pathlib
from typing import Any, Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from .model import Model


class LogisticRegression(Model):
    """Concrete implementation of the Model abstract base class using Logistic Regression."""

    def __init__(self):
        """Initialize the Logistic Regression model."""
        super().__init__()
        self.predictor: Optional[SklearnLogisticRegression] = None

        # Extract configuration parameters
        self.label_column: str = self.cfg.models.base.get("label", "target")
        self.problem_type: str = self.model_config.get("problem_type", "binary")
        self.eval_metric: str = self.model_config.get("metric", "roc_auc")
        self.model_params: Dict[str, Any] = self.model_config.get("model_params", {})
        logger.info(f"LogisticRegression initialized with label column '{self.label_column}'.")

    def train(self) -> SklearnLogisticRegression:
        """Train the Logistic Regression model.

        :return: The trained Logistic Regression model.
        :rtype: SklearnLogisticRegression
        """
        logger.info("Starting training with Logistic Regression.")

        # Set random_state if not already set in model_params
        seed = self.cfg.get("seed", 42)
        if "random_state" not in self.model_params:
            self.model_params["random_state"] = seed

        # Initialize the Logistic Regression model with parameters
        self.predictor = SklearnLogisticRegression(**self.model_params)

        # Fit the model
        try:
            self.predictor.fit(self.X_train, self.y_train)
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

        return self.predictor

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model.

        :param X: Features for prediction.
        :type X: pd.DataFrame
        :return: Predicted labels.
        :rtype: np.ndarray
        """
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        try:
            predictions = self.predictor.predict(X)
            return predictions
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
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        try:
            proba = self.predictor.predict_proba(X)
            return pd.DataFrame(proba, columns=self.predictor.classes_)
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            raise

    def get_estimator(self) -> Any:
        """Retrieve the underlying estimator for feature importance and SHAP computations.

        :return: The underlying trained estimator.
        :rtype: Any
        """
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded.")
            return None
        return self.predictor

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from the model coefficients.

        :return: Dictionary mapping feature names to their importance scores, or None if unavailable.
        :rtype: Optional[Dict[str, float]]
        """
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded.")
            return None

        try:
            feature_names = self.X_train.columns
            coefficients = self.predictor.coef_

            if coefficients.ndim == 1:
                # Binary classification
                feature_importance = dict(zip(feature_names, coefficients))
            else:
                # Multiclass classification
                # For simplicity, take the mean of the coefficients across classes
                mean_coefficients = np.mean(coefficients, axis=0)
                feature_importance = dict(zip(feature_names, mean_coefficients))

            logger.info("Feature importance calculated successfully.")
            return feature_importance
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return None

    def get_prediction_function(self) -> Optional[Callable]:
        """Get a prediction function suitable for SHAP analysis.

        :return: A callable that takes a NumPy array and returns predictions, or None if unavailable.
        :rtype: Optional[Callable]
        """
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded.")
            return None

        try:

            def predict_fn(x: np.ndarray) -> np.ndarray:
                return self.predictor.predict_proba(x)[:, 1]  # Probability of positive class

            return predict_fn
        except Exception as e:
            logger.error(f"Failed to create prediction function: {e}")
            return None

    def save_model(self, model_artifact: Any, filename: str = "model.joblib") -> pathlib.Path:
        """Save the trained Logistic Regression model.

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
                joblib.dump(model_artifact, model_path)
            logger.info(f"Model saved successfully at '{model_path}'.")
            return model_path
        except Exception as e:
            logger.error(f"Failed to save model to '{model_path}': {e}")
            raise

    def load_model(self, filename: str = "model.joblib") -> Any:
        """Load a trained Logistic Regression model from disk.

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
                self.predictor = joblib.load(model_path)
            logger.info(f"Model loaded successfully from '{model_path}'.")
            return self.predictor
        except Exception as e:
            logger.error(f"Failed to load model from '{model_path}': {e}")
            raise
