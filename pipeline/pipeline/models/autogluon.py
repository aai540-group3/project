"""
AutoGluon Model
===============

This module provides a machine learning model implementation using AutoGluon,
inheriting from the abstract base Model class.

.. module:: pipeline.models.autogluon_model
    :synopsis: AutoGluon model implementation

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from loguru import logger
from omegaconf import OmegaConf

from .model import Model


class Autogluon(Model):
    """Concrete implementation of the Model abstract base class using AutoGluon."""

    def __init__(self):
        """Initialize the AutoGluon model."""
        super().__init__()
        self.predictor: Optional[TabularPredictor] = None

        # Extract configuration parameters
        self.label_column: str = self.cfg.models.base.get("label", "target")
        self.problem_type: str = self.model_config.get("problem_type", "binary")
        self.eval_metric: str = self.model_config.get("metric", "roc_auc")
        self.time_limit: int = self.model_config.get("time_limit", 60)
        self.presets: str = self.model_config.get("presets", "medium_quality")
        self.hyperparameters: Optional[Dict[str, Any]] = OmegaConf.to_container(
            self.model_config.get("hyperparameters", {}), resolve=True
        )
        logger.info(f"Autogluon initialized with label column '{self.label_column}'.")

    def train(self) -> TabularPredictor:
        """Train the AutoGluon model and return the trained predictor.

        :return: The trained AutoGluon predictor.
        :rtype: TabularPredictor
        """
        logger.info("Starting training with AutoGluon.")

        # Prepare training and tuning data
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        tuning_data = pd.concat([self.X_val, self.y_val], axis=1)

        # Adjust time limit in quick mode
        if self.mode == "quick":
            self.time_limit = min(self.time_limit, 60)  # Limit time to 60 seconds
            logger.info(f"Quick mode: time_limit set to {self.time_limit} seconds.")

        # Initialize TabularPredictor
        self.predictor = TabularPredictor(
            label=self.label_column,
            path=str(self.models_dir),
            eval_metric=self.eval_metric,
            problem_type=self.problem_type,
        )

        # Log hyperparameters for debugging
        logger.debug(f"Hyperparameters before conversion: {self.model_config.get('hyperparameters', {})}")
        logger.debug(f"Converted hyperparameters: {self.hyperparameters}")

        # Fit TabularPredictor
        try:
            fit_kwargs = {
                "train_data": train_data,
                "tuning_data": tuning_data,
                "time_limit": self.time_limit,
                "presets": self.presets,
                "verbosity": 2,
                "use_bag_holdout": True,
            }

            # Only include hyperparameters if they are provided
            if self.hyperparameters:
                fit_kwargs["hyperparameters"] = self.hyperparameters

            self.predictor.fit(**fit_kwargs)
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

        # Save the predictor
        self.save_model(self.predictor)
        logger.info(f"Predictor saved to '{self.models_dir}'.")

        # Verify available models
        self.list_available_models()

        return self.predictor

    def save_model(self, model_artifact: Any, filename: str = "model.pkl") -> Path:
        """Save the AutoGluon predictor.

        :param model_artifact: AutoGluon predictor to be saved.
        :type model_artifact: TabularPredictor
        :param filename: Ignored for AutoGluon as it manages its own filenames.
        :type filename: str
        :return: Path to the saved model directory.
        :rtype: Path
        """
        try:
            model_artifact.save()
            logger.info(f"Model saved successfully at '{self.models_dir}'.")
            return self.models_dir
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filename: str = "model.pkl") -> Any:
        """Load the AutoGluon predictor.

        :param filename: Ignored for AutoGluon as it manages its own filenames.
        :type filename: str
        :return: Loaded AutoGluon predictor.
        :rtype: TabularPredictor
        """
        try:
            self.predictor = TabularPredictor.load(str(self.models_dir))
            logger.info(f"Model loaded from '{self.models_dir}'.")
            return self.predictor
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions using the trained model.

        :param X: Features for prediction.
        :type X: pd.DataFrame
        :return: Predicted labels.
        :rtype: pd.Series
        """
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        return self.predictor.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate prediction probabilities using the trained model.

        :param X: Features for probability prediction.
        :type X: pd.DataFrame
        :return: Predicted probabilities.
        :rtype: pd.DataFrame
        """
        if not self.predictor:
            raise ValueError("Model has not been trained or loaded.")
        return self.predictor.predict_proba(X)

    def get_estimator(self) -> Any:
        """Get the best base model from the predictor.

        :return: Best base model if available, None otherwise.
        :rtype: Any
        """
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded. Cannot retrieve estimator.")
            return None
        try:
            leaderboard = self.predictor.leaderboard(silent=True)
            logger.debug(f"Leaderboard:\n{leaderboard}")

            # Filter out ensemble models
            base_models = leaderboard[~leaderboard["model"].str.contains("Ensemble")]
            if base_models.empty:
                logger.error("No base models found in the predictor.")
                return None

            # Retrieve the name of the best base model
            best_model_name = base_models.iloc[0]["model"]
            logger.info(f"Best base model identified: '{best_model_name}'.")

            # Get the best model
            best_model = self.predictor._trainer.load_model(best_model_name)

            if not best_model:
                logger.error(f"Best model '{best_model_name}' could not be retrieved.")
                return None

            logger.info(f"Retrieved best base model '{best_model_name}' for feature importance and SHAP.")
            return best_model.model  # Return the underlying model
        except Exception as e:
            logger.error(f"An error occurred while retrieving the best base model: {e}")
            return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from the best model.

        :return: Dictionary mapping feature names to their importance scores, or None if calculation fails.
        :rtype: Optional[Dict[str, float]]
        """
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded.")
            return None

        try:
            # Calculate feature importance using the predictor
            importance_df = self.predictor.feature_importance(
                data=pd.concat([self.X_test, self.y_test], axis=1),
                silent=True,
            )

            # Convert to dictionary format
            feature_importance = importance_df["importance"].to_dict()

            # Ensure all values are float type
            feature_importance = {str(k): float(v) for k, v in feature_importance.items()}

            logger.info("Feature importance calculated successfully.")
            return feature_importance

        except KeyError as e:
            logger.error(f"KeyError while calculating feature importance: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return None

    def get_prediction_function(self) -> Optional[Callable]:
        """Get a prediction function suitable for SHAP analysis.

        :return: A callable that takes a DataFrame and returns predictions, or None if unavailable.
        :rtype: Optional[Callable]
        """
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded.")
            return None

        try:
            # Get the best model name
            best_model_name = self.predictor.get_model_best()
            logger.info(f"Using model '{best_model_name}' for SHAP analysis.")

            # Get the prediction function directly from the predictor for this specific model
            def predict_fn(x: np.ndarray) -> np.ndarray:
                df = pd.DataFrame(x, columns=self.X_test.columns)
                proba = self.predictor.predict_proba(df, model=best_model_name)
                if isinstance(proba, pd.DataFrame):
                    return proba.iloc[:, 1].values  # Return probability of positive class
                else:
                    return proba

            return predict_fn

        except Exception as e:
            logger.error(f"Failed to create prediction function: {e}")
            return None

    def list_available_models(self):
        """List all available model names in the predictor."""
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded. Cannot list models.")
            return
        try:
            model_names = self.predictor.get_model_names()
            logger.info(f"Available models in predictor: {model_names}")
        except Exception as e:
            logger.error(f"Failed to retrieve model names: {e}")

    def verify_model_names(self) -> bool:
        """Verify that the model names in the leaderboard match those available in the predictor.

        :return: True if verification succeeds, False otherwise.
        :rtype: bool
        """
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded. Cannot verify model names.")
            return False
        try:
            leaderboard = self.predictor.leaderboard(silent=True)
            available_models = self.predictor.get_model_names()
            leaderboard_models = leaderboard["model"].tolist()

            # Check for each leaderboard model if it exists in available_models
            for model_name in leaderboard_models:
                if model_name not in available_models:
                    logger.warning(f"Model '{model_name}' from leaderboard not found in available models.")
                else:
                    logger.info(f"Model '{model_name}' is available for retrieval.")
            return True
        except Exception as e:
            logger.error(f"Failed to verify model names: {e}")
            return False
