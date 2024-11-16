import typing
from typing import Any, Optional, Dict

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
        self.predictor: typing.Optional[TabularPredictor] = None

    def train(self) -> TabularPredictor:
        """Train the AutoGluon model and return the trained predictor.

        :return: The trained AutoGluon predictor.
        :rtype: TabularPredictor
        """
        logger.info("Starting training with AutoGluon.")

        # Prepare training and tuning data
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        tuning_data = pd.concat([self.X_val, self.y_val], axis=1)

        # Convert configuration for hyperparameters
        hyperparameters = OmegaConf.to_container(self.model_config.get("hyperparameters", {}), resolve=True)

        # Initialize TabularPredictor
        predictor = TabularPredictor(
            label=self.label_column,
            path=str(self.models_dir),
            eval_metric=self.model_config.get("metric", "roc_auc"),
            problem_type=self.model_config.get("problem_type", "binary"),
        )

        # Fit TabularPredictor
        predictor.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            hyperparameters=hyperparameters,
            time_limit=self.model_config.get("time_limit", 60),
            presets=self.model_config.get("presets", "medium_quality"),
            verbosity=2,
        )

        self.predictor = predictor.load("models/autogluon")

        return self.predictor

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
        if not self.predictor:
            logger.error("Predictor has not been trained or loaded. Cannot retrieve estimator.")
            return None
        try:
            leaderboard = self.predictor.leaderboard(silent=True)
            logger.debug(f"Leaderboard:\n{leaderboard}")
            base_models = leaderboard[~leaderboard["model"].str.contains("Ensemble")]
            if base_models.empty:
                logger.error("No base models found in the predictor.")
                return None
            best_model_name = base_models.iloc[0]["model"]
            best_model = self.predictor._trainer.models.get(best_model_name)
            if not best_model:
                logger.error(f"Best model '{best_model_name}' not found in trainer models.")
                return None
            logger.info(f"Retrieved best base model '{best_model_name}' for feature importance and SHAP.")
            return best_model
        except Exception as e:
            logger.error(f"An error occurred while retrieving the best base model: {e}")
            return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from the best model."""
        model = self.get_estimator()
        if not model:
            logger.error("Could not retrieve model for feature importance calculation")
            return None

        try:
            # Pass the test dataset to feature_importance
            feature_importance = self.predictor.feature_importance(model=model, dataset=self.X_test)

            # Convert to dictionary format if pandas Series or DataFrame
            if hasattr(feature_importance, "to_dict"):
                feature_importance = feature_importance.to_dict()

            # Ensure all values are float type
            feature_importance = {str(k): float(v) for k, v in feature_importance.items()}

            return feature_importance

        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return None
