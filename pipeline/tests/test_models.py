"""
Model Tests
=========

.. module:: tests.test_models
   :synopsis: Test model implementations

.. moduleauthor:: aai540-group3
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.models import (
    LogisticRegressionModel,
    NeuralNetworkModel,
    AutoGluonModel
)

class TestModels:
    """Test suite for model implementations."""

    @pytest.mark.parametrize("model_class,config_key", [
        (LogisticRegressionModel, "logistic"),
        (NeuralNetworkModel, "neural"),
        (AutoGluonModel, "autogluon")
    ])
    def test_model_training(
        self,
        model_class,
        config_key,
        sample_data,
        test_config
    ):
        """Test model training functionality.

        :param model_class: Model class to test
        :param config_key: Configuration key
        :param sample_data: Sample data
        :param test_config: Test configuration
        """
        # Prepare data
        X = sample_data.drop("readmitted", axis=1)
        y = sample_data["readmitted"]

        # Initialize model
        model = model_class(test_config.model[config_key])

        # Train model
        model.train(X, y)
        assert model.is_fitted

        # Test predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions) == {0, 1}

        # Test probabilities
        probas = model.predict_proba(X)
        assert probas.shape == (len(y), 2)
        assert np.all((probas >= 0) & (probas <= 1))

        # Test performance
        auc = roc_auc_score(y, probas[:, 1])
        assert auc > 0.5  # Better than random

    @pytest.mark.parametrize("model_class,config_key", [
        (LogisticRegressionModel, "logistic"),
        (NeuralNetworkModel, "neural"),
        (AutoGluonModel, "autogluon")
    ])
    def test_model_save_load(
        self,
        model_class,
        config_key,
        sample_data,
        test_config,
        temp_dir
    ):
        """Test model serialization.

        :param model_class: Model class to test
        :param config_key: Configuration key
        :param sample_data: Sample data
        :param test_config: Test configuration
        :param temp_dir: Temporary directory
        """
        # Prepare data
        X = sample_data.drop("readmitted", axis=1)
        y = sample_data["readmitted"]

        # Train model
        model = model_class(test_config.model[config_key])
        model.train(X, y)
        original_predictions = model.predict(X)

        # Save model
        save_path = temp_dir / f"{config_key}_model.pkl"
        model.save(save_path)

        # Load model
        loaded_model = model_class(test_config.model[config_key])
        loaded_model.load(save_path)

        # Compare predictions
        loaded_predictions = loaded_model.predict(X)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    @pytest.mark.parametrize("model_class,config_key", [
        (LogisticRegressionModel, "logistic"),
        (NeuralNetworkModel, "neural"),
        (AutoGluonModel, "autogluon")
    ])
    def test_quick_mode(
        self,
        model_class,
        config_key,
        sample_data,
        test_config
    ):
        """Test quick training mode.

        :param model_class: Model class to test
        :param config_key: Configuration key
        :param sample_data: Sample data
        :param test_config: Test configuration
        """
        # Override config for quick mode
        quick_config = test_config.copy()
        quick_config.experiment.name = "quick"

        # Prepare data
        X = sample_data.drop("readmitted", axis=1)
        y = sample_data["readmitted"]

        # Train model in quick mode
        model = model_class(quick_config.model[config_key])
        model.train(X, y)

        # Verify quick mode behavior
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == len(y)
