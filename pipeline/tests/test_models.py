# tests/test_models.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.models import (
    LogisticRegressionModel,
    NeuralNetworkModel,
    AutoGluonModel
)

class TestModels:
    """Test suite for all model implementations."""

    @pytest.mark.parametrize("model_class,config_key", [
        (LogisticRegressionModel, "logistic"),
        (NeuralNetworkModel, "neural"),
        (AutoGluonModel, "autogluon")
    ])
    def test_model_training(self, model_class, config_key, test_data, test_config):
        """Test model training and basic functionality."""
        model = model_class(test_config.model[config_key])
        X = test_data.drop("readmitted", axis=1)
        y = test_data["readmitted"]

        # Test training
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

    @pytest.mark.parametrize("model_class,config_key", [
        (LogisticRegressionModel, "logistic"),
        (NeuralNetworkModel, "neural"),
        (AutoGluonModel, "autogluon")
    ])
    def test_model_save_load(self, model_class, config_key, test_data, test_config, temp_dir):
        """Test model saving and loading."""
        model = model_class(test_config.model[config_key])
        X = test_data.drop("readmitted", axis=1)
        y = test_data["readmitted"]

        # Train model
        model.train(X, y)
        original_predictions = model.predict(X)

        # Save model
        save_path = temp_dir / f"{config_key}_model"
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
    def test_model_feature_importance(self, model_class, config_key, test_data, test_config):
        """Test feature importance calculation."""
        model = model_class(test_config.model[config_key])
        X = test_data.drop("readmitted", axis=1)
        y = test_data["readmitted"]

        # Train model
        model.train(X, y)

        # Get feature importance
        importance = model.feature_importance(X)

        # Verify feature importance
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >= 0 for v in importance.values())

    @pytest.mark.parametrize("model_class,config_key", [
        (LogisticRegressionModel, "logistic"),
        (NeuralNetworkModel, "neural"),
        (AutoGluonModel, "autogluon")
    ])
    def test_model_validation(self, model_class, config_key, test_data, test_config):
        """Test model validation functionality."""
        model = model_class(test_config.model[config_key])
        X = test_data.drop("readmitted", axis=1)
        y = test_data["readmitted"]

        # Split data for validation
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train with validation
        model.train(X_train, y_train, X_val, y_val)

        # Check metrics
        assert model.metrics is not None
        assert "accuracy" in model.metrics
        assert "roc_auc" in model.metrics
        assert all(0 <= v <= 1 for v in model.metrics.values())

    @pytest.mark.parametrize("model_class,config_key", [
        (LogisticRegressionModel, "logistic"),
        (NeuralNetworkModel, "neural"),
        (AutoGluonModel, "autogluon")
    ])
    def test_model_error_handling(self, model_class, config_key, test_data, test_config):
        """Test model error handling."""
        model = model_class(test_config.model[config_key])
        X = test_data.drop("readmitted", axis=1)
        y = test_data["readmitted"]

        # Test prediction before training
        with pytest.raises(RuntimeError):
            model.predict(X)

        # Test invalid input shapes
        model.train(X, y)
        with pytest.raises(ValueError):
            model.predict(X.iloc[:, :2])

        # Test invalid input types
        with pytest.raises(TypeError):
            model.predict([1, 2, 3])

    @pytest.mark.parametrize("model_class,config_key", [
        (LogisticRegressionModel, "logistic"),
        (NeuralNetworkModel, "neural"),
        (AutoGluonModel, "autogluon")
    ])
    def test_model_reproducibility(self, model_class, config_key, test_data, test_config):
        """Test model reproducibility with same seed."""
        X = test_data.drop("readmitted", axis=1)
        y = test_data["readmitted"]

        # Train two models with same seed
        model1 = model_class(test_config.model[config_key])
        model2 = model_class(test_config.model[config_key])

        model1.train(X, y)
        model2.train(X, y)

        # Compare predictions
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        np.testing.assert_array_equal(pred1, pred2)
