"""
Diabetes Readmission Prediction API
===================================

This module defines a FastAPI application for predicting hospital readmission for diabetes patients.
The API provides endpoints for making predictions, checking model health, and returning feature importance.

.. module:: pipeline.deploy.app
   :synopsis: FastAPI application for diabetes readmission prediction.

.. moduleauthor:: aai540-group3
"""

import logging
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)

app = FastAPI(title="Diabetes Readmission Prediction")


class PredictionRequest(BaseModel):
    """Schema for prediction request.

    :param features: Dictionary of features needed for prediction
    :type features: Dict[str, float]
    """

    features: Dict[str, float]


class PredictionResponse(BaseModel):
    """Schema for prediction response.

    :param prediction: Predicted label (0 or 1)
    :type prediction: int
    :param probability: Probability of readmission
    :type probability: float
    :param feature_importance: Dictionary of feature importances
    :type feature_importance: Dict[str, float]
    """

    prediction: int
    probability: float
    feature_importance: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Load the production model on startup.

    Initializes the model registry and retrieves the production model.
    Logs an error if model loading fails.
    """
    global model, registry

    try:
        # Initialize model registry
        registry = ModelRegistry()

        # Load production model
        model = registry.get_production_model()
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint to make predictions.

    Takes in feature data and returns a prediction, probability, and feature importance.

    :param request: JSON body containing feature data
    :type request: PredictionRequest
    :return: PredictionResponse with prediction, probability, and feature importance
    :rtype: PredictionResponse
    :raises HTTPException: 500 error if prediction fails
    """
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0, 1]

        # Get feature importance
        importance = model.feature_importance(features_df)

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            feature_importance=importance,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint to ensure API is running."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
