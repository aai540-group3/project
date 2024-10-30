import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.models.base import Model
from pipeline.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)

app = FastAPI(title="Diabetes Readmission Prediction")


class PredictionRequest(BaseModel):
    """Prediction request model."""

    features: Dict[str, float]


class PredictionResponse(BaseModel):
    """Prediction response model."""

    prediction: int
    probability: float
    feature_importance: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
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
    """Make prediction."""
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
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
