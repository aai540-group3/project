# src/serve/app.py
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models import AutoGluonModel
from src.monitoring import MetricsMonitor, PerformanceMonitor
from src.utils.registry import ModelRegistry

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description="API for predicting hospital readmissions for diabetic patients",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    """Prediction request model."""
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction: int
    probability: float
    feature_importance: Dict[str, float]
    prediction_id: str

# Initialize components
model_registry = ModelRegistry()
metrics_monitor = MetricsMonitor()
performance_monitor = PerformanceMonitor()

@app.on_event("startup")
async def startup_event():
    """Load model and initialize components on startup."""
    global model
    try:
        # Load production model
        model = model_registry.get_production_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_version": model_registry.get_current_version(),
        "model_timestamp": model_registry.get_model_timestamp()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction."""
    try:
        start_time = time.time()

        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])

        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0, 1]
        importance = model.feature_importance(features_df)

        # Generate prediction ID
        prediction_id = str(uuid.uuid4())

        # Record metrics
        latency = (time.time() - start_time) * 1000  # Convert to ms
        performance_monitor.record_prediction(latency=latency)

        # Log prediction
        logger.info(
            f"Prediction made: ID={prediction_id}, "
            f"prediction={prediction}, probability={probability:.3f}"
        )

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            feature_importance=importance,
            prediction_id=prediction_id
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        performance_monitor.record_prediction(latency=0, error=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def record_feedback(
    prediction_id: str,
    actual_outcome: int,
    feedback_type: str = "ground_truth"
):
    """Record prediction feedback."""
    try:
        metrics_monitor.record_feedback(
            prediction_id=prediction_id,
            actual_outcome=actual_outcome,
            feedback_type=feedback_type
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error recording feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get model metrics."""
    return {
        "performance": performance_monitor.get_metrics(),
        "model": metrics_monitor.get_metrics()
    }
