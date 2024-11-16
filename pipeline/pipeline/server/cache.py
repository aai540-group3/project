import json
from typing import Any, Dict, Optional

import redis


class PredictionCache:
    """Cache for model predictions."""

    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600):
        """Initialize cache.

        Args:
            host: Redis host
            port: Redis port
            ttl: Time to live in seconds
        """
        self.client = redis.Redis(host=host, port=port)
        self.ttl = ttl

    def get_prediction(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Get cached prediction.

        Args:
            features: Input features

        Returns:
            Cached prediction if exists
        """
        key = self._generate_key(features)
        cached = self.client.get(key)
        return json.loads(cached) if cached else None

    def set_prediction(self, features: Dict[str, float], prediction: Dict[str, Any]) -> None:
        """Cache prediction.

        Args:
            features: Input features
            prediction: Prediction to cache
        """
        key = self._generate_key(features)
        self.client.setex(key, self.ttl, json.dumps(prediction))

    def _generate_key(self, features: Dict[str, float]) -> str:
        """Generate cache key from features.

        Args:
            features: Input features

        Returns:
            Cache key
        """
        # Sort features to ensure consistent keys
        sorted_features = sorted(features.items())
        return f"pred:{hash(str(sorted_features))}"
