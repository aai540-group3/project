# src/utils/registry.py
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Manage model versions and deployment."""

    def __init__(self, cfg: DictConfig):
        """Initialize model registry.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.client = MlflowClient()

    def register_model(
        self,
        model: Any,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register model in MLflow.

        Args:
            model: Model to register
            name: Model name
            version: Model version
            stage: Model stage
            description: Model description
            tags: Model tags

        Returns:
            Model version
        """
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=name
        )

        # Get latest version
        versions = self.client.search_model_versions(f"name='{name}'")
        latest_version = versions[0].version if versions else "1"

        # Set version if not provided
        if not version:
            version = str(int(latest_version) + 1)

        # Set stage if provided
        if stage:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage
            )

        # Set description if provided
        if description:
            self.client.update_model_version(
                name=name,
                version=version,
                description=description
            )

        # Set tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=name,
                    version=version,
                    key=key,
                    value=value
                )

        return version

    def load_model(
        self,
        name: str,
        version: Optional[str] = "latest",
        stage: Optional[str] = None
    ) -> Any:
        """Load model from registry.

        Args:
            name: Model name
            version: Model version
            stage: Model stage

        Returns:
            Loaded model
        """
        if stage:
            uri = f"models:/{name}/{stage}"
        elif version == "latest":
            versions = self.client.search_model_versions(f"name='{name}'")
            if not versions:
                raise ValueError(f"No versions found for model {name}")
            uri = f"models:/{name}/{versions[0].version}"
        else:
            uri = f"models:/{name}/{version}"

        return mlflow.sklearn.load_model(uri)

    def get_best_model(
        self,
        name: str,
        metric: str,
        ascending: bool = False
    ) -> Any:
        """Get best model version based on metric.

        Args:
            name: Model name
            metric: Metric to use for comparison
            ascending: Whether to sort ascending

        Returns:
            Best model
        """
        versions = self.client.search_model_versions(f"name='{name}'")
        if not versions:
            raise ValueError(f"No versions found for model {name}")

        # Sort versions by metric
        sorted_versions = sorted(
            versions,
            key=lambda x: float(x.metrics.get(metric, float("-inf"))),
            reverse=not ascending
        )

        return self.load_model(name, sorted_versions[0].version)

    def delete_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> None:
        """Delete model from registry.

        Args:
            name: Model name
            version: Model version to delete (if None, deletes all versions)
        """
        if version:
            self.client.delete_model_version(name=name, version=version)
        else:
            self.client.delete_registered_model(name=name)

    def save_metadata(self, path: Path) -> None:
        """Save registry metadata.

        Args:
            path: Path to save metadata
        """
        # Get all registered models
        models = self.client.search_registered_models()

        metadata = {
            "models": [
                {
                    "name": model.name,
                    "latest_version": model.latest_versions[0].version,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                }
                for model in models
            ]
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
