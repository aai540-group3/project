"""
Model Registry
==============

.. module:: pipeline.utils.registry
   :synopsis: Model registry and versioning

.. moduleauthor:: aai540-group3
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import mlflow
from omegaconf import DictConfig

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Model registry and versioning system."""

    def __init__(self, cfg: DictConfig):
        """Initialize model registry.

        :param cfg: Registry configuration
        :type cfg: DictConfig
        """
        self.cfg = cfg
        self.metadata_path = Path(cfg.registry.metadata.save_path)
        self.metadata = self._load_metadata()

        if cfg.registry.type == "mlflow":
            mlflow.set_registry_uri(cfg.registry.uri)

    def _load_metadata(self) -> Dict:
        """Load registry metadata.

        :return: Registry metadata
        :rtype: Dict
        """
        if self.metadata_path.exists():
            with self.metadata_path.open("r") as f:
                return json.load(f)
        return {"models": [], "production_model": None}

    def register_model(
        self,
        model_path: Union[str, Path],
        name: str,
        metrics: Dict,
        tags: Optional[Dict] = None,
    ) -> str:
        """Register model in registry.

        :param model_path: Path to model
        :type model_path: Union[str, Path]
        :param name: Model name
        :type name: str
        :param metrics: Model metrics
        :type metrics: Dict
        :param tags: Model tags
        :type tags: Optional[Dict]
        :return: Model version
        :rtype: str
        """
        try:
            # Register with MLflow if enabled
            if self.cfg.registry.type == "mlflow":
                model_uri = f"models:/{name}"
                mlflow.register_model(model_uri, name)

            # Update local metadata
            version = self._get_next_version(name)
            model_info = {
                "name": name,
                "version": version,
                "path": str(model_path),
                "metrics": metrics,
                "tags": tags or {},
                "creation_date": datetime.now().isoformat(),
                "status": "registered",
            }

            self.metadata["models"].append(model_info)
            self._save_metadata()

            logger.info(f"Registered model {name} version {version}")
            return version

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def transition_model(self, name: str, version: str, stage: str) -> None:
        """Transition model to new stage.

        :param name: Model name
        :type name: str
        :param version: Model version
        :type version: str
        :param stage: Target stage
        :type stage: str
        """
        try:
            # Update MLflow if enabled
            if self.cfg.registry.type == "mlflow":
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=name, version=version, stage=stage
                )

            # Update local metadata
            for model in self.metadata["models"]:
                if model["name"] == name and model["version"] == version:
                    model["status"] = stage
                    if stage == "production":
                        self.metadata["production_model"] = {
                            "name": name,
                            "version": version,
                            "deployment_date": datetime.now().isoformat(),
                        }

            self._save_metadata()
            logger.info(f"Transitioned model {name} version {version} to {stage}")

        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            raise

    def get_model_info(
        self, name: str, version: Optional[str] = None
    ) -> Optional[Dict]:
        """Get model information.

        :param name: Model name
        :type name: str
        :param version: Model version
        :type version: Optional[str]
        :return: Model information
        :rtype: Optional[Dict]
        """
        for model in self.metadata["models"]:
            if model["name"] == name:
                if version is None or model["version"] == version:
                    return model
        return None

    def get_production_model(self) -> Optional[Dict]:
        """Get current production model.

        :return: Production model information
        :rtype: Optional[Dict]
        """
        return self.metadata.get("production_model")

    def _get_next_version(self, name: str) -> str:
        """Get next version number.

        :param name: Model name
        :type name: str
        :return: Next version
        :rtype: str
        """
        versions = [
            int(model["version"])
            for model in self.metadata["models"]
            if model["name"] == name
        ]
        return str(max(versions + [0]) + 1)

    def _save_metadata(self) -> None:
        """Save registry metadata."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w") as f:
            json.dump(self.metadata, f, indent=2)

    def list_models(
        self, name: Optional[str] = None, stage: Optional[str] = None
    ) -> List[Dict]:
        """List registered models.

        :param name: Filter by name
        :type name: Optional[str]
        :param stage: Filter by stage
        :type stage: Optional[str]
        :return: List of models
        :rtype: List[Dict]
        """
        models = self.metadata["models"]

        if name:
            models = [m for m in models if m["name"] == name]
        if stage:
            models = [m for m in models if m["status"] == stage]

        return models

    def delete_model(self, name: str, version: Optional[str] = None) -> None:
        """Delete model from registry.

        :param name: Model name
        :type name: str
        :param version: Model version
        :type version: Optional[str]
        """
        try:
            # Delete from MLflow if enabled
            if self.cfg.registry.type == "mlflow":
                client = mlflow.tracking.MlflowClient()
                if version:
                    client.delete_model_version(name=name, version=version)
                else:
                    client.delete_registered_model(name=name)

            # Update local metadata
            if version:
                self.metadata["models"] = [
                    m
                    for m in self.metadata["models"]
                    if not (m["name"] == name and m["version"] == version)
                ]
            else:
                self.metadata["models"] = [
                    m for m in self.metadata["models"] if m["name"] != name
                ]

            # Update production model if needed
            prod_model = self.metadata.get("production_model")
            if prod_model and prod_model["name"] == name:
                if not version or prod_model["version"] == version:
                    self.metadata["production_model"] = None

            self._save_metadata()
            logger.info(
                f"Deleted model {name}" + (f" version {version}" if version else "")
            )

        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            raise
