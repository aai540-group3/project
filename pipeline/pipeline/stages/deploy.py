"""
Deploy Stage
============

.. module:: pipeline.stages.deploy
   :synopsis: Model deployment to HuggingFace Hub

.. moduleauthor:: aai540-group3
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from huggingface_hub import HfApi, ModelCard, ModelCardData
from loguru import logger
from omegaconf import DictConfig

from .stage import Stage


class Deploy(Stage):
    """Deployment stage for model artifacts.

    Handles model deployment to HuggingFace Hub, including:
    - Model artifact organization
    - Model card generation
    - Configuration creation
    - HuggingFace upload

    :param cfg: Deployment configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def __init__(self, cfg: DictConfig):
        """Initialize deployment stage.

        :param cfg: Deployment configuration
        :type cfg: DictConfig
        """
        super().__init__(cfg)
        self.model_dirs = self._get_model_directories()
        self.output_dir = Path(self.cfg.paths.deploy) / "huggingface"
        self.repo_id = self.cfg.huggingface.repo_id

    def _get_model_directories(self) -> Dict:
        """Get model directory configurations.

        :return: Dictionary of model paths
        :rtype: Dict
        """
        return {
            "logisticregression": {
                "root": Path(self.cfg.paths.models) / "logisticregression/artifacts",
                "model": Path(self.cfg.paths.models) / "logisticregression/artifacts/model/model.joblib",
                "metrics": Path(self.cfg.paths.models) / "logisticregression/artifacts/metrics/metrics.json",
                "feature_importance": Path(self.cfg.paths.models)
                / "logisticregression/artifacts/metrics/feature_importance.csv",
                "plots": Path(self.cfg.paths.models) / "logisticregression/artifacts/plots",
                "scaler": Path(self.cfg.paths.models) / "logisticregression/artifacts/model/scaler.joblib",
            },
        }

    def run(self):
        """Execute deployment pipeline.

        :raises RuntimeError: If deployment fails
        """
        logger.info("Starting model deployment")

        try:
            # Copy model artifacts
            if not self._copy_model_artifacts():
                raise RuntimeError("Failed to copy model artifacts")

            # Find best model
            best_model, best_metrics = self._find_best_model()
            if not best_model or not best_metrics:
                raise RuntimeError("Could not determine best model")

            # Create necessary files
            self._create_model_card(best_model, best_metrics)
            self._create_preprocessing_config(best_model)
            self._create_model_config(best_model)
            self._create_tokenizer_config()

            # Upload to HuggingFace
            self._upload_to_huggingface()

            logger.info("Deployment completed successfully")

        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise RuntimeError(f"Deployment failed: {str(e)}")

    def _copy_model_artifacts(self) -> bool:
        """Copy model artifacts to deployment directory.

        :return: Success status
        :rtype: bool
        :raises IOError: If copying fails
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            for model_type, paths in self.model_dirs.items():
                if paths["root"].exists():
                    dest_dir = self.output_dir / model_type
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    # Copy model files
                    self._copy_model_files(paths, dest_dir)
                    # Copy metrics and plots
                    self._copy_metrics_and_plots(paths, dest_dir)

                    logger.info(f"Copied {model_type} artifacts to {dest_dir}")

            return True

        except Exception as e:
            logger.error(f"Error copying model artifacts: {e}")
            return False

    def _find_best_model(self) -> Tuple[Optional[str], Optional[Dict]]:
        """Find the best performing model.

        :return: Best model name and metrics
        :rtype: Tuple[Optional[str], Optional[Dict]]
        """
        best_model = None
        best_metrics = None

        for model_type, paths in self.model_dirs.items():
            metrics = self._load_metrics(paths["metrics"])
            if metrics and (not best_metrics or metrics["test_auc"] > best_metrics["test_auc"]):
                best_metrics = metrics
                best_model = model_type

        if best_model:
            logger.info(f"Best model is {best_model} with AUC: {best_metrics['test_auc']:.4f}")
        else:
            logger.error("No valid model metrics found")

        return best_model, best_metrics

    # pipeline/pipeline/stages/deploy.py (continued)

    def _create_model_card(self, best_model: str, metrics: Dict):
        """Create model card with metrics and documentation.

        :param best_model: Name of best model
        :type best_model: str
        :param metrics: Model metrics
        :type metrics: Dict
        :raises IOError: If model card creation fails
        """
        try:
            # Create evaluation results
            eval_results = [
                {
                    "task_type": "binary-classification",
                    "dataset_type": "hospital-readmission",
                    "dataset_name": "Diabetes 130-US Hospitals",
                    "metric_type": metric_type,
                    "metric_value": metrics.get(f"test_{metric_type}", "N/A"),
                    "metric_name": metric_type,
                }
                for metric_type in ["accuracy", "auc"]
            ]

            # Load feature importance if available
            feature_importance_df = self._load_feature_importance(best_model)

            # Create card data
            card_data = ModelCardData(
                language="en",
                license="mit",
                model_name=self.repo_id,
                eval_results=eval_results,
                library_name="transformers",
            )

            # Generate card content
            content = self._generate_model_card_content(best_model, metrics, card_data, feature_importance_df)

            # Save model card
            card = ModelCard(content)
            card.save(self.output_dir / "README.md")
            logger.info(f"Created model card at {self.output_dir / 'README.md'}")

        except Exception as e:
            logger.error(f"Failed to create model card: {e}")
            raise IOError(f"Model card creation failed: {e}")

    def _load_feature_importance(self, model_type: str) -> Optional[pd.DataFrame]:
        """Load feature importance data.

        :param model_type: Type of model
        :type model_type: str
        :return: Feature importance DataFrame
        :rtype: Optional[pd.DataFrame]
        """
        if "feature_importance" not in self.model_dirs[model_type]:
            return None

        try:
            fi_path = self.model_dirs[model_type]["feature_importance"]
            if fi_path.exists():
                df = pd.read_csv(fi_path)
                return df[df["importance"] > 0].sort_values("importance", ascending=False)
        except Exception as e:
            logger.warning(f"Could not load feature importance: {e}")
        return None

    def _generate_model_card_content(
        self,
        model_type: str,
        metrics: Dict,
        card_data: ModelCardData,
        feature_importance_df: Optional[pd.DataFrame],
    ) -> str:
        """Generate model card content.

        :param model_type: Type of model
        :type model_type: str
        :param metrics: Model metrics
        :type metrics: Dict
        :param card_data: Model card metadata
        :type card_data: ModelCardData
        :param feature_importance_df: Feature importance data
        :type feature_importance_df: Optional[pd.DataFrame]
        :return: Model card content
        :rtype: str
        """
        content = f"""
---
pipeline_tag: tabular-classification
{card_data.to_yaml()}
---

# {self.repo_id}

## Model Description

This model predicts 30-day hospital readmissions for diabetic patients using historical patient data
and machine learning techniques. The model aims to identify high-risk individuals enabling targeted
interventions and improved healthcare resource allocation.

## Performance Metrics

- **Test Accuracy:** {metrics.get('test_accuracy', 'N/A'):.4f}
- **Test ROC-AUC:** {metrics.get('test_auc', 'N/A'):.4f}
"""

        # Add feature importance section if available
        if feature_importance_df is not None and not feature_importance_df.empty:
            content += self._format_feature_importance(feature_importance_df)

        # Add remaining sections
        content += self._get_model_card_sections()

        return content

    def _create_preprocessing_config(self, model_type: str):
        """Create preprocessing configuration.

        :param model_type: Type of model
        :type model_type: str
        :raises IOError: If config creation fails
        """
        try:
            config = {
                "preprocessor": {
                    "numeric_features": self.cfg.features.numeric_features,
                    "binary_features": self.cfg.features.binary_features,
                    "categorical_features": self.cfg.features.categorical_features,
                    "interaction_features": self.cfg.features.interaction_features,
                    "ratio_features": self.cfg.features.ratio_features,
                },
                "transformations": {
                    "numeric_scaling": "standard",
                    "outlier_handling": {"method": "clip", "std_multiplier": 5},
                    "missing_values": {"numeric": "mean", "categorical": "mode"},
                },
                "target": {
                    "name": "readmitted",
                    "type": "binary",
                    "mapping": {">30": 0, "<30": 1, "NO": 0},
                },
            }

            config_path = self.output_dir / "preprocessing_config.json"
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created preprocessing config at {config_path}")

        except Exception as e:
            logger.error(f"Failed to create preprocessing config: {e}")
            raise IOError(f"Preprocessing config creation failed: {e}")

    def _create_model_config(self, model_type: str):
        """Create model configuration.

        :param model_type: Type of model
        :type model_type: str
        :raises IOError: If config creation fails
        """
        try:
            config = {
                "architectures": ["TabularBinaryClassification"],
                "model_type": model_type,
                "num_classes": 2,
                "id2label": {"0": "NO_READMISSION", "1": "READMISSION"},
                "label2id": {"NO_READMISSION": 0, "READMISSION": 1},
                "task_specific_params": {"classification": {"problem_type": "binary_classification"}},
                "preprocessing": {"featurization_config": "preprocessing_config.json"},
            }

            config_path = self.output_dir / "config.json"
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created model config at {config_path}")

        except Exception as e:
            logger.error(f"Failed to create model config: {e}")
            raise IOError(f"Model config creation failed: {e}")

    def _upload_to_huggingface(self):
        """Upload model to HuggingFace Hub.

        :raises RuntimeError: If upload fails
        """
        try:
            token = self._get_hf_token()
            os.environ["HF_TRANSFER"] = "1"  # Enable faster uploads

            # Upload folder
            logger.info(f"Starting upload to {self.repo_id}...")
            api = HfApi(token=token)
            api.create_repo(
                repo_id=self.repo_id,
                exist_ok=True,
                private=self.cfg.huggingface.private,
            )

            api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(self.output_dir),
                commit_message="Update model artifacts",
            )

            logger.info(f"Successfully uploaded model to {self.repo_id}")

        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace: {e}")
            raise RuntimeError(f"HuggingFace upload failed: {e}")

    def _get_hf_token(self) -> str:
        """Get HuggingFace token from configuration.

        :return: HuggingFace token
        :rtype: str
        :raises EnvironmentError: If token is not available
        """
        token = self.cfg.huggingface.token
        if not token:
            raise EnvironmentError("HuggingFace token not found in configuration")
        return token
