import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

from huggingface_hub import HfApi, ModelCard
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class HuggingFaceDeployer:
    """Deploy models to HuggingFace Hub."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.api = HfApi(token=cfg.huggingface.token)
        self.repo_id = cfg.huggingface.repo_id

    def prepare_model_card(self, metrics: Dict[str, float], feature_info: Dict[str, List[str]]) -> ModelCard:
        """Create model card with metadata."""
        card_data = {
            "language": "en",
            "license": "mit",
            "library_name": "scikit-learn",
            "tags": ["tabular-classification", "diabetes", "healthcare"],
            "metrics": [{"type": metric, "value": value, "name": metric} for metric, value in metrics.items()],
        }

        # Create detailed model card content
        content = f"""
# Diabetes Readmission Prediction

## Model Description
This model predicts hospital readmission risk for diabetic patients.

## Intended Use
Healthcare professionals can use this model to identify patients at high risk of readmission.

## Training Data
The model was trained on the Diabetes 130-US hospitals dataset.

## Performance Metrics
{self._format_metrics(metrics)}

## Features
{self._format_features(feature_info)}

## Limitations
- Model trained on historical data (1999-2008)
- May not generalize to significantly different healthcare systems
- Should be used as a supportive tool, not sole decision maker

## Ethical Considerations
- Model should be regularly audited for bias
- Patient privacy must be protected
- Healthcare professionals should maintain decision authority

## Technical Specifications
- Framework: scikit-learn
- Input: Tabular data
- Output: Binary classification (readmission risk)
"""

        return ModelCard(content)

    def prepare_deployment(
        self,
        model_path: Path,
        output_dir: Path,
        metrics: Dict[str, float],
        feature_info: Dict[str, List[str]],
    ) -> None:
        """Prepare model for deployment."""
        # Create deployment directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        shutil.copy2(model_path, output_dir / "model.pkl")

        # Create model card
        card = self.prepare_model_card(metrics, feature_info)
        card.save(output_dir / "README.md")

        # Create configuration files
        self._create_config_files(output_dir, feature_info)

    def push_to_hub(self, local_dir: Path, commit_message: str) -> None:
        """Push model to HuggingFace Hub."""
        self.api.upload_folder(
            repo_id=self.repo_id,
            folder_path=str(local_dir),
            commit_message=commit_message,
        )
        logger.info(f"Model pushed to HuggingFace Hub: {self.repo_id}")

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for model card."""
        return "\n".join([f"- {metric}: {value:.4f}" for metric, value in metrics.items()])

    def _format_features(self, feature_info: Dict[str, List[str]]) -> str:
        """Format feature information for model card."""
        sections = []
        for group, features in feature_info.items():
            features_str = "\n".join([f"  - {feat}" for feat in features])
            sections.append(f"### {group}\n{features_str}")
        return "\n\n".join(sections)

    def _create_config_files(self, output_dir: Path, feature_info: Dict[str, List[str]]) -> None:
        """Create necessary configuration files."""
        # Create config.json
        config = {
            "architectures": ["TabularBinaryClassification"],
            "model_type": "scikit-learn",
            "num_labels": 2,
            "id2label": {"0": "NO_READMISSION", "1": "READMISSION"},
            "label2id": {"NO_READMISSION": 0, "READMISSION": 1},
            "feature_groups": feature_info,
        }

        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
