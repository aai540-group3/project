#!/bin/bash

# Create necessary directories
mkdir -p src/{utils,models,stages} requirements

# Create __init__.py files
touch __init__.py src/__init__.py src/utils/__init__.py src/models/__init__.py src/stages/__init__.py

# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "omegaconf",
        "boto3",
    ],
)
EOF

# Create/update src/stages/base.py
cat > src/stages/base.py << 'EOF'
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..utils.experiment import ExperimentTracker
from ..utils.registry import ModelRegistry

class PipelineStage(ABC):
    """Base class for pipeline stages."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize pipeline stage."""
        self.cfg = cfg
        self.tracker = ExperimentTracker(cfg, cfg.experiment.name)
        self.registry = ModelRegistry(cfg)

    @abstractmethod
    def run(self) -> None:
        """Run pipeline stage."""
        pass
EOF

# Create/update src/stages/infrastruct.py
cat > src/stages/infrastruct.py << 'EOF'
import logging
from pathlib import Path
import boto3
from .base import PipelineStage

logger = logging.getLogger(__name__)

class InfrastructureStage(PipelineStage):
    """Infrastructure setup stage."""

    def run(self) -> None:
        """Execute infrastructure setup."""
        self.tracker.start_run(run_name="infrastructure")

        try:
            # Initialize AWS session
            session = boto3.Session(
                region_name=self.cfg.aws.region,
                profile_name=self.cfg.aws.profile
            )

            # Setup IAM
            self._setup_iam(session)

            # Setup S3
            self._setup_s3(session)

            # Setup DynamoDB
            self._setup_dynamodb(session)

            # Setup monitoring
            self._setup_monitoring(session)

            # Save infrastructure metadata
            self._save_metadata()

            logger.info("Infrastructure setup completed successfully")

        finally:
            self.tracker.end_run()
EOF

# Create/update dvc.yaml
cat > dvc.yaml << 'EOF'
stages:
  infrastruct:
    cmd: >
      uv venv .venv-infrastruct &&
      uv pip install -r requirements/requirements-infrastruct.txt --python .venv-infrastruct/bin/python &&
      uv pip install -e . --python .venv-infrastruct/bin/python &&
      .venv-infrastruct/bin/python src/stages/infrastruct.py
    deps:
      - src/stages/infrastruct.py
      - conf/infrastructure/aws.yaml
      - requirements/infrastruct.txt
    params:
      - aws
      - paths
    outs:
      - infrastructure/metadata.json:
          cache: false
    metrics:
      - metrics/infrastructure/resources.json:
          cache: false

  ingest:
    cmd: >
      uv venv .venv-ingest && \
      uv pip install -r requirements/requirements-ingest.txt --python .venv-ingest/bin/python && \
      uv pip install -e . --python .venv-ingest/bin/python && \
      .venv-ingest/bin/python src/stages/ingest.py
    deps:
      - src/stages/ingest.py
      - conf/data/sources.yaml
      - requirements/ingest.txt
    params:
      - data.sources
      - paths
    outs:
      - data/raw/data.parquet
      - data/raw/data.csv
      - data/raw/metadata.json
      - data/raw/variables.json

  prepare:
    cmd: >
      uv venv .venv-preprocess && \
      uv pip install -r requirements/requirements-preprocess.txt --python .venv-preprocess/bin/python && \
      uv pip install -e . --python .venv-preprocess/bin/python && \
      .venv-preprocess/bin/python src/stages/prepare.py
    deps:
      - src/stages/prepare.py
      - data/raw/data.csv
      - conf/data/preprocessing.yaml
      - requirements/preprocess.txt
    params:
      - data.preprocessing
      - data.splits
      - paths
    outs:
      - data/interim/train.parquet
      - data/interim/val.parquet
      - data/interim/test.parquet
    metrics:
      - metrics/data/data_stats.json:
          cache: false
    plots:
      - metrics/plots/data_distribution.png

  explore:
    cmd: >
      uv venv .venv-explore && \
      uv pip install -r requirements/requirements-explore.txt --python .venv-explore/bin/python && \
      uv pip install -e . --python .venv-explore/bin/python && \
      .venv-explore/bin/python src/stages/explore.py
    deps:
      - src/stages/explore.py
      - data/raw/data.csv
      - data/interim/data_cleaned.parquet
      - conf/explore/analysis.yaml
      - requirements/explore.txt
    params:
      - explore
      - paths
    metrics:
      - metrics/explore/data_quality.json:
          cache: false
    plots:
      - metrics/plots/explore/feature_distributions.png
      - metrics/plots/explore/correlation_matrix.png
      - metrics/plots/explore/target_distribution.png

  featurize:
    cmd: >
      uv venv .venv-featurize && \
      uv pip install -r requirements/requirements-featurize.txt --python .venv-featurize/bin/python && \
      uv pip install -e . --python .venv-featurize/bin/python && \
      .venv-featurize/bin/python src/stages/featurize.py
    deps:
      - src/stages/featurize.py
      - data/interim/train.parquet
      - data/interim/val.parquet
      - data/interim/test.parquet
      - conf/features/engineering.yaml
      - requirements/featurize.txt
    params:
      - features
      - paths
    outs:
      - data/processed/train_features.parquet
      - data/processed/val_features.parquet
      - data/processed/test_features.parquet
      - models/preprocessor.joblib
    metrics:
      - metrics/features/feature_stats.json:
          cache: false
    plots:
      - metrics/plots/feature_distributions.png
      - metrics/plots/correlation_matrix.png

  optimize:
    foreach: ${model_types}
    do:
      cmd: >
        uv venv .venv-optimize && \
        uv pip install -r requirements/requirements-optimize.txt --python .venv-optimize/bin/python && \
        uv pip install -e . --python .venv-optimize/bin/python && \
        .venv-optimize/bin/python src/stages/optimize.py model=${item}
      deps:
        - src/stages/optimize.py
        - src/models/${item}.py
        - data/processed/train_features.parquet
        - data/processed/val_features.parquet
        - requirements/optimize.txt
      params:
        - model.${item}
        - optimization
        - paths
      metrics:
        - metrics/optimization/${item}/trials.json:
            cache: false
      plots:
        - metrics/plots/optimization/${item}/parameter_importance.png
        - metrics/plots/optimization/${item}/optimization_history.png

  train:
    foreach: ${model_types}
    do:
      cmd: >
        uv venv .venv-${item} && \
        uv pip install -r requirements/requirements-${item}.txt --python .venv-${item}/bin/python && \
        uv pip install -e . --python .venv-${item}/bin/python && \
        .venv-${item}/bin/python src/stages/train.py model=${item}
      deps:
        - src/stages/train.py
        - src/models/${item}.py
        - data/processed/train_features.parquet
        - data/processed/val_features.parquet
        - metrics/optimization/${item}/trials.json
        - requirements/${item}.txt
      params:
        - model.${item}
        - training
        - paths
      outs:
        - models/${item}/model.pkl
      metrics:
        - metrics/training/${item}/metrics.json:
            cache: false
      plots:
        - metrics/plots/training/${item}/loss.png
        - metrics/plots/training/${item}/confusion_matrix.png
        - metrics/plots/training/${item}/roc_curve.png

  evaluate:
    cmd: >
      uv venv .venv-evaluate && \
      uv pip install -r requirements/requirements-evaluate.txt --python .venv-evaluate/bin/python && \
      uv pip install -e . --python .venv-evaluate/bin/python && \
      .venv-evaluate/bin/python src/stages/evaluate.py
    deps:
      - src/stages/evaluate.py
      - data/processed/test_features.parquet
      - models/*/model.pkl
      - requirements/evaluate.txt
    params:
      - evaluation
      - paths
    metrics:
      - metrics/evaluation/model_comparison.json:
          cache: false
    plots:
      - metrics/plots/evaluation/model_comparison.png
      - metrics/plots/evaluation/feature_importance.png

  register:
    cmd: >
      uv venv .venv-register && \
      uv pip install -r requirements/requirements-register.txt --python .venv-register/bin/python && \
      uv pip install -e . --python .venv-register/bin/python && \
      .venv-register/bin/python src/stages/register.py
    deps:
      - src/stages/register.py
      - metrics/evaluation/model_comparison.json
      - models/*/model.pkl
      - requirements/register.txt
    params:
      - registry
      - paths
    outs:
      - registry/metadata.json

  deploy:
    cmd: >
      uv venv .venv-deploy && \
      uv pip install -r requirements/requirements-deploy.txt --python .venv-deploy/bin/python && \
      uv pip install -e . --python .venv-deploy/bin/python && \
      .venv-deploy/bin/python src/stages/deploy.py
    deps:
      - src/stages/deploy.py
      - models/*/model.pkl
      - metrics/evaluation/model_comparison.json
      - requirements/deploy.txt
    params:
      - huggingface
      - paths
    outs:
      - deploy/huggingface:
          persist: true

  serve:
    cmd: >
      uv venv .venv-serve && \
      uv pip install -r requirements/requirements-serve.txt --python .venv-serve/bin/python && \
      uv pip install -e . --python .venv-serve/bin/python && \
      .venv-serve/bin/uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
    deps:
      - src/serve/app.py
      - models/*/model.pkl
      - conf/serve/api.yaml
      - requirements/serve.txt
    params:
      - serve
      - paths
    outs:
      - serve/logs:
          persist: true

  monitor:
    cmd: >
      uv venv .venv-monitoring && \
      uv pip install -r requirements/requirements-monitoring.txt --python .venv-monitoring/bin/python && \
      uv pip install -e . --python .venv-monitoring/bin/python && \
      .venv-monitoring/bin/python src/stages/monitoring.py
    deps:
      - src/monitoring/app.py
      - conf/monitoring/config.yaml
      - requirements/monitoring.txt
    params:
      - monitoring
      - paths
    metrics:
      - metrics/monitoring/system_metrics.json:
          cache: false
    plots:
      - metrics/plots/monitoring/resource_usage.png
      - metrics/plots/monitoring/prediction_latency.png
      - metrics/plots/monitoring/error_rate.png

metrics:
  - metrics/evaluation/model_comparison.json

plots:
  - metrics/plots/evaluation/model_comparison.png

vars:
  - model_types:
    - logistic
    - neural
    - autogluon

params:
  - params.yaml
EOF

# Make the script executable
chmod +x update.sh

echo "Files have been created/updated successfully!"