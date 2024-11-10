"""
Configuration Schema
===================

.. module:: pipeline.conf.schema
   :synopsis: Configuration schema definitions using dataclasses

.. moduleauthor:: aai540-group3
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING


@dataclass
class StageConfig:
    """Base configuration for pipeline stages."""

    enabled: bool = True
    requires_data: bool = True
    save_data: bool = True
    generate_visualizations: bool = True
    monitor_system: bool = False
    dependencies: List[str] = field(default_factory=list)
    params: List[str] = field(default_factory=list)


@dataclass
class ResourceConfig:
    """Configuration for resource management."""

    max_memory: Optional[str] = None
    max_cpu: Optional[float] = None
    timeout: Optional[int] = None
    retry_count: int = 3
    cleanup: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    rotation: str = "1 day"
    retention: str = "7 days"


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking."""

    enabled: bool = True
    wandb: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "project": "", "entity": "", "mode": "offline"}
    )
    mlflow: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "tracking_uri": "file://./mlruns", "experiment_name": "default"}
    )
    dvc: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "remote": None, "auto_push": False})
    tags: List[str] = field(default_factory=list)


@dataclass
class AWSConfig:
    """Configuration for AWS services."""

    enabled: bool = False
    region: str = "us-east-1"
    profile: Optional[str] = None
    credentials: Dict[str, str] = field(default_factory=dict)
    s3: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "bucket": {"name": "", "versioning": "disabled"}}
    )
    dynamodb: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "table": {"name": "", "billing_mode": "PAY_PER_REQUEST"}}
    )


@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""

    enabled: bool = False
    interval: int = 60
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    alerts: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheConfig:
    """Configuration for caching."""

    enabled: bool = True
    type: str = "local"
    ttl: int = 3600
    max_size: Optional[str] = None
    dir: Optional[str] = None


@dataclass
class ValidationConfig:
    """Configuration for data validation."""

    enabled: bool = True
    schema: Dict[str, Any] = field(default_factory=dict)
    required_columns: List[str] = field(default_factory=list)
    value_ranges: Dict[str, Dict[str, float]] = field(default_factory=dict)
    unique_columns: List[str] = field(default_factory=list)


@dataclass
class TransformConfig:
    """Configuration for data transformations."""

    enabled: bool = True
    encoding: Dict[str, Any] = field(default_factory=dict)
    scaling: Dict[str, Any] = field(default_factory=dict)
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    cleaning: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactConfig:
    """Configuration for artifact management."""

    save_format: str = "parquet"
    compression: Optional[str] = None
    versioning: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for data handling."""

    format: str = "parquet"
    path: Optional[str] = None
    stage: Optional[str] = None
    features: Optional[List[str]] = None
    target: Optional[str] = None
    validation: Dict[str, Any] = field(default_factory=dict)
    transforms: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageExecutionConfig:
    """Complete configuration for stage execution."""

    stage: StageConfig = field(default_factory=StageConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Stage-specific configurations
@dataclass
class BuildConfig(StageExecutionConfig):
    """Configuration specific to build stage."""

    registry: str = "ghcr.io"
    repository: str = "aai540-group3/pipeline"
    push_images: bool = True
    dockerfile_dir: str = "docker"
    images: List[str] = field(
        default_factory=lambda: [
            "base",
            "infrastruct",
            "ingest",
            "preprocess",
            "explore",
            "featurize",
            "optimize",
            "autogluon",
            "logistic",
            "neural",
            "evaluate",
            "register",
            "deploy",
            "serve",
        ]
    )


@dataclass
class IngestConfig(StageExecutionConfig):
    """Configuration specific to ingest stage."""

    sources: List[Dict[str, Any]] = field(default_factory=list)
    batch_size: int = 1000
    parallel_downloads: int = 4
    quality_checks: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "missing_threshold": 0.1, "duplicate_check": True}
    )


@dataclass
class PreprocessConfig(StageExecutionConfig):
    """Configuration specific to preprocess stage."""

    cleaning: Dict[str, Any] = field(
        default_factory=lambda: {
            "drop_columns": [],
            "missing_strategy": "mean",
            "outlier_removal": {"method": "iqr", "threshold": 1.5},
        }
    )
    encoding: Dict[str, Any] = field(
        default_factory=lambda: {"categorical": {"method": "label"}, "numerical": {"method": "standard"}}
    )
    feature_selection: Dict[str, Any] = field(default_factory=dict)
    target_processing: Dict[str, Any] = field(
        default_factory=lambda: {"encoding": {"NO": 0, ">30": 0, "<30": 1}, "validation": {"min_class_ratio": 0.1}}
    )


@dataclass
class FeaturizeConfig(StageExecutionConfig):
    """Configuration specific to featurize stage."""

    feature_groups: List[Dict[str, Any]] = field(default_factory=list)
    interactions: bool = False
    polynomial: bool = False
    feature_selection: Dict[str, Any] = field(default_factory=lambda: {"method": "importance", "n_features": 20})
    scaling: Dict[str, Any] = field(default_factory=lambda: {"method": "standard", "with_mean": True, "with_std": True})


@dataclass
class ModelConfig:
    """Configuration for model training."""

    name: str = ""
    type: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    architecture: Optional[Dict[str, Any]] = None
    optimizer: Optional[Dict[str, Any]] = None
    loss: Optional[str] = None
    metrics: List[str] = field(default_factory=list)


@dataclass
class TrainConfig(StageExecutionConfig):
    """Configuration specific to train stage."""

    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 32
    epochs: int = 10
    validation_split: float = 0.2
    early_stopping: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "patience": 5, "monitor": "val_loss"}
    )


@dataclass
class OptimizeConfig(StageExecutionConfig):
    """Configuration specific to optimize stage."""

    n_trials: int = 100
    timeout: int = 3600
    metric: str = "accuracy"
    direction: str = "maximize"
    param_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluateConfig(StageExecutionConfig):
    """Configuration specific to evaluate stage."""

    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "roc_auc"])
    threshold: float = 0.5
    feature_importance: bool = True
    confusion_matrix: bool = True
    plots: Dict[str, Any] = field(
        default_factory=lambda: {"roc_curve": True, "precision_recall_curve": True, "confusion_matrix": True}
    )


@dataclass
class DeployConfig(StageExecutionConfig):
    """Configuration specific to deploy stage."""

    model_format: str = "onnx"
    platform: str = "huggingface"
    requirements: List[str] = field(default_factory=list)
    endpoints: Dict[str, Any] = field(default_factory=lambda: {"prediction": {"timeout": 30, "batch_size": 32}})
