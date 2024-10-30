from abc import ABC, abstractmethod

from omegaconf import DictConfig

from utils.experiment import ExperimentTracker
from utils.registry import ModelRegistry


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
