import json
import sys
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


class Stage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self):
        """Initialize the pipeline stage."""
        self.name = self.__class__.__name__.lower()
        self.cfg = OmegaConf.load("params.yaml")
        self.live = None

    def load_data(self, filename: str, subdir: str = "data") -> pd.DataFrame:
        """Load data from file based on extension."""
        file_path = Path(subdir) / filename
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def log_metrics(self, metrics):
        """Log metrics to DVC Live and console."""
        for name, value in metrics.items():
            if self.live:
                self.live.log_metric(name, value)
            logger.info(f"{name}: {value:.4f}")

    def log_params(self, config, data_hash, X_train, X_val, X_test):
        """Log parameters to DVC Live."""
        params = {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "features": len(X_train.columns),
            "data_hash": data_hash,
        }

        hyperparameters = config.get("hyperparameters", {})
        for param_name, param_value in hyperparameters.items():
            params[f"{self.name}_{param_name}"] = param_value

        if self.live:
            self.live.log_params(params)

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types."""
        import numpy as np
        import pandas as pd

        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def save_output(self, data: Union[pd.DataFrame, dict], filename: str, subdir: str = "data") -> Path:
        """Save output data in appropriate format."""
        output_path = Path(subdir) / filename
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(data, pd.DataFrame):
                if filename.endswith(".parquet"):
                    data.to_parquet(output_path)
                elif filename.endswith(".csv"):
                    data.to_csv(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported format for DataFrame: {filename}")
            elif isinstance(data, (dict, list)):
                if filename.endswith(".json"):
                    # Convert numpy types before JSON serialization
                    serializable_data = self._convert_numpy_types(data)
                    output_path.write_text(json.dumps(serializable_data, indent=2))
                else:
                    raise ValueError(f"Unsupported format for dict/list: {filename}")
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            logger.debug(f"Saved {filename} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save {filename}: {str(e)}")
            raise

    def save_metrics(self, name: str, data: Dict[str, Any], timestamp: bool = True) -> Path:
        """Save metric data as JSON."""
        if timestamp:
            data = {"timestamp": datetime.now(timezone.utc).isoformat(), "stage": self.name, **data}
        return self.save_output(data, f"{name}.json", f"metrics/{self.name}")

    def execute(self):
        """Execute the pipeline stage in a separate thread."""
        logger.info(f"EXECUTING STAGE: {self.name}")
        start_time = datetime.now()

        def thread_run():
            try:
                self.run()
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"COMPLETED STAGE: {self.name} in {duration:.2f}s")
            except Exception as e:
                logger.error(f"FAILED STAGE: {self.name}")
                logger.exception(e)
                sys.exit(1)
            finally:
                if self.live:
                    self.live.end()
                logger.complete()

        thread = threading.Thread(target=thread_run)
        thread.start()
        thread.join()

    @abstractmethod
    def run(self):
        """Execute stage-specific logic."""
        pass
