import json
import pathlib
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from omegaconf import OmegaConf

load_dotenv(find_dotenv(filename=".env"))


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self):
        """Initialize the pipeline stage."""
        self.name = self.__class__.__name__.lower()
        self.cfg = OmegaConf.load("params.yaml")

        # Set visualization defaults from config
        plt.style.use(self.cfg.visualization.style)
        sns.set_theme(
            style=self.cfg.visualization.theme.style,
            context=self.cfg.visualization.theme.context,
            font_scale=self.cfg.visualization.theme.font_scale,
            rc=self.cfg.visualization.theme.rc,
        )

        # Set color palette from config
        self.colors = self.cfg.colors

    @logger.catch(reraise=True)
    def load_data(self, filename: str, subdir: str = "data") -> Union[pd.DataFrame, dict]:
        """Load data from file based on extension."""
        file_path = pathlib.Path(subdir) / filename
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def save_output(self, data: Union[pd.DataFrame, dict], filename: str, subdir: str = "data") -> Path:
        """Save output data in appropriate format."""
        output_path = pathlib.Path(subdir) / filename
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
                    # Ensure that all keys are serializable
                    def make_serializable(obj):
                        if isinstance(obj, dict):
                            return {str(k): make_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [make_serializable(v) for v in obj]
                        else:
                            return obj

                    data = make_serializable(data)
                    output_path.write_text(json.dumps(data, indent=2))
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

    def save_plot(
        self,
        name: str,
        plot_func: Callable,
        data: Optional[pd.DataFrame] = None,
        additional_elements: Optional[Callable] = None,
        **kwargs,
    ) -> Path:
        """Create and save a plot."""
        plot_path = pathlib.Path(self.cfg.paths.plots) / self.name / f"{name}.png"

        try:
            # Create figure with specified size
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)))

            # Extract title if provided
            title = kwargs.pop("title", None)

            # Create main plot
            if data is not None:
                if callable(plot_func):
                    plot_func(data=data, **kwargs)
                else:
                    plot_func(data=data, ax=ax, **kwargs)
            else:
                if callable(plot_func):
                    plot_func(**kwargs)
                else:
                    plot_func(ax=ax, **kwargs)

            # Add any additional plot elements
            if additional_elements:
                additional_elements()

            # Set title if provided
            if title:
                plt.title(title)

            # Save plot
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches="tight")

            logger.debug(f"Saved plot to {plot_path}")
            return plot_path

        except Exception as e:
            logger.error(f"Failed to save plot {name}: {str(e)}")
            raise
        finally:
            plt.close("all")

    def load_config(self, stage: Optional[str] = None) -> dict:
        """Load stage-specific configuration."""
        if stage is None:
            stage = self.name
        return self.cfg.stages[stage]

    @abstractmethod
    def run(self):
        """Execute stage-specific logic."""
        pass

    def execute(self):
        """Execute the pipeline stage in a separate thread."""
        logger.info(f"EXECUTING STAGE: {self.name}")
        start_time = datetime.now()

        # Define a function to run in the thread
        def thread_run():
            try:
                self.run()
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"COMPLETED STAGE: {self.name} in {duration:.2f}s")
            except Exception as e:
                logger.error(f"FAILED STAGE: {self.name}")
                logger.error(f"Error: {str(e)}")
                logger.error("Full error:", exc_info=True)
                raise
            finally:
                logger.complete()

        # Start the run function in a separate thread
        thread = threading.Thread(target=thread_run)
        thread.start()
        thread.join()
