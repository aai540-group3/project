"""
Ingest Stage
============

.. module:: pipeline.stages.ingest
   :synopsis: Data ingestion and source management with detailed logging

.. moduleauthor:: aai540-group3
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import hydra
import pandas as pd
import requests
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ucimlrepo import fetch_ucirepo

from pipeline.stages.base import PipelineStage


class IngestStage(PipelineStage):
    """Data ingestion stage implementation with detailed logging."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize data ingestion stage."""
        super().__init__(cfg)
        self.start_time = time.time()
        self.sources = {}
        self.ingestion_errors: List[Dict[str, str]] = []
        self._initialize_sources()

        self.metadata: Dict[str, Any] = {
            "ingestion": {
                "sources_processed": [],
                "status": {"successful": [], "failed": []},
                "execution_time": None,
            },
            "config": OmegaConf.to_container(self.stage_config, resolve=True),
        }

    def run(self) -> None:
        """Execute data ingestion pipeline."""
        logger.info("Starting data ingestion")
        success = False

        try:
            # Fetch data from configured sources
            data = self._fetch_data()

            # Save raw data
            self._save_raw_data(data)

            # Log data statistics
            self._log_data_stats(data)

            # Record execution time
            self.metadata["ingestion"]["execution_time"] = time.time() - self.start_time

            # Save metadata
            self._save_metadata()

            success = True
            logger.info("Data ingestion completed successfully")

        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            self.ingestion_errors.append({"error": str(e)})
            raise RuntimeError(f"Data ingestion failed: {str(e)}")
        finally:
            self._log_final_status(success)
            self._cleanup_tracking()

    def _initialize_sources(self) -> None:
        """Initialize data sources."""
        self.sources = {
            "uci": self._fetch_from_uci,
            "s3": self._fetch_from_s3,
            "api": self._fetch_from_api,
            "local": self._fetch_from_local,
        }

    def _fetch_data(self) -> pd.DataFrame:
        """Fetch data from configured sources.

        :return: Combined dataset
        :rtype: pd.DataFrame
        :raises ValueError: If no valid sources configured
        """
        datasets = []

        data_config = self.cfg.get("data", {})
        sources_config = data_config.get("sources", [])
        if not sources_config:
            logger.error("No data sources configured in 'data.sources'")
            raise ValueError("No data sources configured")

        for source_config in sources_config:
            source_type = source_config.get("type")
            if source_type not in self.sources:
                logger.warning(f"Unknown source type: {source_type}")
                continue

            try:
                data = self.sources[source_type](source_config)
                if data is not None:
                    datasets.append(data)
                    self.metadata["ingestion"]["sources_processed"].append(source_config.get("name", source_type))
                    self.metadata["ingestion"]["status"]["successful"].append(source_config.get("name", source_type))
            except Exception as e:
                logger.error(f"Failed to fetch data from {source_type}: {e}")
                self.ingestion_errors.append({"source": source_type, "error": str(e)})
                if source_config.get("required", False):
                    raise

        if not datasets:
            raise ValueError("No data fetched from any source")

        # Combine datasets if multiple sources
        if len(datasets) > 1:
            combined_data = pd.concat(datasets, ignore_index=True)
            logger.info("Datasets from multiple sources combined successfully")
            return combined_data
        return datasets[0]

    def _fetch_from_uci(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Fetch data from UCI repository.

        :param config: Source configuration
        :type config: Dict[str, Any]
        :return: UCI dataset
        :rtype: pd.DataFrame
        :raises RuntimeError: If UCI fetch fails
        """
        try:
            logger.info("Fetching data from UCI repository")
            # Fetch dataset using the provided dataset_id
            dataset_id = config.get("dataset_id")
            if not dataset_id:
                logger.error("UCI source configuration missing 'dataset_id'")
                raise ValueError("UCI source configuration must include 'dataset_id'")

            diabetes_data = fetch_ucirepo(id=dataset_id)

            # Extract components
            X = diabetes_data.data.features
            y = diabetes_data.data.targets
            metadata = diabetes_data.metadata
            variables = diabetes_data.variables

            # Save metadata (using the new saving mechanism)
            self._save_ucirepo_metadata(metadata, variables)

            # Combine features and target
            df = pd.concat([X, y], axis=1)
            logger.debug(f"UCI data fetched successfully with shape {df.shape}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch UCI data: {e}")
            raise RuntimeError(f"UCI data fetch failed: {e}")

    def _save_ucirepo_metadata(self, metadata: Dict[str, Any], variables: pd.DataFrame) -> None:
        """Save UCI repository metadata."""
        output_dir = self.get_path("raw")
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / "metadata.json"
        variables_path = output_dir / "variables.json"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved UCI metadata to {metadata_path}")

        variables.to_json(variables_path, orient="records", indent=2)
        logger.debug(f"Saved UCI variables to {variables_path}")

    def _fetch_from_s3(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch data from S3.

        :param config: Source configuration
        :type config: Dict[str, Any]
        :return: S3 dataset
        :rtype: Optional[pd.DataFrame]
        :raises RuntimeError: If S3 fetch fails
        """
        try:
            bucket = config.get("bucket")
            key = config.get("key")
            logger.info(f"Fetching data from S3 bucket {bucket}, key {key}")
            s3_client = self._init_s3_client(config)

            response = s3_client.get_object(Bucket=bucket, Key=key)

            data_format = config.get("format", "csv").lower()
            if data_format == "csv":
                df = pd.read_csv(response["Body"])
            elif data_format == "parquet":
                df = pd.read_parquet(response["Body"])
            else:
                raise ValueError(f"Unsupported format: {data_format}")

            logger.debug(f"S3 data fetched successfully with shape {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch S3 data: {e}")
            if config.get("required", False):
                raise RuntimeError(f"S3 data fetch failed: {e}")
            return None

    def _init_s3_client(self, config: Dict[str, Any]) -> boto3.client:
        """Initialize S3 client.

        :param config: Source configuration
        :type config: Dict[str, Any]
        :return: S3 client
        :rtype: boto3.client
        """
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=config.get("credentials", {}).get("access_key"),
            aws_secret_access_key=config.get("credentials", {}).get("secret_key"),
            region_name=config.get("region", "us-east-1"),
        )
        return s3_client

    def _fetch_from_api(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch data from API.

        :param config: Source configuration
        :type config: Dict[str, Any]
        :return: API dataset
        :rtype: Optional[pd.DataFrame]
        :raises RuntimeError: If API fetch fails
        """
        try:
            url = config.get("url")
            logger.info(f"Fetching data from API endpoint {url}")
            headers = config.get("headers", {})
            params = config.get("params", {})

            response = requests.get(url, headers=headers, params=params, timeout=config.get("timeout", 30))
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data)
            logger.debug(f"API data fetched successfully with shape {df.shape}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch API data: {e}")
            if config.get("required", False):
                raise RuntimeError(f"API data fetch failed: {e}")
            return None

    def _fetch_from_local(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Fetch data from local file.

        :param config: Source configuration
        :type config: Dict[str, Any]
        :return: Local dataset
        :rtype: Optional[pd.DataFrame]
        :raises RuntimeError: If local fetch fails
        """
        try:
            file_path = Path(config.get("path"))
            logger.info(f"Fetching data from local file {file_path}")
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_suffix = file_path.suffix.lower()
            if file_suffix == ".csv":
                df = pd.read_csv(file_path)
            elif file_suffix == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_suffix == ".json":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_suffix}")

            logger.debug(f"Local data fetched successfully with shape {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch local data: {e}")
            if config.get("required", False):
                raise RuntimeError(f"Local data fetch failed: {e}")
            return None

    def _save_raw_data(self, data: pd.DataFrame) -> None:
        """Save raw data in multiple formats.

        :param data: Data to save
        :type data: pd.DataFrame
        """
        output_dir = self.get_path("raw")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save in multiple formats
        parquet_path = output_dir / "data.parquet"
        csv_path = output_dir / "data.csv"

        data.to_parquet(parquet_path)
        data.to_csv(csv_path, index=False)

        logger.info(f"Raw data saved to {parquet_path} and {csv_path}")

        # Log artifacts if tracking is enabled
        self.log_artifact(str(parquet_path))
        self.log_artifact(str(csv_path))

    def _save_metadata(self) -> None:
        """Save ingestion metadata."""
        try:
            output_path = self.get_path("data") / "ingestion_metadata.json"

            # Create metadata dictionary
            metadata_dict = {
                "ingestion": self.metadata["ingestion"],
                "config": self.metadata["config"],  # Don't use OmegaConf.to_container() here
            }

            # If config is an OmegaConf object, convert it
            if OmegaConf.is_config(metadata_dict["config"]):
                metadata_dict["config"] = OmegaConf.to_container(metadata_dict["config"], resolve=True)

            with open(output_path, "w") as f:
                json.dump(metadata_dict, f, indent=2)

            logger.info(f"Saved ingestion metadata to {output_path}")

            # Log metadata as artifact if tracking is enabled
            if self.tracking_initialized:
                self.log_artifact(str(output_path))

        except Exception as e:
            logger.error(f"Error in _save_metadata: {str(e)}")
            raise

    def _log_data_stats(self, data: pd.DataFrame) -> None:
        """Log data statistics.

        :param data: Input data
        :type data: pd.DataFrame
        """
        stats = {
            "n_samples": len(data),
            "n_features": len(data.columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": int(data.isnull().sum().sum()),
            "dtypes": {str(k): int(v) for k, v in data.dtypes.value_counts().items()},
        }

        logger.info(f"Data statistics: {stats}")

        if self.tracking_initialized:
            self.log_metrics(stats)

    def _log_final_status(self, success: bool) -> None:
        """Log final ingestion status.

        :param success: Whether ingestion was successful
        :type success: bool
        """
        if not self.tracking_initialized:
            return

        # Log configuration
        self.log_metrics(
            {
                "data_ingestion_success": int(success),
                "sources_processed": len(self.metadata["ingestion"]["sources_processed"]),
                "ingestion_duration_seconds": time.time() - self.start_time,
            }
        )

        # Log detailed metrics about each source type
        for source in self.metadata["ingestion"]["status"]["successful"]:
            self.log_metrics({f"{source}_ingestion_status": 1})

        for source in [error["source"] for error in self.ingestion_errors]:
            self.log_metrics({f"{source}_ingestion_status": 0})

        if self.ingestion_errors:
            error_count = len(self.ingestion_errors)
            total_sources = len(self.metadata["ingestion"]["sources_processed"])
            self.log_metrics(
                {
                    "ingestion_errors": error_count,
                    "failure_rate": error_count / total_sources if total_sources > 0 else 0,
                }
            )


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the InfrastructStage.

    :param cfg: Configuration dictionary
    :type cfg: DictConfig
    """
    stage = IngestStage(cfg)
    stage.run()


if __name__ == "__main__":
    main()
