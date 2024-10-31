"""
Featurize Stage
=====================

.. module:: pipeline.stages.featurize
   :synopsis: Feature engineering and preprocessing stage

.. moduleauthor:: aai540-group3
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.stages.base import PipelineStage
from pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class FeaturizeStage(PipelineStage):
    """Feature engineering stage implementation.

    Handles feature creation, selection, and preprocessing.
    Supports both quick validation and full processing modes.

    :param cfg: Feature engineering configuration
    :type cfg: DictConfig
    :raises ValueError: If configuration is invalid
    """

    def run(self) -> None:
        """Execute feature engineering pipeline.

        :raises RuntimeError: If feature engineering fails
        :raises IOError: If saving features fails
        """
        logger.info(f"Starting feature engineering in {self.cfg.experiment.name} mode")

        try:
            # Load data
            train_data = pd.read_parquet(self.cfg.paths.interim / "train.parquet")
            val_data = pd.read_parquet(self.cfg.paths.interim / "val.parquet")
            test_data = pd.read_parquet(self.cfg.paths.interim / "test.parquet")

            # Create features
            train_features = self._create_features(train_data)
            val_features = self._create_features(val_data)
            test_features = self._create_features(test_data)

            # Scale features
            train_scaled, val_scaled, test_scaled = self._scale_features(
                train_features, val_features, test_features
            )

            # Save processed features
            self._save_features(train_scaled, val_scaled, test_scaled)

            logger.info("Feature engineering completed successfully")

        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise RuntimeError(f"Feature engineering failed: {str(e)}")

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from input data.

        :param df: Input dataframe
        :type df: pd.DataFrame
        :return: Dataframe with engineered features
        :rtype: pd.DataFrame
        """
        features = df.copy()

        if self.cfg.experiment.name != "quick":
            # Create interaction features
            if self.cfg.features.interactions:
                features = self._create_interactions(features)

            # Create polynomial features
            if self.cfg.features.polynomial:
                features = self._create_polynomial_features(features)

            # Create domain-specific features
            features = self._create_domain_features(features)

        return features

    def _scale_features(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler.

        :param train: Training data
        :type train: pd.DataFrame
        :param val: Validation data
        :type val: pd.DataFrame
        :param test: Test data
        :type test: pd.DataFrame
        :return: Scaled datasets
        :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        numeric_features = train.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()

        train_scaled = train.copy()
        val_scaled = val.copy()
        test_scaled = test.copy()

        train_scaled[numeric_features] = scaler.fit_transform(train[numeric_features])
        val_scaled[numeric_features] = scaler.transform(val[numeric_features])
        test_scaled[numeric_features] = scaler.transform(test[numeric_features])

        # Save scaler
        self._save_scaler(scaler)

        return train_scaled, val_scaled, test_scaled

    def _save_features(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        """Save processed features.

        :param train: Training features
        :type train: pd.DataFrame
        :param val: Validation features
        :type val: pd.DataFrame
        :param test: Test features
        :type test: pd.DataFrame
        :raises IOError: If saving fails
        """
        try:
            output_dir = Path(self.cfg.paths.processed)
            output_dir.mkdir(parents=True, exist_ok=True)

            train.to_parquet(output_dir / "train_features.parquet")
            val.to_parquet(output_dir / "val_features.parquet")
            test.to_parquet(output_dir / "test_features.parquet")

            # Log feature statistics
            self._log_feature_stats(train)

        except Exception as e:
            raise IOError(f"Failed to save features: {e}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        stage = FeaturizeStage(cfg)
        stage.run()

    main()
