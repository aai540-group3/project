import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import dvc.api
import yaml

logger = logging.getLogger(__name__)


class DataVersioning:
    """Data and model versioning with DVC."""

    def __init__(self, repo_path: Union[str, Path]):
        self.repo_path = Path(repo_path)
        self.cache = {}

    def get_data_hash(self, path: Union[str, Path]) -> str:
        """Get hash of data file."""
        path = Path(path)
        if path not in self.cache:
            with path.open("rb") as f:
                content = f.read()
                self.cache[path] = hashlib.md5(content).hexdigest()
        return self.cache[path]

    def log_data_version(
        self,
        data_path: Union[str, Path],
        metadata_path: Union[str, Path],
        additional_info: Optional[Dict] = None,
    ) -> None:
        """Log data version information."""
        data_path = Path(data_path)
        metadata_path = Path(metadata_path)

        metadata = {
            "data_path": str(data_path),
            "data_hash": self.get_data_hash(data_path),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        if additional_info:
            metadata.update(additional_info)

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)

    def check_data_version(
        self, data_path: Union[str, Path], metadata_path: Union[str, Path]
    ) -> bool:
        """Check if data version matches metadata."""
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            return False

        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        current_hash = self.get_data_hash(data_path)
        return current_hash == metadata.get("data_hash")
