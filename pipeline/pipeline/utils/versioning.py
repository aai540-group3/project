"""
Model and Data Versioning
======================

.. module:: pipeline.utils.versioning
   :synopsis: Version control for models and data

.. moduleauthor:: aai540-group3
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class VersionControl:
    """Version control for models and data."""

    def __init__(self, base_path: Path):
        """Initialize version control.

        :param base_path: Base path for versioning
        :type base_path: Path
        """
        self.base_path = Path(base_path)
        self.version_file = self.base_path / "versions.json"
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict:
        """Load version information.

        :return: Version information
        :rtype: Dict
        """
        if self.version_file.exists():
            with self.version_file.open("r") as f:
                return json.load(f)
        return {}

    def _save_versions(self) -> None:
        """Save version information."""
        with self.version_file.open("w") as f:
            json.dump(self.versions, f, indent=2)

    def compute_hash(self, file_path: Path) -> str:
        """Compute file hash.

        :param file_path: Path to file
        :type file_path: Path
        :return: File hash
        :rtype: str
        """
        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def register_version(self, name: str, path: Path, metadata: Optional[Dict] = None) -> str:
        """Register new version.

        :param name: Version name
        :type name: str
        :param path: Path to versioned file
        :type path: Path
        :param metadata: Additional metadata
        :type metadata: Optional[Dict]
        :return: Version hash
        :rtype: str
        """
        version_hash = self.compute_hash(path)

        if name not in self.versions:
            self.versions[name] = []

        self.versions[name].append({"hash": version_hash, "path": str(path), "metadata": metadata or {}})

        self._save_versions()
        return version_hash

    def get_version(self, name: str, version_hash: str) -> Optional[Dict]:
        """Get version information.

        :param name: Version name
        :type name: str
        :param version_hash: Version hash
        :type version_hash: str
        :return: Version information
        :rtype: Optional[Dict]
        """
        if name in self.versions:
            for version in self.versions[name]:
                if version["hash"] == version_hash:
                    return version
        return None

    def get_latest_version(self, name: str) -> Optional[Dict]:
        """Get latest version.

        :param name: Version name
        :type name: str
        :return: Latest version information
        :rtype: Optional[Dict]
        """
        if name in self.versions and self.versions[name]:
            return self.versions[name][-1]
        return None
