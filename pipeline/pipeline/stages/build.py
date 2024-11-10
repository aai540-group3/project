"""
Build Stage
===========

.. module:: pipeline.stages.build
   :synopsis: Container image build and push stage

.. moduleauthor:: aai540-group3
"""

from pathlib import Path

from loguru import logger

import docker
from pipeline.stages.base import PipelineStage


class BuildStage(PipelineStage):
    """Build stage for container images.

    Handles building and pushing Docker images for pipeline stages.
    """

    def __init__(self) -> None:
        """Initialize build stage."""
        super().__init__()

        # Initialize Docker client
        try:
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Docker client: {e}")

    def run(self) -> None:
        """Execute build pipeline."""
        logger.info("Starting container image builds")

        try:
            # Build images defined in configuration
            for image in self.stage_config.images:
                self._build_and_push_image(image)

            # Log final metrics
            self.log_metrics(
                {
                    f"{self.stage_name}_images_built": len(self.stage_config.images),
                    f"{self.stage_name}_build_success": 1.0,
                }
            )

        except Exception:
            self.log_metrics({f"{self.stage_name}_build_success": 0.0})
            raise

    def _build_and_push_image(self, image_name: str) -> None:
        """Build and push a single container image.

        :param image_name: Name of the image to build
        :type image_name: str
        :raises RuntimeError: If build or push fails
        """
        tag = f"{self.stage_config.registry}/{self.stage_config.repository}-{image_name}:latest"
        dockerfile = Path(self.stage_config.dockerfile_dir) / f"Dockerfile.{image_name}"

        if not dockerfile.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")

        try:
            # Build image
            logger.info(f"Building image: {tag}")
            build_logs = self.client.api.build(
                path=str(dockerfile.parent), dockerfile=str(dockerfile.name), tag=tag, decode=True, rm=True
            )

            # Process build logs
            for chunk in build_logs:
                if error := chunk.get("error"):
                    raise RuntimeError(f"Build failed: {error}")
                if stream := chunk.get("stream"):
                    logger.debug(stream.strip())

            # Push if configured
            if self.stage_config.push_images:
                logger.info(f"Pushing image: {tag}")
                push_logs = self.client.api.push(tag, stream=True, decode=True)

                for chunk in push_logs:
                    if error := chunk.get("error"):
                        raise RuntimeError(f"Push failed: {error}")
                    if status := chunk.get("status"):
                        logger.debug(status.strip())

            # Log artifact
            self.log_artifact(str(dockerfile))

        except Exception as e:
            raise RuntimeError(f"Failed to build/push {image_name}: {e}")

    def _generate_stage_visualizations(self, output_dir: Path) -> None:
        """Generate stage-specific visualizations.

        The build stage doesn't generate visualizations, so this is a no-op.

        :param output_dir: Output directory for visualizations
        :type output_dir: Path
        """
        # Build stage doesn't generate visualizations
        pass


if __name__ == "__main__":
    BuildStage().execute()
