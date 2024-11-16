"""
Pipeline Stage Runner
=====================

This module provides functionality to dynamically run pipeline stages.

.. module:: run_stages
  :synopsis: Dynamic pipeline stage executor with threading support

.. moduleauthor:: aai540-group3
"""

import argparse
import importlib
import threading

from loguru import logger


def run_stage(stage_name: str) -> None:
    """Load and execute a pipeline stage using a mapping.

    :param stage_name: Name of the stage to execute
    :type stage_name: str
    :raises ImportError: If stage module cannot be imported
    """
    stage_mapping = {
        "autogluon": ("pipeline.stages.autogluon", "Autogluon"),
        "explore": ("pipeline.stages.explore", "Explore"),
        "feast": ("pipeline.stages.feast", "Feast"),
        "featurize": ("pipeline.stages.featurize", "Featurize"),
        "infrastruct": ("pipeline.stages.infrastruct", "Infrastruct"),
        "ingest": ("pipeline.stages.ingest", "Ingest"),
        "logisticregression": ("pipeline.stages.logisticregression", "LogisticRegression"),
        "neuralnetwork": ("pipeline.stages.neuralnetwork", "NeuralNetwork"),
        "preprocess": ("pipeline.stages.preprocess", "Preprocess"),
    }

    if stage_name not in stage_mapping:
        logger.error(f"Stage '{stage_name}' is not defined in the stage mapping.")
        raise ValueError(f"Stage '{stage_name}' is not defined in the stage mapping.")

    module_name, class_name = stage_mapping[stage_name]
    logger.debug(f"Importing module: {module_name}")

    try:
        module = importlib.import_module(module_name)
        stage_class = getattr(module, class_name)
        stage_instance = stage_class()
        stage_instance.execute()
    except Exception as e:
        logger.error(f"Error in stage '{stage_name}': {e}")
        raise


@logger.catch()
def main() -> None:
    """Main entry point for parallel pipeline execution."""
    parser = argparse.ArgumentParser(description="Execute pipeline stages in parallel")
    parser.add_argument(
        "stages",
        nargs="+",
        type=str,
        help="Names of stages to execute",
    )
    args = parser.parse_args()
    stage_names = args.stages

    threads = []
    for stage_name in stage_names:
        thread = threading.Thread(target=run_stage, args=(stage_name,), name=f"Thread-{stage_name}")
        thread.start()
        threads.append(thread)
        logger.info(f"Started thread for stage: {stage_name}")

    for thread in threads:
        thread.join()
        logger.debug(f"Thread completed: {thread.name}")

    logger.info("All stages completed successfully")


if __name__ == "__main__":
    main()
