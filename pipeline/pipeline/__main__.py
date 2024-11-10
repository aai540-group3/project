import argparse
import importlib
import threading

from loguru import logger


@logger.catch(reraise=True)
def run_stage(stage_name):
    # Construct the module name and class name based on the stage name
    module_name = f"pipeline.stages.{stage_name}"
    class_name = "".join(word.capitalize() for word in stage_name.split("_"))
    logger.debug(f"Importing module: {module_name}")

    # Dynamically import the module
    module = importlib.import_module(module_name)

    # Dynamically get the class from the module
    stage_class = getattr(module, class_name)

    # Instantiate and execute the stage
    stage_instance = stage_class()
    stage_instance.execute()


@logger.catch(reraise=True)
def main():
    # GET STAGE NAMES
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stages",
        nargs="+",  # Accept multiple stages as arguments
        type=str,
        help="Names of the pipeline stages to execute",
    )
    args = parser.parse_args()
    stage_names = args.stages

    # Create and start a thread for each stage
    threads = []
    for stage_name in stage_names:
        thread = threading.Thread(target=run_stage, args=(stage_name,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    logger.info("All stages completed.")


if __name__ == "__main__":
    main()
