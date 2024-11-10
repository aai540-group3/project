import importlib

__all__ = [
    "Autogluon",
    "Explore",
    "Feast",
    "Featurize",
    "Infrastruct",
    "Ingest",
    "PipelineStage",
    "Preprocess",
]


# Use a custom __getattr__ for lazy loading of modules
def __getattr__(name):
    if name in __all__:
        # Import module dynamically based on the class name
        module = importlib.import_module(f"pipeline.stages.{name.lower()}")
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
