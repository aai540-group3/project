import importlib

__all__ = [
    "autogluon",
    "logistic_regression",
    "explore",
    "feast",
    "featurize",
    "infrastruct",
    "ingest",
    "pipeline",
    "preprocess",
]


# Use __getattr__ for lazy loading of modules
def __getattr__(name):
    if name in __all__:
        # Import module dynamically by class name
        module = importlib.import_module(f"pipeline.stages.{name}")
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
