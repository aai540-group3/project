clean:
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".uv_cache" -exec rm -rf {} +
	find . -type d -name ".venv" -exec rm -rf {} +
	find . -type d -name ".venv-*" -exec rm -rf {} +
	find . -type d -name "$(VENV_DIR)" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "debug" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "dvclive" -exec rm -rf {} +
	find . -type d -name "mlruns" -exec rm -rf {} +
	find . -type d -name "outputs" -exec rm -rf {} +
	find . -type d -name "pipeline.egg-info" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.log" -exec rm -f {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

