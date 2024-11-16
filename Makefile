all:
	pip install dvc && cd pipeline && dvc repro

clean:
	rm -rf \
		.ruff_cache \
		.uv_cache \
		.venv \
		.venv-* \
		build \
		debug \
		dist \
		dvclive \
		mlruns \
		outputs \
		pipeline.egg-info \
		__pycache__ \
		.pytest_cache \
		*.egg-info
	find . -name "*.log" -delete


.PHONY: all clean
