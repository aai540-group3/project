all:
	./run.sh

chkpkg:
	cd ../packaging && ./check.sh

unlock:
	rm -f .dvc/tmp/lock
	rm -f .dvc/tmp/rwlock

clean:
	cd ..
	find . -name ".ruff_cache" -exec rm -rf {} +
	find . -name ".uv_cache" -exec rm -rf {} +
	find . -name ".venv" -exec rm -rf {} +
	find . -name ".venv-*" -exec rm -rf {} +
	find . -name "build" -exec rm -rf {} +
	find . -name "debug" -exec rm -rf {} +
	find . -name "dist" -exec rm -rf {} +
	find . -name "dvclive" -exec rm -rf {} +
	find . -name "mlruns" -exec rm -rf {} +
	find . -name "outputs" -exec rm -rf {} +
	find . -name "pipeline.egg-info" -exec rm -rf {} +
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name "*.egg-info" -exec rm -rf {} +
	find . -name "*.log" -delete

.PHONY: all clean unlock chkpkg