all: install
	cd pipeline && ./deconflict-dvc.sh
	cd pipeline && dvc repro

install:
	pip install -q dvc[s3]

chkpkg:
	cd packaging && make chkpkg

clean: clean_uv clean_dvc clean_docker clean_cache

clean_uv:
	uv cache clean

clean_dvc:
	dvc gc -fw && cd ..

clean_docker:
	docker system prune -a -f

clean_cache:
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
	find . -name "*.log" -exec rm -rf {} +

.PHONY: all clean install chkpkg clean_uv clean_dvc clean_docker clean_cache
