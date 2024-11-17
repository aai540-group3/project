all: install
	./deconflict-dvc.sh
	dvc repro

install:
	pip install -q dvc[s3]

chkpkg:
	cd ../packaging && make chkpkg

unlock:
	rm -f .dvc/tmp/lock
	rm -f .dvc/tmp/rwlock

clean: clean_uv clean_dvc clean_docker clean_cache

clean_uv:
	uv cache clean

clean_dvc:
	cd pipeline && dvc gc -fw && cd ..

clean_docker:
	docker system prune -a -f

clean_cache:
	cd .. && find . -name ".ruff_cache" -exec rm -rf {} +
	cd .. && find . -name ".uv_cache" -exec rm -rf {} +
	cd .. && find . -name ".venv" -exec rm -rf {} +
	cd .. && find . -name ".venv-*" -exec rm -rf {} +
	cd .. && find . -name "build" -exec rm -rf {} +
	cd .. && find . -name "debug" -exec rm -rf {} +
	cd .. && find . -name "dist" -exec rm -rf {} +
	cd .. && find . -name "dvclive" -exec rm -rf {} +
	cd .. && find . -name "mlruns" -exec rm -rf {} +
	cd .. && find . -name "outputs" -exec rm -rf {} +
	cd .. && find . -name "pipeline.egg-info" -exec rm -rf {} +
	cd .. && find . -name "__pycache__" -exec rm -rf {} +
	cd .. && find . -name ".pytest_cache" -exec rm -rf {} +
	cd .. && find . -name "*.egg-info" -exec rm -rf {} +
	cd .. && find . -name "*.log" -exec rm -rf {} +

.PHONY: all clean install chkpkg unlock clean_uv clean_dvc clean_docker clean_cache
