setup: requirements.txt requirements-dev.txt requirements-test.txt
	python3 -m venv venv

	. venv/bin/activate && \
	pip3 install --no-cache-dir -r requirements-ml.txt && \
	pip3 install --no-cache-dir -r requirements.txt && \
	pip3 install --no-cache-dir -r requirements-dev.txt && \
	pip3 install --no-cache-dir -r requirements-reports.txt && \
	pip3 install --no-cache-dir -r requirements-test.txt && \
	pip3 install -e . && \
	pre-commit install && \
	pre-commit run

format:
	pre-commit run

test:
	DATAPROFILER_SEED=0 venv/bin/python -m unittest discover -p "test*.py"
