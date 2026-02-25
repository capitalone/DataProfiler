PYTHON_VERSION ?= python3.9
VENV_DIR ?= venv
REQ_FILES := requirements.txt requirements-dev.txt requirements-test.txt requirements-ml.txt requirements-reports.txt

check-python:
	@$(PYTHON_VERSION) --version | grep -E "Python (3\.9|3\.10|3\.11)" || \
	(echo "Python 3.9, 3.10, or 3.11 is required. Ensure $(PYTHON_VERSION) is installed and try again." && exit 1)

setup: check-python $(REQ_FILES)
	@$(PYTHON_VERSION) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && \
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
	DATAPROFILER_SEED=0 $(VENV_DIR)/bin/python -m unittest discover -p "test*.py"

clean:
	rm -rf .pytest_cache __pycache__

help:
	@echo "Makefile Commands:"
	@echo "  setup       - Set up the virtual environment with Python $(PYTHON_VERSION)"
	@echo "  format      - Format the code using pre-commit hooks"
	@echo "  test        - Run unit tests with unittest"
	@echo "  clean       - Remove temporary files (caches), but keep the virtual environment"
	@echo "  help        - Display this help message"
