setup: requirements.txt requirements-dev.txt requirements-test.txt
	python3 -m venv venv

	. venv/bin/activate; \
	pip install -r requirements.txt; \
	pip install -r requirements-dev.txt; \
	pip install -r requirements-test.txt; \
	pip install -e .; \
	pre-commit install; \
	pre-commit run; \
