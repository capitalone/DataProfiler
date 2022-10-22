venv: requirements-dev.txt requirements-test.txt
	python3 -m venv venv

	. venv/bin/activate; \
	pip3 install -r requirements-dev.txt; \
	pip3 install -r requirements-test.txt

	pre-commit install
	pre-commit run
