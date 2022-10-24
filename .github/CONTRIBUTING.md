# Contributing to DataProfiler
First off, thanks for your input! We love to hear feedback from the community and we want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## Quick Setup
To set up the development environment, run the following commands:
```cli
make setup
source ./venv/bin/activate
```
This Makefile creates a Python virtual environment and installs all of the developer dependencies. Alternatively, follow the steps below.

## Dependencies
To install the dependencies for developing and updating the code base, be sure to run `pip install -r requirements-dev.txt`

## Pre-Commit
To install `pre-commit` hooks, run the following commands:

```cli
pre-commit install
pre-commit run
```

If you want to run the `pre-commit` fresh over over all the files, run the following:
```cli
pre-commit run --all-files
```

## Testing
Before running unit tests, make sure you install the testing dependencies with `pip3 install -r requirements-test.txt`.

To execute unit tests, run the following
```cli
DATAPROFILER_SEED=0 python3 -m unittest discover -p "test*.py"
```

For more nuanced testing runs, check out more detailed documentation [here](https://capitalone.github.io/DataProfiler/docs/0.8.1/html/install.html#testing).

## Creating [Pull Requests](https://github.com/capitalone/DataProfiler/pulls)
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the Apache License 2.0
In short, when you submit code changes, your submissions are understood to be under the same [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [Issues](https://github.com/capitalone/DataProfiler/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/capitalone/DataProfiler/issues/new/choose); it's that easy!

## Write bug reports with detail, background, and sample code
Detailed bug reports will make fixing the bug significantly easier.

**Great Bug Reports** tend to have:
- General information of the working environment
- A quick summary of the bug
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- Screenshots of the bug
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Use a Consistent Coding Style
Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding conventions to maintain consistency in the repo. For
docstrings, please follow [reStructuredText](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) format as Sphinx is used to autogenerate
the documentation.
