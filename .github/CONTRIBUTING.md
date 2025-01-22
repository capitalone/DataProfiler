# Contributing to DataProfiler
First off, thanks for your input! We love to hear feedback from the community and we want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## Makefile Commands
The Makefile at the root of the repo contains several useful commands.

To setup a Python virtual environment, run the following commands:
```cli
make setup
source ./venv/bin/activate
```

To format the code, run `make format`.

To test the code, run `make test`.

Alternatively, follow the steps below.

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

For more nuanced testing runs, check out more detailed documentation [here](https://capitalone.github.io/DataProfiler/docs/0.11.0/html/install.html#testing).

## Creating [Pull Requests](https://github.com/capitalone/DataProfiler/pulls)
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `dev`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Feature Pull Request Workflow
When working on a new feature, which will require multiple pull requests, the following workflow is to ensure that it is most efficient and safe for developers and reviewers.

1. A feature branch that an author is working on should live on their fork and not on `capitalone/dataprofiler`. This is becuase a rebase requires a force push and we do not want to open up permissions on a feature branch on `capitalone/dataprofiler` for any potential user to force push to a `feature/<branch_name>` on `capitalone/dataprofiler`.
2. Feature branch naming convention:
  - Feature branches on forks and on `capitalone/dataprofiler`: `feature/<feature_name>`
  - GH Pages feature branch naming convention: `feature/dev-gh-pages/<feature_name>`
3. Staging Branch is to be used as a proxy for a rebase. Note: there should be no commits to the staging branch once it is made. Staging is exactly what the name suggest -- a staging area for PRs to `dev` and `dev-gh-pages` and to `main` and `gh-pages`. Naming convention for `staging/` is:
  - `staging/dev/<feature_name>`
  - `staging/dev-gh-pages/<feature_name>`
  - `staging/main/<version_tag>`
4. Release will live under the naming convention and folder of `release/`. The naming convention, for example, is `release/0.0.1` for the branch itself. `0.0.1` or whatever the version tag is should match the version tag that will be used for deployment when drafting a new release. Hot fixes and and commits are permitted to be PR'd into the release branch.
5. Maintainers may from time to time make an exception to the above workflow and host a feature branch on `capitalone/dataprofiler`. Permissions would then be reconfigured for a force push to core team membership exclusively.

See below image for the above text points visualized for both `main` and `gh-pages` workflows.

![image](https://github.com/capitalone/DataProfiler/raw/gh-pages/docs/source/_static/images/branching_workflow_diagram.png)

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

## Updating Dependencies
If you make changes to the `requirements` text files, please also update the `additional_dependencies` list under the `mypy` hook in `.pre-commit-config.yaml`. This is necessary for accurate type-checking.

## Contributing Documentation Changes and Fixes
When making adjustments or contributions to documentation, please use the `dev-gh-pages` branch.  This is where all the documentation lives.
After you've completed your edits, open a Github Pull Request (PR) to merge into `dev-gh-pages` from your fork.  During a version release, `dev-gh-pages` is merged
into the `gh-pages` branch (after `update_documentation.py` is run) and is the version associated with the documentation website and stable version.
