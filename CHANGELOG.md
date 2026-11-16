# Changelog

## 0.14.1 - 2026-11-16

### Security

- Raised `pyarrow` to `>=26.0.0,<27.0.0` (previously `>=1.0.1,<24.0.0`) to
  resolve CVE-2026-92287.
- Raised `urllib3` to `>=2.8.0` (previously `>=2.5.0`) to resolve
  CVE-2026-97689, CVE-2026-97687, and CVE-2026-97688.
- Added `certifi>=2026.10.19` to resolve WS-2026-0133 in the CA bundle pulled
  in by `requests`.

## 0.14.0 - 2026-09-08

### Breaking Changes

- Dropped support for Python 3.9. The minimum supported version is now Python
  3.10 (`python_requires>=3.10`).
- Raised the `keras` requirement to `>3.4.0,<4.0.0` (previously `<=3.4.0`) and
  migrated the bundled labeler models from the TensorFlow SavedModel format to
  the single-file `.keras` format. Environments pinned to `keras<=3.4.0` are no
  longer supported, and previously saved models must be re-saved.
- Moved the bundled labeler resources from the top-level `resources/` directory
  into the package at `dataprofiler/resources/`. Code that referenced the old
  top-level path must be updated; use
  `dataprofiler.labelers.utils.find_resources_dir()` to resolve resource paths.

### Added

- Added compatibility support for NumPy 2.0 while constraining `numpy` to
  `>=1.22.0,<3.0.0` to avoid future breakage from NumPy 3.
- Added compatibility support for Keras versions newer than 3.4.0 while
  constraining `keras` to `>3.4.0,<4.0.0` to avoid future breakage from Keras 4.
- Added Python 3.12 and 3.13 to the test matrix.
- Added `find_resources_dir()` for resolving packaged resources via
  `importlib.resources`.

### Changed

- Replaced the deprecated `pkg_resources` usage with `importlib.resources`, and
  switched packaging from `data_files` to `package_data`.
- Constrained `chardet` to `<7.0.0` and `pandas` to `<3.0.0` to avoid breakage
  from upcoming major releases.
- Constrained `pyarrow` to `<24.0.0`.
- Simplified the `tensorflow` requirement to `tensorflow>=2.16.0`, removing the
  platform-specific `tensorflow-macos` marker.
- Updated the pre-commit configuration to align hook versions and hook
  dependencies with the current project requirements.
- Enabled `mypy` checking across the `dataprofiler` package (excluding tests).
- Removed `--forked` from the `tox` pytest invocation and raised `memray` to
  `>=1.18.0`.

### Security

- Resolved reported vulnerabilities by raising `boto3>=1.37.15`,
  `requests>=2.32.4`, and adding an explicit `urllib3>=2.5.0` floor.
