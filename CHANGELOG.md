# Changelog

## Unreleased

- Added compatibility support for NumPy 2.0 while constraining `numpy` to
  `>=1.22.0,<3.0.0` to avoid future breakage from NumPy 3.
- Added compatibility support for Keras versions newer than 3.4.0 while
  constraining `keras` to `>3.4.0,<4.0.0` to avoid future breakage from Keras 4.
- Updated the pre-commit configuration to align hook versions and hook
  dependencies with the current project requirements.
