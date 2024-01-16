# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

### Added

- Adopted semantic versioning, starting now from 2.0.0
- Inherited functionality from previous diatom code by Jacob Blackmore
- Added `.gitignore` file, with python and OS level relevant ignores
- Existing molecules are now implemented with a dictionary lookup into a well-defined `SingletSigmaMolecule` data structure
- Added structure for a test suite in `/tests`
- Added structure for documentation in `/docs` with sphinx
- Changelog standardised
- Pre-commit hooks, and github actions to enforce a black format, and ruff linter
- Moved to `/src` layout (as opposed to a 'flat' `/.` layout)
- Added README.rst skeleton with basic instructions
- Plotting 2D/3D rotational structure to a given `ax`
- Added logger with function timer to see where code is using a lot of time

### Changed

- API redefined to consistency
- examples folder moved from `/example scripts` -> `/examples` to maintain naming consistency
- optimised many functions for performance.
- `hamiltonians` -> `operators` module name
- `constants` -> `systems` module name
- Label states now split into many functions
- Setup now with `pyproject.toml`

