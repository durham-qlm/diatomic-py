# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Adopted semantic versioning, starting from 0.0.0
- Inherited functionality from previous diatom code by Jacob Blackmore
- Added `.gitignore` file, with python and OS level relevant ignores
- Existing molecules are now implemented with a dictionary lookup into a well-defined `Molecule` data structure
- Added structure for a test suite in `/tests`
- Added structure for documentation in `/docs` with sphinx
- Setup now with `pyproject.toml`
- Changelog standardised
- Pre-commit hooks, and github actions to enforce a black format, and ruff linter
- Moved to `/src` layout (as opposed to a 'flat' `/.` layout)
- Added README.md skeleton with basic instructions (maybe move to documentation?)

### Changed

- examples folder moved from `/example scripts` -> `/examples` to maintain naming consistency
