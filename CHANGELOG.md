# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0]

### Added

- Added arbitrary-polarisation AC Stark support via `operators.ac_ham_ellip` and `operators.unit_ac_aniso_ellip`.
- Added a 3D polarisation ellipse plotting helper, `plotting.plot_polarization_ellipse`.
- Added support for non-zero minimum rotational basis levels through `SingletSigmaMolecule.Nmin` and the relevant basis/operator builders.
- Added `MN` basis-cropping helpers: `proj_iter_bounded`, `mn_crop_indices`, and `crop_by_indices`.
- Added optional chunked diagonalisation and progress bars to `calculate.solve_system` with `progress` and `chunk_size`.
- Added mixed-label warnings to `calculate.label_states`, with `warn_mixed`, `min_weight`, and cropped-basis `basis_idx` support.
- Added `to_states` selection to `calculate.transition_electric_moments`.
- Added the `Rb87Cs133-RR` rigid-rotor preset and additional RbCs polarisability entries.
- Added circular AC Stark and label-mixing warning examples.
- Added tests covering logging, chunked diagonalisation, state labelling warnings, basis cropping, transition sub-selection, and new operator behaviour.

### Changed

- Bumped the package version to `2.1.0` and the documentation release metadata to match.
- Raised the Python requirement to `>=3.11`.
- Added lower bounds to core runtime dependencies and committed a `uv.lock` file.
- Moved development-only dependencies into PEP 735 dependency groups; published extras are now limited to user-facing `plotting` and `progress` features.
- Updated installation documentation to distinguish package extras from local development dependency groups.
- Updated CI test installation to use `uv` dependency groups instead of the removed `test` extra.
- Reworked `calculate.sort_smooth` to use one-to-one assignment, avoiding duplicated state trajectories when overlaps are tied or nearly degenerate.
- Updated `sort_by_labels`, `magnetic_moment`, and `electric_moment` to handle both single-Hamiltonian and scan-shaped outputs.
- Updated nested timing logs so top-level timed calls log at `INFO`, nested calls log at `DEBUG`, and failures are logged before re-raising.
- Replaced deprecated `scipy.special.sph_harm` usage with `scipy.special.sph_harm_y`.
- Updated examples to use chunked/progress diagonalisation where useful and to avoid blocking `input()` calls.
- Updated the PyPI publish action to `pypa/gh-action-pypi-publish@v1.13.0`.

### Fixed

- Corrected molecule preset data for `K40Rb87` and `Na23Cs133`.
- Fixed transition electric moment sub-selection for circular transitions so selected rows/columns match the corresponding full transition matrix.
- Fixed basis-size and indexing calculations to respect `Nmin`.
- Fixed rotational plotting and rigid-rotor example basis iteration to respect the molecule basis limits.

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
