# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-18

### Added
- `xera.loom.functional` module with activation and utility functions, exposed via the `F` alias.
- `auto_flash_attention` (AutoFA) core primitive with automatic backend selection across TPU/GPU, including TPU shape validation and GPU compute-capability checks with fallback behavior.
- `XeraNaiveFlash` attention implementation and `xenafl_attention`, with fallback handling for unsupported attention requests.
- `xera.loom.flash_attention` subpackage, separating flash-attention implementations from the rest of `xera.loom`.
- Additional normalization layers, convolution operations, pooling operations, embedding and rotary embedding support, and `Conv`/`SSM` layers in `xera.loom`.
- `Partition` optimizer and additional optimizer wrappers in `xera.weave`.
- Model/state serialization upgraded from pickle to `safetensors`, plus `Weave.Callback` support.
- `XeraWarning` for surfacing backend/runtime warnings.
- Comprehensive unit test suite covering core, initializers, loom (attention, combinators, conv, embedding, functional, linear, normalization, pooling, recurrent, stochastic, transformer, flash attention), optimizer partitioning, serialization, and weave (callbacks, loss, metrics, optimizer core/wrapper, train loop).
- GitHub Actions workflow for building and publishing the package to PyPI on release.
- CI workflow to run the test suite and build distributions on pushes, pull requests, and tags.

### Changed
- Renamed the `training` parameter to `deterministic` in normalization layers for clarity.
- Refactored combinator functions for layer calls.
- Refactored `auto_flash_attention` to build on `xera.core`, and reworked LSE handling and block specifications.
- Refactored serialization test cases and optimizer-wrapper tests to match backend changes.
- Updated the installation command and API section of the README.

### Fixed
- Fixed an invalid import syntax issue.
- Fixed the `fori_loop` implementation used by `Loop`.
- Fixed GPU compute-capability checks for the Triton backend.
- Removed dead code paths.

### Removed
- Removed the naive Pallas flash-attention implementation in favor of `XeraNaiveFlash`.
- Removed a stale test for `xera` module API aliases.

## [0.0.2-alpha] - 2025

Initial pre-alpha release of Xera, a neural network library in JAX.

[0.1.0]: https://github.com/arunika-labs/xera/compare/v0.0.2-alpha...v0.1.0
[0.0.2-alpha]: https://github.com/arunika-labs/xera/releases/tag/v0.0.2-alpha
