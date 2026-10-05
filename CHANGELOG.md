# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `scripts/export_fixtures.py` — exports twelve JSON fixtures for the
  neuroespresso video animations (`airbnb_prezzi`), checked against the arc
  run they quote.
- `airbnb run-arc` also writes `arc_summary.csv` and `arc_run.json`, a manifest
  of the finished run (snapshot SHA-256, seed, split, ensemble sizes, and the
  SHA-256 of every CSV it wrote).

### Changed

- `paths.py` moved from `airbnb/infrastructure/` to `airbnb/`, a leaf module:
  `modeling` no longer imports `infrastructure`, which closed a package-level
  cycle. The layering test now forbids mutual imports and sees relative and
  `from airbnb import x` imports.
- Part 4's tree correlation is now the unbiased one-way ANOVA estimate,
  averaged over four slicings with its spread: ρ 0.083 (bagging) and 0.046
  (forest), where the old estimator read 0.058 and 0.029.
- Part 8's summary row no longer puts its cross-validated MAE in the held-out
  test column.

### Fixed

- The simulator exits cleanly on end of input (Ctrl-D, a finished pipe) and on
  Ctrl-C, instead of raising a traceback.
- The simulator refuses non-whole guests, bedrooms and beds ("2.5" was silently
  read as 2) and NaN or infinity (which crashed it).
- `run_part7` no longer crashes when given fewer than 100 rounds.
- The fixture export refuses an arc run with another snapshot, split or ensemble
  size (it only checked seed and row count) and any CSV altered after the run,
  ships pruning alphas with
  significant digits (12 of 20 were rounded to 0), and writes strict JSON.
- README and CONTRIBUTING figures re-measured: model ~1 MB, test suite ~40 s.

## [0.1.0] - 2026-10-05

### Added

- **Real data pipeline** — `airbnb fetch-data` downloads the Inside Airbnb dump
  for Rome (2026-06-20, CC BY 4.0) and cleans it into a committed, host-free
  snapshot of 31,870 listings priced for a short stay.
- **The eight-part tree-ensemble arc** — Parts 1–8 (CART regression tree,
  cost-complexity pruning, bagging, random forest and tree decorrelation,
  out-of-bag error, permutation importance, gradient boosting, no free lunch)
  with per-part CSVs, optional charts and a summary report (`airbnb run-arc`).
- **Interactive price simulator** — pick a place in Rome, a room type and a size;
  get a suggested nightly price, the band 80% of similar listings fall in, and
  the nearest real listings like it (`airbnb simulate`).
- **Persisted models** — `airbnb train` saves the three boosting models;
  `airbnb simulate --model` reloads them instead of training on launch.
- **Unified `airbnb` CLI** with `fetch-data | run-arc | train | simulate` and a
  `python -m airbnb` entry point.
- Clean-architecture layout (`domain` / `application` / `infrastructure` / `cli`
  plus `datasource` and `modeling` feature packages) with ports & adapters, and
  a test that enforces the dependency rule.
- Project tooling: packaging metadata, `ruff` lint + format, a `pyright` config,
  a GitHub Actions CI workflow, and issue/PR templates.

[Unreleased]: https://github.com/SilvioBaratto/airbnb-price-prediction/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SilvioBaratto/airbnb-price-prediction/releases/tag/v0.1.0
