# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
