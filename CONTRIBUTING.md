# Contributing

Thanks for your interest in improving **airbnb-price-prediction** — a teaching
companion to the *Let's Build Airbnb's Algorithm* series. Contributions that keep
the project a clear, honest learning resource are very welcome.

## Getting set up

```bash
git clone https://github.com/SilvioBaratto/airbnb-price-prediction
cd airbnb-price-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The cleaned snapshot `data/raw/listings_rome.csv` is committed, so every command
works offline. To rebuild it from the upstream dump:

```bash
airbnb fetch-data      # downloads ~22 MB from Inside Airbnb, then cleans it
```

## The development loop (test-first)

Every change earns a test. The loop we follow for each task:

1. **RED** — write a failing test that pins the behavior you want.
2. **GREEN** — write the minimum code to make it pass.
3. **Regression** — run the whole suite; keep it green.
4. **Check** — `ruff check .`, `ruff format .`, and `pyright`.
5. **Commit** — one focused commit per logical change.

```bash
pytest -q              # full suite (~20 s, never touches the network or the real snapshot)
ruff check . && ruff format --check .
pyright                # informational; a known pandas/numpy stub baseline is tolerated
```

## Architecture: the dependency rule

The package is organized in clean-architecture layers; **dependencies always
point inward** and inner layers never import outer ones:

```
cli  ->  application  ->  domain
 \                         ^
  \--> infrastructure -----/   (implements the application ports)
datasource, modeling  ->  domain + infrastructure   (supporting feature packages)
```

- `domain/` — pure rules and value objects (config, geo, entities). No I/O, no framework.
- `application/` — the pricing use case + ports (Protocols). Depends only on `domain`.
- `infrastructure/` — adapters (CSV repositories, predictors, paths) that implement
  the ports and do the I/O.
- `cli/` — the terminal entry points and the composition root that wires it all up.

The rule is not a convention: `tests/test_layering.py` parses every module's
imports and fails the build on a forbidden one.

When adding a capability, put pure logic in `domain`, an interface in
`application/ports.py`, and the concrete adapter in `infrastructure`.

## Numbers are never written by hand

Every number in the README and in the videos comes from a run of `airbnb run-arc`
on the committed snapshot at `SEED = 20260907`. If a change moves a number, re-run
the arc and update the README table from its output, not from memory.

## Commit & PR style

- Small, focused commits with an imperative subject (e.g. `Add CsvListingRepository`).
- Reference the issue you're closing in the PR description.
- Fill in the PR checklist; make sure `ruff` and `pytest` pass.

By contributing you agree that your work is licensed under the project's
[MIT License](LICENSE). The data keeps its own license (CC BY 4.0, see
[`data/sources/SOURCE.md`](data/sources/SOURCE.md)).
