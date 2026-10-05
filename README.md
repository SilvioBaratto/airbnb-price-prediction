# airbnb-price-prediction

[![CI](https://github.com/SilvioBaratto/airbnb-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/SilvioBaratto/airbnb-price-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](data/sources/SOURCE.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Companion project to the series **"Let's Build Airbnb's Algorithm"** (neuroespresso).
Predict what a listing in **Rome** should charge per night, from **31,870 real listings**
(Inside Airbnb), climbing from one decision tree to random forests and gradient boosting in
eight parts, and price your own listing from the terminal.

> Simplified for teaching: this is not Airbnb's Smart Pricing, which also sees demand,
> seasonality and the calendar. It is the core idea — learn a nightly price from what similar
> listings ask — rebuilt on public data as a project you can run.

```text
Place: trastevere
  -> Trastevere (I Centro Storico, 1.3 km from Piazza Venezia)
Room type (1 Entire home/apt, 2 Private room, 3 Hotel room, 4 Shared room) [1]:
Guests [2]:
Bedrooms [1]:
Beds [1]:
Bathrooms [1]:

Trastevere · Entire home/apt · 2 guests · 1 bedroom · 1 bed · 1 bathroom
Suggested price: €193 a night (80% of similar listings ask €141 – €279)

The nearest real listings like it:
Distance  Guests  Bedrooms  Baths  Price  Listing id
--------  ------  --------  -----  -----  -------------------
 0.03 km       2         1      1   €147  1410805100941638474
 0.05 km       2         1      1   €373  1331465
 0.07 km       2         1      1   €131  847573816608722974
 0.07 km       2         ?      1   €226  27675712
 0.08 km       2         1      1   €164  1249812475033796722
```

The spread of the real prices next door (€131 to €373 for the same flat, same street) is the
honest answer to "how good can a price model get": part of a price is the host, not the flat.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

airbnb run-arc --charts              # the eight parts on the committed snapshot (~2 min)
airbnb train                         # fit the price models once, save them to models/
airbnb simulate --model models/price_model.joblib   # price a listing interactively
```

Every command is also reachable as `python -m airbnb <subcommand>`; `airbnb --help` (or
`airbnb <subcommand> --help`) lists the options. The snapshot is committed, so nothing above
needs the network. `airbnb fetch-data` rebuilds it from the upstream dump.

### Train once, then serve fast

`airbnb train` fits three gradient-boosting models (the suggested price, and the 10th and 90th
percentile that bound the band) and saves them to `models/price_model.joblib` (~0.7 MB).
`airbnb simulate --model <path>` loads them instantly. Without `--model`, or if the file is
missing, the simulator trains on launch, which takes a couple of seconds on the snapshot. The
weights are not committed: joblib files are tied to the scikit-learn version that wrote them.

## How it works

1. pick a **place** among 23 well-known spots in Rome (by name or number), whose municipio is
   read from the listings around it;
2. pick a **room type** and say how many **guests, bedrooms, beds and bathrooms**;
3. compute the **distance from Piazza Venezia**; everything a host does not state (reviews,
   rating, amenities, availability) is filled with the value of a typical listing;
4. the **absolute-loss boosting model** suggests the price, two **quantile boosting models**
   give the band 80% of similar listings fall in;
5. the terminal prints the price, the band, and the **five nearest real listings** with the
   same room type and size, so the suggestion can be checked against the street.

Each price comes from a fitted model — never from a hand-written rule.

## Architecture

The code follows **clean architecture**: dependencies point inward and inner layers never
import outer ones. The pure rules sit at the center; I/O and the CLI live at the edges and are
wired together only in the composition root (`airbnb/cli/app.py`). `tests/test_layering.py`
parses every import and fails the build if the rule is broken.

```
        cli  ─────────▶  application  ─────────▶  domain
   (entry points,     (use case + ports,       (config, geo,
    composition)         Protocols)             entities)
        │                    ▲
        │                    │ implements ports
        └────────▶  infrastructure  (CSV repositories, predictors, paths)
                             ▲
     datasource ─────────────┘   modeling ─────▶  domain + infrastructure
   (download + clean the dump)  (the ML arc — supporting feature packages)
```

```
airbnb/
  __main__.py               python -m airbnb -> cli.app.main()
  domain/
    config.py               the problem: city, price slice, feature contract, seed
    geo.py                  haversine distance, distance from the centre
    entities.py             Place, ListingRequest, PriceQuote, Comparable
  application/
    ports.py                Protocols: PlaceRepository, ListingRepository, PricePredictor
    pricing.py              PricingService.quote(request) -> PriceQuote, .comparables(request)
  infrastructure/
    paths.py                where every file lives, the upstream URL
    repositories.py         CsvPlaceRepository, CsvListingRepository
    predictors.py           ModelPredictor (three boosters, save/load), MedianPredictor
  datasource/
    insideairbnb.py         download the dump, parse its formats, clean it into the snapshot
  modeling/
    pipeline.py             loading, the fixed split, preprocessing, model pipelines, metrics,
                            per-tree and out-of-bag helpers, the report
    arc.py                  the eight parts + orchestration
  cli/
    app.py                  the `airbnb` CLI (fetch-data | run-arc | train | simulate)
    simulator.py            the interactive loop
    render.py               the quote sentence and the comparables table
```

## The data

**Inside Airbnb, Rome, scraped 2026-06-20, CC BY 4.0.** The raw dump (90 columns, host names
and descriptions included) is downloaded on demand and git-ignored; the committed
`data/raw/listings_rome.csv` keeps only the id, the features and the price. Provenance, cleaning
rules and caveats are in [`data/sources/SOURCE.md`](data/sources/SOURCE.md).

| Column | Type | Note |
|---|---|---|
| `listing_id` | int | Airbnb's id, for traceability; never a feature |
| `accommodates` | float | guests |
| `bedrooms`, `beds` | float | `bedrooms` missing for 17% (imputed) |
| `bathrooms` | float | numeric column, gaps filled from `bathrooms_text` |
| `bathroom_shared` | 0/1 | from `bathrooms_text` ("1 shared bath") |
| `amenities_count` | float | length of the amenities list |
| `minimum_nights` | float | |
| `availability_365` | float | nights available in the next year |
| `number_of_reviews` | float | |
| `review_scores_rating` | float | 0–5, missing for 9% (imputed) |
| `host_is_superhost` | 0/1 | |
| `latitude`, `longitude` | float | anonymised by Airbnb (±150 m) |
| `km_to_center` | float | haversine distance from Piazza Venezia |
| `room_type` | cat | Entire home/apt, Private room, Hotel room, Shared room |
| `neighbourhood` | cat | one of Rome's 15 municipi |
| `price_eur` | € | **target**: nightly price quoted for a stay of at most 7 nights, 20–1000 € |

> ⚠️ Inside Airbnb's `price` is a **quote for a specific stay**, not a list price. Quotes for a
> month or more carry Airbnb's long-stay discount, so the snapshot keeps only short-stay quotes.

## The model, part by part

The arc teaches a **progression**: each part lowers the error of the suggested price (mean
absolute error, in euro per night, on 6,374 listings set aside before any part runs). Every
number below is printed by `airbnb run-arc` — never typed by hand.

| Part | Model / concept | Test MAE |
|---|---|---:|
| — | always answer the mean (200.52 €) | **87.13 €** |
| 1 | **regression tree (CART)**: the first cut is `bathrooms ≤ 1.75`; three levels, 8 leaves | 61.14 € |
| 2 | grown to 24,846 leaves it memorizes (train 0.00, test 67.19); **cost-complexity pruning** with α by 5-fold CV keeps 132 | 54.14 € |
| 3 | **bagging**: 200 trees on bootstrap resamples (each holds 63.2% of the listings); a lone tree scores 68.07 | 47.78 € |
| 4 | **random forest** (11 of 33 columns per split): the 50 bagged trees all open on `bathrooms`, the forest's open on seven different ones; tree correlation ρ 0.06 → 0.03 | 46.66 € |
| 5 | **out-of-bag error**: each listing graded by the 36.8% of trees that never saw it — OOB 47.23 vs test 46.66, no test set needed | — |
| 6 | **permutation importance** (OOB): shuffling `km_to_center` costs +21.06 €, `bathroom_shared` +0.22 €; split-gain ranks `bathrooms` first instead | — |
| 7 | **gradient boosting**, 8-leaf trees on the residuals: best at 398 rounds, worse by 3,000 (validation 48.62 → 50.00); with **absolute loss** | 46.65 € → **44.81 €** |
| 8 | **no free lunch**: the same 8 methods by 5-fold CV — Rome: boosting 1st, OLS 7th; scikit-learn's *diabetes* data: OLS 1st, boosting 3rd | — |

Part 8 brings back the previous series' methods (kNN, OLS, Lasso) and ranks everything on two
datasets. On Rome's prices, which depend on thresholds and interactions (a second bathroom
matters, distance matters more in the centre), trees win. On diabetes progression, which is
close to additive and linear, the straight line wins and boosting drops to third. Which method
is best depends on whether its assumptions match the data, not on how sophisticated it is.

Every part writes its curve to `output/` (`part2_pruning.csv` … `part8_nfl.csv`); `--charts`
adds a PNG next to each. The summary table lands in `output/arc_summary.csv`, and
`output/arc_run.json` records the seed and snapshot it came from.

### Fixtures for the videos

`scripts/export_fixtures.py` turns a run of the arc into the small JSON files the neuroespresso
videos animate (videocraft, data key `airbnb_prezzi`): one per scene family, from the depth-3
tree node by node to fifty trees' verdicts on one listing and the residuals shrinking round by
round. Curves and headline numbers are read from the arc's output, after checking its seed
and snapshot match; per-scene details are recomputed with the arc's own pipelines and seed.
Nothing is typed by hand.

```bash
airbnb run-arc                                   # once
python scripts/export_fixtures.py                # -> output/fixtures/airbnb_prezzi/
python scripts/export_fixtures.py --out-dir <videocraft>/public/fixtures/airbnb_prezzi
```

| File | Scene |
|---|---|
| `repo.json` | the project's layout, from `git ls-files` |
| `arc_metrics.json` | the summary table, plus the simulator's own test MAE and band coverage |
| `mappa.json` | 2,000 listing dots on Rome, the 23 places, each municipio's median price |
| `albero.json` | Part 1: the depth-3 tree, the error at each depth, 120 listings walking through it |
| `potatura.json` | Part 2: the memorizing full tree and the pruning path |
| `bagging.json` | Part 3: a ten-row bootstrap redraw, the error vs trees curve, fifty verdicts |
| `foresta.json` | Part 4: first cuts of bagged vs forest trees, ρ and σ², error vs `m` |
| `oob.json` | Part 5: which trees never drew one listing, and their verdict |
| `importanza.json` | Part 6: the permutation-importance ranking, a shuffled column |
| `boosting.json` | Part 7: error vs rounds, residuals of five listings round by round |
| `nfl.json` | Part 8: the two leaderboards |
| `simulatore.json` | the README's Trastevere quote and its comparables |

## ML project checklist → where it lives in the code

The project follows the eight-step **Machine Learning Project Checklist** (Aurélien Géron,
*Hands-On Machine Learning*, Appendix B).

**1. Frame the problem & look at the big picture** — predict a listing's nightly price (€) from
its location, type and size, and show the host a band and real comparables.
→ `airbnb/domain/config.py` (the price slice and the feature contract),
`airbnb/domain/entities.py` (`ListingRequest`, `PriceQuote`, `Comparable`).

**2. Get the data** *(automated and reproducible)* — download the Inside Airbnb dump and clean
it into a committed snapshot.
→ `airbnb fetch-data` = `airbnb/datasource/insideairbnb.py` (`download`, `clean_listings`,
`build_snapshot`); provenance in `data/sources/SOURCE.md`.

**3. Explore the data** — attribute types, roles and gaps are documented in *The data* above;
the arc's first parts are themselves an exploration (which cut comes first, which columns
matter).
→ `run_part1` (the first cut and the depth-3 rules), `run_part6` (permutation importance).

**4. Prepare the data** *(every transform is a reusable function)* — parse prices, bathrooms
and amenities; derive the distance from the centre; impute and one-hot inside the pipeline.
→ `airbnb/datasource/insideairbnb.py` (`parse_price`, `parse_bathrooms`, `count_amenities`);
`airbnb/modeling/pipeline.py` (`make_split` holds out the test set, `build_preprocessor`).

**5. Shortlist promising models** — tree, pruned tree, bagging, forest, boosting, and the
previous series' kNN/OLS/Lasso, compared on one split and by cross-validation.
→ `airbnb/modeling/arc.py` `run_part1` … `run_part4`, `run_part7`, `run_part8` (`lineup`).

**6. Fine-tune the system** *(choices on training rows only; the test set is only scored)* —
α by 5-fold CV, the number of boosting rounds on a validation slice, out-of-bag error as a free
estimate.
→ `run_part2`, `_tune_rounds`, `run_part5`.

**7. Present your solution** — the summary table, per-part CSVs and charts, and the
simulator's sentence and comparables table.
→ `pipeline.format_report`; `arc._plot_lines` / `_plot_bars`; `airbnb/cli/render.py`.

**8. Launch, monitor & maintain** — the models are trained once and persisted (`airbnb train`),
loaded to serve (`airbnb simulate --model`), or trained on launch as a fallback; a new
snapshot is one `airbnb fetch-data` away; unit tests and CI guard everything.
→ `airbnb/infrastructure/predictors.py` `ModelPredictor` (`save`/`load`); `airbnb/cli/app.py`
`run_train` / `run_simulate`; `tests/`; `.github/workflows/ci.yml`.

> Checklist source: Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn, Keras &
> TensorFlow*, Appendix B — "Machine Learning Project Checklist." Tree-ensemble theory follows
> Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, chapters 8–10 and 15.

## Development

```bash
pip install -e ".[dev]"
pytest -q                            # the full test suite (~20 s)
ruff check . && ruff format --check .
pyright                              # informational; a known pandas/numpy stub baseline is tolerated
```

Tests never read the committed snapshot nor the network: they build a few hundred synthetic
listings in memory, so the suite is fast and hermetic. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the test-first workflow and the dependency rule.

## License

Code released under the [MIT License](LICENSE). Data from
[Inside Airbnb](https://insideairbnb.com), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
[`data/sources/SOURCE.md`](data/sources/SOURCE.md).
