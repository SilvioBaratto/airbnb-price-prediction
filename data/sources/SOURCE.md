# Data sources

## 1. Inside Airbnb — Rome listings (primary)

- **Publisher:** [Inside Airbnb](https://insideairbnb.com/get-the-data/), an independent,
  non-commercial project that publishes scraped snapshots of public Airbnb listings.
- **Snapshot:** Rome (Lazio, Italy), scraped **2026-06-20**.
- **File fetched:** https://data.insideairbnb.com/italy/lazio/rome/2026-06-20/data/listings.csv.gz
- **Retrieved:** 2026-10-05.
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Format:** gzipped CSV, 37,084 listings × 90 columns.

### What is committed, and what is not

The raw dump is **not committed**: it carries host names, profile photos and URLs, and
free-text descriptions, i.e. personal data that this project has no reason to redistribute.
`airbnb fetch-data` downloads it to this folder (git-ignored) when it is missing.

The committed file is the derived snapshot **`data/raw/listings_rome.csv`** (31,870 rows),
built by `airbnb fetch-data` (`airbnb/datasource/insideairbnb.py`). It keeps the listing id, the
model's features and the price, and nothing about hosts:

| Rule | Listings dropped |
|---|---:|
| no price (not bookable when scraped) | 2,760 |
| price quoted for a stay longer than 7 nights (carries Airbnb's long-stay discount) | 2,072 |
| price outside 20–1000 EUR per night | 382 |
| **kept** | **31,870** |

### Notes on the columns

- **`price` is in EUR**, despite the `$` in the dump: it is a formatting artefact, and the
  `currency` field of `price_quote_raw` reads `EUR` for Rome.
- **`price` is a quote for one requested stay**, not a list price: Inside Airbnb asks Airbnb for
  the price of a specific check-in/check-out pair. That is why long-stay quotes are dropped.
- **`latitude`/`longitude` are anonymised by Airbnb** (shifted by up to ~150 m), so distances
  are accurate to the neighbourhood, not the doorstep.
- `bathrooms` is missing for ~15% of listings; the free-text `bathrooms_text` (`"1.5 baths"`,
  `"1 shared bath"`, `"Half-bath"`) fills the gaps and gives `bathroom_shared`.
- `instant_bookable` is empty for every listing in this snapshot, so it is not used.
- `neighbourhood` is Inside Airbnb's `neighbourhood_cleansed`: Rome's 15 *municipi*.

## 2. Places — hand-picked spots (secondary)

`rome_places.csv` lists 23 well-known spots (Colosseo, Trastevere, Termini, EUR, Ostia, ...)
with the approximate coordinates of their centre, typed by hand for the simulator. The
municipio each one belongs to is **not** written in the file: it is read from the 25 nearest
listings when the simulator starts, so it always matches the snapshot's vocabulary.

## 3. scikit-learn diabetes dataset (Part 8 only)

The *no free lunch* comparison in Part 8 also ranks the methods on scikit-learn's bundled
diabetes dataset (Efron, Hastie, Johnstone and Tibshirani, *Least Angle Regression*, Annals of
Statistics, 2004): 442 patients, 10 baseline variables, disease progression one year later. It
ships with scikit-learn (BSD-3-Clause), so no download is needed.

## Attribution

> Contains data from Inside Airbnb (https://insideairbnb.com), Rome, 20 June 2026, licensed
> under CC BY 4.0. Data was cleaned and reduced; see the rules above.
