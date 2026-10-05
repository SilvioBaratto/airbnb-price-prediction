"""Business constants and the model's data contract (pure domain, no I/O).

Everything here is a decision about the *problem*, not about files or libraries: which city,
which slice of the listings counts as a "nightly price", which columns the model may read. The
feature lists are the single source of truth shared by the modeling arc and the pricing use
case, so a listing priced in the simulator is described exactly like a listing in training.
"""

from __future__ import annotations

SEED = 20260907

CITY = "Rome"

# Piazza Venezia, the conventional centre of Rome. Distances to it are what a guest means by
# "central", which the raw coordinates only express through an interaction a tree must learn.
CITY_CENTER_LAT = 41.8962
CITY_CENTER_LON = 12.4823
CITY_CENTER_NAME = "Piazza Venezia"

# The nightly-price slice the model learns. Below 20 EUR are placeholders and data-entry slips,
# above 1000 EUR are villas and event venues that a nightly-price model for ordinary stays
# should not chase. Together they drop about 1% of the priced listings.
PRICE_MIN_EUR = 20.0
PRICE_MAX_EUR = 1000.0

# Inside Airbnb prices each listing with a quote for one requested stay. Quotes for a month or
# more carry Airbnb's long-stay discount, so a "nightly price" is only comparable across
# listings when the quote is for a short stay.
MAX_QUOTE_NIGHTS = 7

# Airbnb's own room-type labels, most common in Rome first.
ROOM_TYPES = ("Entire home/apt", "Private room", "Hotel room", "Shared room")

TARGET = "price_eur"

NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "bathroom_shared",
    "amenities_count",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "host_is_superhost",
    "latitude",
    "longitude",
    "km_to_center",
]
CATEGORICAL_FEATURES = ["room_type", "neighbourhood"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The cleaned snapshot's column order: an id for traceability, the features, the target.
SNAPSHOT_COLUMNS = ["listing_id", *FEATURES, TARGET]
