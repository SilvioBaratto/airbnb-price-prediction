"""airbnb — predict the nightly price of a Rome listing from real Inside Airbnb data.

The cleaned snapshot lives in ``data/raw/listings_rome.csv``; ``airbnb fetch-data`` rebuilds it
from the upstream dump, ``airbnb run-arc`` runs the eight-part tree-ensemble arc on it, and
``airbnb simulate`` prices a listing a host describes.
"""

__version__ = "0.1.0"
