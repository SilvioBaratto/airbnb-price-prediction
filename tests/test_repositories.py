"""CSV repositories: places labelled from the data, and nearby comparables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airbnb.application.ports import ListingRepository, PlaceRepository
from airbnb.domain.entities import ListingRequest
from airbnb.infrastructure.repositories import CsvListingRepository, CsvPlaceRepository


@pytest.fixture
def places(places_csv: Path, listings: pd.DataFrame) -> CsvPlaceRepository:
    """The three test places, labelled from the synthetic listings."""
    return CsvPlaceRepository(places_csv, listings=listings)


def test_adapters_satisfy_their_ports(places: CsvPlaceRepository, listings: pd.DataFrame) -> None:
    """Both repositories plug into the application's ports."""
    assert isinstance(places, PlaceRepository)
    assert isinstance(CsvListingRepository(listings=listings), ListingRepository)


def test_each_place_gets_the_municipio_of_its_surrounding_listings(
    places: CsvPlaceRepository,
) -> None:
    """The neighbourhood comes from the nearest listings, not from a hand-written table."""
    by_name = {p.name: p.neighbourhood for p in places.all()}
    assert by_name == {
        "Città Vecchia": "I Centro Storico",
        "Parioli": "II Parioli/Nomentano",
        "Ostia": "X Ostia/Acilia",
    }


def test_get_by_id_and_missing_id(places: CsvPlaceRepository) -> None:
    """Ids resolve, and an unknown id raises KeyError for the caller to handle."""
    assert places.get(2).name == "Parioli"
    with pytest.raises(KeyError):
        places.get(99)


@pytest.mark.parametrize("query", ["citta", "CITTÀ", " vecchia "])
def test_search_ignores_case_accents_and_padding(places: CsvPlaceRepository, query: str) -> None:
    """A host typing without accents or capitals still finds the place."""
    assert [p.name for p in places.search(query)] == ["Città Vecchia"]


def test_blank_search_matches_nothing(places: CsvPlaceRepository) -> None:
    """An empty query is not a wildcard."""
    assert places.search("   ") == []


def test_place_repository_reads_the_listings_file_when_no_frame_is_given(
    places_csv: Path, snapshot_csv: Path
) -> None:
    """The file-reading path labels places the same way."""
    repo = CsvPlaceRepository(places_csv, snapshot_csv)
    assert repo.get(3).neighbourhood == "X Ostia/Acilia"


def test_comparables_share_room_type_and_guests_nearest_first(
    places: CsvPlaceRepository, snapshot_csv: Path
) -> None:
    """Only same-type, same-size listings, sorted by distance, at most k of them."""
    repo = CsvListingRepository(snapshot_csv)
    request = ListingRequest(places.get(1), "Entire home/apt", 2, 1, 1, 1.0)
    found = repo.comparables(request, k=4)
    assert len(found) == 4
    assert all(c.room_type == "Entire home/apt" and c.guests == 2 for c in found)
    assert [c.km for c in found] == sorted(c.km for c in found)


def test_comparables_relax_the_guest_count_when_nothing_matches(
    places: CsvPlaceRepository, listings: pd.DataFrame
) -> None:
    """A size nobody offers still shows the nearest listings of that type."""
    repo = CsvListingRepository(listings=listings)
    request = ListingRequest(places.get(1), "Entire home/apt", 16, 8, 8, 4.0)
    found = repo.comparables(request, k=3)
    assert len(found) == 3
    assert all(c.room_type == "Entire home/apt" for c in found)
