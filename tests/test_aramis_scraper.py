"""Tests of scripts/aramis_scraper.py (no network: requests.get is replaced)."""
from pathlib import Path

import pandas as pd
import pytest
import requests

from scripts import aramis_scraper
from scripts.aramis_scraper import (
    CRAWL_DELAY,
    COLUMNS,
    parse_listing_page,
    parse_number,
    save_listings,
    scrape_listings,
)

FIXTURE = (Path(__file__).parent / 'fixtures' / 'aramis_listing.html').read_text(encoding='utf-8')
EMPTY_PAGE = '<html><body><main></main></body></html>'


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error')


@pytest.fixture
def fake_site(monkeypatch):
    """Replace Session.get: page 1 and 2 have vehicles, the following ones are empty."""
    calls = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls.append({'url': url, 'params': params, 'headers': headers, 'timeout': timeout})
        page = (params or {}).get('page', 1)
        return FakeResponse(FIXTURE if page <= 2 else EMPTY_PAGE)

    monkeypatch.setattr(requests.Session, 'get', fake_get)
    return calls


def test_parse_number_handles_all_kinds_of_spaces():
    assert parse_number('11 559 km') == 11559  # narrow no-break space
    assert parse_number('21\xa0299 €') == 21299  # no-break space
    assert parse_number('86 318 km') == 86318
    assert parse_number('2017') == 2017


def test_parse_number_without_digit():
    assert parse_number('') is None
    assert parse_number(None) is None
    assert parse_number('Auto.') is None


def test_parse_listing_page_extracts_the_vehicles():
    rows = parse_listing_page(FIXTURE)
    assert len(rows) == 3  # the card without name is ignored
    assert rows[0] == {
        'Nom': 'Peugeot 208',
        'Carburant': 'Essence',
        'Boite de vitesse': 'Manuelle',
        'Km': 94410,
        'année': 2017,
        'Prix': 6999,
    }


def test_parse_listing_page_electric_car_keeps_gearbox():
    electric = parse_listing_page(FIXTURE)[1]
    assert electric['Carburant'] == 'Électrique'
    assert electric['Boite de vitesse'] == 'Auto.'
    assert electric['année'] == 2020
    assert electric['Prix'] == 16999


def test_parse_listing_page_missing_fields_are_none():
    incomplete = parse_listing_page(FIXTURE)[2]
    assert incomplete['Nom'] == 'MG ZS'
    assert incomplete['Prix'] is None
    assert incomplete['Km'] is None
    assert incomplete['année'] is None
    assert incomplete['Carburant'] is None


def test_parse_listing_page_without_vehicle():
    assert parse_listing_page(EMPTY_PAGE) == []


def test_scrape_listings_stops_on_empty_page_and_waits_between_requests(fake_site):
    waits = []
    rows = scrape_listings(max_pages=10, sleep=waits.append)
    assert len(rows) == 6  # 2 pages x 3 vehicles
    assert len(fake_site) == 3  # page 3 is empty: we stop there
    assert waits == [CRAWL_DELAY, CRAWL_DELAY]  # no wait before the first request


def test_scrape_listings_first_page_has_no_parameter(fake_site):
    scrape_listings(max_pages=2, sleep=lambda seconds: None)
    assert fake_site[0]['params'] is None
    assert fake_site[1]['params'] == {'page': 2}
    assert fake_site[0]['url'] == aramis_scraper.LISTING_URL
    assert 'User-Agent' in fake_site[0]['headers']
    assert fake_site[0]['timeout'] == aramis_scraper.REQUEST_TIMEOUT


def test_scrape_listings_never_goes_faster_than_the_crawl_delay(fake_site):
    waits = []
    scrape_listings(max_pages=2, delay=0, sleep=waits.append)
    assert waits == [CRAWL_DELAY]


def test_scrape_listings_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(requests.Session, 'get', lambda self, *args, **kwargs: FakeResponse('', 404))
    with pytest.raises(requests.HTTPError):
        scrape_listings(max_pages=1)


def test_save_listings_creates_folder_and_keeps_empty_values(tmp_path):
    destination = tmp_path / 'data' / 'cars.csv'  # 'data' does not exist yet
    df = save_listings(parse_listing_page(FIXTURE), str(destination))
    assert list(df.columns) == COLUMNS
    saved = pd.read_csv(destination)
    assert len(saved) == 3
    assert saved.loc[0, 'Prix'] == 6999
    assert pd.isna(saved.loc[2, 'Prix'])


def test_save_listings_without_vehicle(tmp_path):
    df = save_listings([], str(tmp_path / 'empty.csv'))
    assert df.empty
    assert list(df.columns) == COLUMNS
