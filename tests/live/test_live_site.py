"""Check that the scraper still understands the real aramisauto.com website.

These tests use the network (about 2 requests), so they are NOT run by default.
Run them with:

    python -m pytest -m live

GitHub runs them once a week (.github/workflows/site-check.yml): if the website
changes its pages, the scraper reads nothing or empty values and a test fails,
before somebody scrapes for nothing.
"""

import time
from urllib.robotparser import RobotFileParser

import pytest
import requests

from scripts import aramis_scraper
from scripts.aramis_scraper import fetch_page, parse_listing_page

pytestmark = pytest.mark.live

# On a listing page, at least this share of the cars must have each field
MIN_SHARE = 0.9
# The fields that must always be found (the Id only exists in the more recent versions of the scraper)
FIELDS = [
    c for c in ['Id', 'Nom', 'Carburant', 'Boite de vitesse', 'Km', 'année', 'Prix'] if c in aramis_scraper.COLUMNS
]


def _identify(row):
    """What tells two cars apart: the Id, or (without Id) the name, mileage, year and price."""
    return row.get('Id') or (row['Nom'], row['Km'], row['année'], row['Prix'])


@pytest.fixture(scope='module')
def first_page_rows():
    return parse_listing_page(fetch_page(1))


def test_the_first_page_has_cars(first_page_rows):
    # 24 cars per page at the time of writing: far fewer means the pages changed
    assert len(first_page_rows) >= 20, 'The scraper found almost no car: the website probably changed its pages'


@pytest.mark.parametrize('field', FIELDS)
def test_each_field_is_found(first_page_rows, field):
    found = sum(row[field] is not None for row in first_page_rows)
    assert found >= MIN_SHARE * len(first_page_rows), f'{field} is empty for too many cars: the website changed?'


def test_values_are_realistic(first_page_rows):
    for row in first_page_rows:
        if row['Prix'] is not None:
            assert 500 <= row['Prix'] <= 500_000, row
        if row['Km'] is not None:
            assert 0 <= row['Km'] <= 1_000_000, row
        if row['année'] is not None:
            assert 1990 <= row['année'] <= time.localtime().tm_year + 1, row


def test_the_second_page_has_other_cars(first_page_rows):
    time.sleep(aramis_scraper.CRAWL_DELAY)  # robots.txt: 5 seconds between two requests
    second = parse_listing_page(fetch_page(2))
    assert second, 'No car on the second page'
    first_cars = {_identify(row) for row in first_page_rows}
    assert (
        len({_identify(row) for row in second} & first_cars) <= 3
    )  # the listing can move a little between two requests


def test_robots_txt_still_allows_our_pages():
    time.sleep(aramis_scraper.CRAWL_DELAY)
    response = requests.get(
        'https://www.aramisauto.com/robots.txt', headers=aramis_scraper.HEADERS, timeout=aramis_scraper.REQUEST_TIMEOUT
    )
    response.raise_for_status()
    robots = RobotFileParser()
    robots.parse(response.text.splitlines())

    agent = aramis_scraper.HEADERS['User-Agent']
    assert robots.can_fetch(agent, aramis_scraper.LISTING_URL), 'robots.txt now forbids the listing: stop scraping'
    assert robots.can_fetch(agent, aramis_scraper.LISTING_URL + '?page=2'), 'robots.txt now forbids the next pages'
    delay = robots.crawl_delay(agent)
    assert delay is None or delay <= aramis_scraper.CRAWL_DELAY, (
        f'robots.txt now asks {delay} seconds between requests, the scraper waits {aramis_scraper.CRAWL_DELAY}'
    )
