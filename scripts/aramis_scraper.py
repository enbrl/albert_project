"""Scraper for used-car listings on aramisauto.com.

It replaces the prototype of ``notebooks/Notebook 1.ipynb`` (the old URL and the
old CSS classes no longer exist on the website).

Rules followed by the scraper (see https://www.aramisauto.com/robots.txt):
- only the public listing pages ``/voitures/occasion/?page=N`` are requested;
- ``Crawl-delay: 5``: at least 5 seconds between two requests, so scraping
  pages takes time (about 10 minutes for 100 pages);
- every request has a timeout and identifies the project in its User-Agent.

Usage (from the root of the project):

    python -m scripts.aramis_scraper --pages 5 --output data/cars_raw.csv

The CSV has the same columns as the notebook: Nom, Carburant, Boite de vitesse,
Km, année, Prix. Km, année and Prix are integers (empty when not found).
"""

import argparse
import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Listing of used cars. The page number is given with the "page" parameter.
LISTING_URL = 'https://www.aramisauto.com/voitures/occasion/'

# Seconds to wait between two requests, as asked by the Crawl-delay of robots.txt
CRAWL_DELAY = 5

# Maximum time (in seconds) to wait for a server response
REQUEST_TIMEOUT = 15

HEADERS = {'User-Agent': 'albert_project (school project, scraping used-car listings politely)'}

COLUMNS = ['Nom', 'Carburant', 'Boite de vitesse', 'Km', 'année', 'Prix']


def parse_number(text):
    """
    Extract an integer from a text such as '11 559 km' or '21 299 €'.

    Spaces can be normal, non-breaking or narrow non-breaking spaces.

    Returns:
    - The integer, or None if the text has no digit
    """
    if not text:
        return None
    digits = re.sub(r'\D', '', text)  # keep only the digits
    return int(digits) if digits else None


def _text(card, selector):
    """Return the stripped text of the first element matching selector, or None."""
    element = card.select_one(selector)
    return element.get_text(' ', strip=True) if element else None


def parse_card(card):
    """
    Convert one vehicle card (<article class="vehicle-card">) into a dictionary.

    Returns:
    - A dictionary with the keys of COLUMNS, or None if the card has no name
    """
    name = _text(card, '.vehicle-properties__maker-model')
    if not name:
        return None

    # Specs look like "2025 • Essence • Manuelle" (year • fuel • gearbox [• range])
    specs = _text(card, '.vehicle-properties__specs') or ''
    parts = [part.strip() for part in specs.split('•')]
    fuel = parts[1] if len(parts) > 1 and parts[1] else None
    gearbox = parts[2] if len(parts) > 2 and parts[2] else None

    return {
        'Nom': name,
        'Carburant': fuel,
        'Boite de vitesse': gearbox,
        'Km': parse_number(_text(card, '.vehicle-properties__mileage')),
        'année': parse_number(parts[0]),
        'Prix': parse_number(_text(card, '.vehicle-card__selling-price')),
    }


def parse_listing_page(html):
    """
    Extract all the vehicles of one listing page.

    Parameters:
    - html: HTML content of the page

    Returns:
    - A list of dictionaries (one per vehicle), empty if the page has no vehicle
    """
    soup = BeautifulSoup(html, 'html.parser')
    rows = [parse_card(card) for card in soup.select('article.vehicle-card')]
    return [row for row in rows if row is not None]


def fetch_page(page, session=None):
    """
    Download one listing page.

    Parameters:
    - page: page number, starting at 1
    - session: optional requests.Session (reuses the connection)

    Returns:
    - The HTML of the page
    """
    getter = session.get if session is not None else requests.get
    params = {'page': page} if page > 1 else None  # page 1 is the page without parameter
    response = getter(LISTING_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()  # Stop with a clear error if the request failed
    return response.text


def scrape_listings(max_pages=5, delay=CRAWL_DELAY, sleep=time.sleep):
    """
    Scrape the first pages of the used-car listing.

    Parameters:
    - max_pages: number of pages to download (24 vehicles per page)
    - delay: seconds between two requests (never below CRAWL_DELAY)
    - sleep: function used to wait (replaced in the tests)

    Returns:
    - A list of dictionaries, one per vehicle
    """
    delay = max(delay, CRAWL_DELAY)  # robots.txt asks for at least 5 seconds
    rows = []
    with requests.Session() as session:
        for page in range(1, max_pages + 1):
            if page > 1:
                sleep(delay)  # be polite: wait before every request except the first
            page_rows = parse_listing_page(fetch_page(page, session))
            if not page_rows:  # no vehicle: we are after the last page (or the layout changed)
                break
            rows.extend(page_rows)
    return rows


def save_listings(rows, filename):
    """
    Save the scraped vehicles to a CSV file.

    Parameters:
    - rows: list of dictionaries returned by scrape_listings
    - filename: name of the CSV file to save

    Returns:
    - The saved DataFrame
    """
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.astype({'Km': 'Int64', 'année': 'Int64', 'Prix': 'Int64'})  # integers that can be empty

    folder = os.path.dirname(filename)
    if folder:  # Create the output folder (e.g. 'data/') if it does not exist yet
        os.makedirs(folder, exist_ok=True)

    df.to_csv(filename, index=False)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scrape used-car listings from aramisauto.com')
    parser.add_argument('--pages', type=int, default=5, help='number of pages to scrape (default: 5)')
    parser.add_argument('--output', default='data/cars_raw.csv', help='CSV file to write (default: data/cars_raw.csv)')
    args = parser.parse_args(argv)

    rows = scrape_listings(max_pages=args.pages)
    df = save_listings(rows, args.output)
    print(f'{len(df)} vehicles saved in {args.output}')
    return df


if __name__ == '__main__':
    main()
