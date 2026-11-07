"""Scraper for used-car listings on aramisauto.com.

It replaces the prototype of ``notebooks/Notebook 1.ipynb`` (the old URL and the
old CSS classes no longer exist on the website).

Rules followed by the scraper (see https://www.aramisauto.com/robots.txt):
- only the public listing pages ``/voitures/occasion/?page=N`` are requested;
- ``Crawl-delay: 5``: at least 5 seconds between two requests, so scraping
  pages takes time (about 10 minutes for the whole listing, around 110 pages);
- every request has a timeout and identifies the project in its User-Agent.

If the server answers with an error that can go away (429, 500, 502, 503, 504)
or if the connection fails, the request is tried again a few times, waiting
longer each time. The CSV file is saved after every page, so a problem on page
90 does not lose the first 89 pages.

Usage (from the root of the project):

    python -m scripts.aramis_scraper --pages 5 --output data/cars_raw.csv
    python -m scripts.aramis_scraper --all --output data/cars_raw.csv

If the scraping stops, the message tells which command to run to continue:

    python -m scripts.aramis_scraper --all --start-page 57 --append --output data/cars_raw.csv

The CSV has the columns of the notebook (Nom, Carburant, Boite de vitesse, Km,
année, Prix), the Id of the listing (the same car has the same Id every time,
which removes the duplicates) and the details of the engine: Motorisation
(for example '1.2 PureTech 68 BVM5'), Finition (for example 'Like') and
Puissance (power in horsepower, found in the text of the engine when it is
written there). Id, Km, année, Prix and Puissance are integers (empty when not
found).
"""

import argparse
import os
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.cli import UserError, positive_int, run

# Listing of used cars. The page number is given with the "page" parameter.
LISTING_URL = 'https://www.aramisauto.com/voitures/occasion/'

# Seconds to wait between two requests, as asked by the Crawl-delay of robots.txt
CRAWL_DELAY = 5

# Maximum time (in seconds) to wait for a server response
REQUEST_TIMEOUT = 15

# Errors that can go away if we try again, number of tries and waiting time between them
RETRY_STATUSES = (429, 500, 502, 503, 504)
RETRY_TOTAL = 3
RETRY_BACKOFF = 5

# With --all, stop after this number of pages even if the site still answers (safety)
MAX_PAGES = 500

HEADERS = {'User-Agent': 'albert_project (school project, scraping used-car listings politely)'}

COLUMNS = ['Id', 'Nom', 'Carburant', 'Boite de vitesse', 'Km', 'année', 'Prix', 'Motorisation', 'Finition', 'Puissance']
INTEGER_COLUMNS = ['Id', 'Km', 'année', 'Prix', 'Puissance']

# Power below and above these values is not a power (it is another number of the text)
MIN_POWER = 30
MAX_POWER = 1000


class ScrapingStopped(Exception):
    """The scraping stopped on a page. next_page is the page to give to --start-page to continue."""

    def __init__(self, next_page, cause):
        super().__init__(f'stopped on page {next_page}: {cause}')
        self.next_page = next_page


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


def parse_power(engine, name=''):
    """
    Find the power (in horsepower) in the text of the engine.

    Examples: '1.2 PureTech 68 BVM5' -> 68, '2008 Hybrid 145 ch e-DCS6' -> 145,
    '50 kWh - 136' -> 136, '75 kWh - RWD' -> None.

    The number of the model (for example 2008 in 'Peugeot 2008'), the displacement
    (1.2), the battery (50 kWh) and the codes stuck to letters (BVM5, 16V) are ignored.

    Parameters:
    - engine: text of the engine
    - name: name of the car, to ignore the number of the model

    Returns:
    - The power as an integer, or None if it is not written
    """
    if not engine:
        return None
    text = re.sub(r'\b\d+[.,]\d+\b', ' ', engine)  # displacement: 1.2, 1,6
    text = re.sub(r'\b\d+\s*kWh\b', ' ', text, flags=re.IGNORECASE)  # battery: 50 kWh
    explicit = re.search(r'\b(\d{2,3})\s*ch\b', text, flags=re.IGNORECASE)  # '145 ch'
    if explicit:
        candidates = [explicit.group(1)]
    else:
        model_numbers = set(re.findall(r'\d+', name or ''))
        candidates = [n for n in re.findall(r'(?<![\w.])(\d{2,3})(?![\w.])', text) if n not in model_numbers]
    for candidate in candidates:
        if MIN_POWER <= int(candidate) <= MAX_POWER:
            return int(candidate)
    return None


def parse_id(card):
    """
    Return the Id of the listing, read in the link of the card (...?vehicleId=1036829), or None.
    """
    link = card.select_one('a[href]')
    match = re.search(r'vehicleId=(\d+)', link['href']) if link else None
    return int(match.group(1)) if match else None


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

    # Engine and trim look like "1.2 PureTech 68 BVM5 • Like"
    engine_parts = [part.strip() for part in (_text(card, '.vehicle-properties__engine-trim') or '').split('•')]
    engine = engine_parts[0] or None
    trim = ' • '.join(part for part in engine_parts[1:] if part) or None

    return {
        'Id': parse_id(card),
        'Nom': name,
        'Carburant': fuel,
        'Boite de vitesse': gearbox,
        'Km': parse_number(_text(card, '.vehicle-properties__mileage')),
        'année': parse_number(parts[0]),
        'Prix': parse_number(_text(card, '.vehicle-card__selling-price')),
        'Motorisation': engine,
        'Finition': trim,
        'Puissance': parse_power(engine, name),
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


def make_session(backoff_factor=RETRY_BACKOFF):
    """
    Create a requests.Session that tries again when the server is busy or the connection fails.

    Parameters:
    - backoff_factor: waiting time (in seconds) that grows after each failed try (0 in the tests)
    """
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=backoff_factor,
        status_forcelist=RETRY_STATUSES,
        allowed_methods=['GET'],
        respect_retry_after_header=True,  # if the server says "come back in 30 s", we wait
    )
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retry))
    session.mount('http://', HTTPAdapter(max_retries=retry))
    return session


def fetch_page(page, session=None, url=LISTING_URL):
    """
    Download one listing page.

    Parameters:
    - page: page number, starting at 1
    - session: optional requests.Session (reuses the connection)
    - url: address of the listing

    Returns:
    - The HTML of the page
    """
    getter = session.get if session is not None else requests.get
    params = {'page': page} if page > 1 else None  # page 1 is the page without parameter
    response = getter(url, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()  # Stop with a clear error if the request failed
    return response.text


def scrape_listings(
    max_pages=5, delay=CRAWL_DELAY, sleep=None, start_page=1, session=None, on_page=None, url=LISTING_URL
):
    """
    Scrape the pages of the used-car listing.

    Parameters:
    - max_pages: number of pages to download (24 vehicles per page)
    - delay: seconds between two requests (never below CRAWL_DELAY)
    - sleep: function used to wait (time.sleep by default, replaced in the tests)
    - start_page: first page to download (to continue a scraping that stopped)
    - session: requests.Session to use (by default one that tries again on errors)
    - on_page: function called after each page with (page_number, rows_of_the_page, all_rows)
    - url: address of the listing

    Returns:
    - A list of dictionaries, one per vehicle (a car seen twice is kept once)

    Raises:
    - ScrapingStopped if a page could not be downloaded; the vehicles of the previous pages
      were already given to on_page
    """
    delay = max(delay, CRAWL_DELAY)  # robots.txt asks for at least 5 seconds
    sleep = sleep or time.sleep
    rows = []
    seen_ids = set()
    own_session = session is None
    session = session or make_session()
    try:
        for page in range(start_page, start_page + max_pages):
            if page > start_page:
                sleep(delay)  # be polite: wait before every request except the first
            try:
                page_rows = parse_listing_page(fetch_page(page, session, url))
            except requests.HTTPError as error:
                if page > start_page and error.response is not None and error.response.status_code == 404:
                    break  # after the last page the site answers "not found"
                raise ScrapingStopped(page, error) from error
            except requests.RequestException as error:
                raise ScrapingStopped(page, error) from error
            if not page_rows:  # no vehicle: we are after the last page (or the layout changed)
                break
            # The listing moves while we scrape: a car can appear on two pages
            new_rows = [row for row in page_rows if row['Id'] is None or row['Id'] not in seen_ids]
            seen_ids.update(row['Id'] for row in new_rows if row['Id'] is not None)
            rows.extend(new_rows)
            if on_page is not None:
                on_page(page, new_rows, rows)
    finally:
        if own_session:
            session.close()
    return rows


def save_listings(rows, filename):
    """
    Save the scraped vehicles to a CSV file (a listing with the same Id is kept once).

    Parameters:
    - rows: list of dictionaries returned by scrape_listings
    - filename: name of the CSV file to save

    Returns:
    - The saved DataFrame
    """
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.astype({column: 'Int64' for column in INTEGER_COLUMNS})  # integers that can be empty
    df = df[df['Id'].isna() | ~df['Id'].duplicated()]  # same Id = same car

    folder = os.path.dirname(filename)
    if folder:  # Create the output folder (e.g. 'data/') if it does not exist yet
        os.makedirs(folder, exist_ok=True)

    df.to_csv(filename, index=False)
    return df


def load_listings(filename):
    """Read a CSV made by save_listings and return it as a list of dictionaries (empty if the file does not exist)."""
    if not os.path.exists(filename):
        return []
    rows = []
    for record in pd.read_csv(filename).to_dict('records'):
        row = {}
        for column, value in record.items():
            if pd.isna(value):
                row[column] = None  # empty value
            elif column in INTEGER_COLUMNS:
                row[column] = int(value)
            else:
                row[column] = value
        for column in COLUMNS:  # a file made by an older version has fewer columns
            row.setdefault(column, None)
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scrape used-car listings from aramisauto.com')
    parser.add_argument('--pages', type=positive_int, default=5, help='number of pages to scrape (default: 5)')
    parser.add_argument('--all', action='store_true', help='scrape all the pages, until the end of the listing')
    parser.add_argument('--start-page', type=positive_int, default=1, help='first page to scrape (default: 1)')
    parser.add_argument('--append', action='store_true', help='add to the output file instead of replacing it')
    parser.add_argument('--output', default='data/cars_raw.csv', help='CSV file to write (default: data/cars_raw.csv)')
    args = parser.parse_args(argv)

    previous = load_listings(args.output) if args.append else []
    known_ids = {row['Id'] for row in previous if row['Id'] is not None}
    saved = {'rows': previous}

    def save_progress(page, page_rows, all_rows):
        # The CSV is rewritten after every page: the previous rows + the cars not already in the file
        new = [row for row in all_rows if row['Id'] is None or row['Id'] not in known_ids]
        saved['rows'] = previous + new
        save_listings(saved['rows'], args.output)
        print(f'page {page}: {len(page_rows)} vehicles ({len(saved["rows"])} in total)', flush=True)

    max_pages = MAX_PAGES if args.all else args.pages
    try:
        scrape_listings(max_pages=max_pages, start_page=args.start_page, on_page=save_progress)
    except ScrapingStopped as error:
        print(f'Scraping {error}.', file=sys.stderr)
        print('The pages before were saved. To continue:', file=sys.stderr)
        print(
            f'  python -m scripts.aramis_scraper --all --start-page {error.next_page} --append --output {args.output}',
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    if not saved['rows']:  # never replace a good file by an empty one
        raise UserError(f'No vehicle was found: the website probably changed its pages. {args.output} was not changed.')
    df = save_listings(saved['rows'], args.output)
    print(f'{len(df)} vehicles saved in {args.output}')
    return df


if __name__ == '__main__':
    run(main)
