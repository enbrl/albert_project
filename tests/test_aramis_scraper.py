"""Tests of scripts/aramis_scraper.py (no access to the real website)."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest
import requests

from scripts import aramis_scraper
from scripts.aramis_scraper import (
    CRAWL_DELAY,
    COLUMNS,
    ScrapingStopped,
    load_listings,
    main,
    make_session,
    parse_listing_page,
    parse_number,
    parse_battery,
    parse_power,
    save_listings,
    scrape_listings,
)

FIXTURE = (Path(__file__).parent / 'fixtures' / 'aramis_listing.html').read_text(encoding='utf-8')
EMPTY_PAGE = '<html><body><main></main></body></html>'


def fixture_page(number):
    """The page of the fixture, with other listing Ids for each page (like the real website)."""
    return FIXTURE.replace('vehicleId=10368', f'vehicleId={number}0368')


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error', response=self)


@pytest.fixture
def fake_site(monkeypatch):
    """Replace Session.get: page 1 and 2 have vehicles, the following ones are empty."""
    calls = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls.append({'url': url, 'params': params, 'headers': headers, 'timeout': timeout})
        page = (params or {}).get('page', 1)
        return FakeResponse(fixture_page(page) if page <= 2 else EMPTY_PAGE)

    monkeypatch.setattr(requests.Session, 'get', fake_get)
    return calls


@pytest.fixture
def no_wait(monkeypatch):
    """The command line must not really wait 5 seconds between pages in the tests."""
    monkeypatch.setattr(aramis_scraper.time, 'sleep', lambda seconds: None)


# ---------------------------------------------------------------- parsing


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
        'Id': 1036829,
        'Nom': 'Peugeot 208',
        'Carburant': 'Essence',
        'Boite de vitesse': 'Manuelle',
        'Km': 94410,
        'année': 2017,
        'Prix': 6999,
        'Motorisation': '1.2 PureTech 68 BVM5',
        'Finition': 'Like',
        'Puissance': 68,
    }


def test_parse_listing_page_reads_the_id_of_the_listing():
    assert [row['Id'] for row in parse_listing_page(FIXTURE)] == [1036829, 1036830, None]


@pytest.mark.parametrize(
    'engine, name, expected',
    [
        ('1.2 PureTech 68 BVM5', 'Peugeot 208', 68),  # displacement and gearbox code ignored
        ('1.5 dCi 100 BVM6', 'Renault Clio', 100),
        ('2008 Hybrid 145 ch e-DCS6', 'Peugeot 2008', 145),  # explicit 'ch'
        ('Hybrid 145 e-DCS6', 'Peugeot 3008', 145),
        ('1.5 L Hybrid+ 197 ch', 'MG ZS', 197),
        ('BlueHDi 130ch S&S EAT8', 'Peugeot 308', 130),  # 'ch' stuck to the number
        ('1.6 HDI 16V 92', 'Citroen C4', 92),  # 16V is a code, not a power
        ('50 kWh - 136', 'Peugeot 2008', 136),  # battery ignored
        ('75 kWh - RWD', 'Tesla Model 3', None),  # no power written
        ('208 1.2 PureTech', 'Peugeot 208', None),  # 208 is the model, not a power
        ('PureTech 5', 'Peugeot 208', None),  # one digit
        # Audi writes a power class before the horsepower: the 40 of '40 TFSIe' is not horsepower
        ('40 TFSIe 204 S tronic 6', 'Audi A3 Sportback', 204),
        ('35 TDI 150 S tronic 7', 'Audi Q2', 150),
        ('55 TFSI 340 Tiptronic 8 Quattro 5pl', 'Audi Q7', 340),
        ('35 TFSI 150S tronic 7', 'Audi Q2', 150),  # 'S tronic' stuck to the number
        ('35 TFSI CoD 150', 'Audi A3', 150),
        ('45 TFSI e 245 S tronic 6', 'Audi Q3', 245),
        # a hybrid writes its engine and its electric motor: the engine is kept
        ('2.0 T8 Twin Engine 303 + 87 ch Geartronic 8', 'Volvo V60', 303),
        ('T6 Hybride Rechargeable 253 + 145 AWD Geartronic 8', 'Volvo XC60', 253),
        ('1.5 T5 Recharge 180+82 DCT7', 'Volvo XC40', 180),
        ('Cooper SE 125 + 95 ALL4 BVA6', 'Mini Mini', 125),
        # what already worked must not change
        ('2.0 ECOBLUE 105 BVM6 280 L1H1', 'Ford Transit', 105),  # 280 is the model of the van
        ('1.5 TSI 150 DSG7', 'Volkswagen Golf', 150),  # Volkswagen writes the displacement first
        ('2.0 TDI 150 DSG7', 'Volkswagen Passat', 150),
        ('1.5 L Hybrid+ 197 ch', 'MG ZS', 197),
        ('', 'Peugeot 208', None),
        (None, 'Peugeot 208', None),
    ],
)
def test_parse_power(engine, name, expected):
    assert parse_power(engine, name) == expected


def test_parse_power_ignores_impossible_values():
    assert parse_power('Moteur 12') is None
    assert parse_power('Moteur 2500') is None  # four digits
    assert parse_power('Type 999 Special') == 999
    assert parse_power('Type 20') is None  # below MIN_POWER


def test_parse_listing_page_electric_car_keeps_gearbox():
    electric = parse_listing_page(FIXTURE)[1]
    assert electric['Carburant'] == 'Électrique'
    assert electric['Boite de vitesse'] == 'Auto.'
    assert electric['année'] == 2020
    assert electric['Prix'] == 16999
    assert electric['Motorisation'] == '50 kWh - 136'
    assert electric['Finition'] == 'GT-Line'
    assert electric['Puissance'] == 136


def test_parse_listing_page_missing_fields_are_none():
    incomplete = parse_listing_page(FIXTURE)[2]
    assert incomplete['Nom'] == 'MG ZS'
    assert incomplete['Prix'] is None
    assert incomplete['Km'] is None
    assert incomplete['année'] is None
    assert incomplete['Carburant'] is None
    assert incomplete['Motorisation'] is None
    assert incomplete['Finition'] is None
    assert incomplete['Puissance'] is None


def test_parse_listing_page_without_vehicle():
    assert parse_listing_page(EMPTY_PAGE) == []


# ---------------------------------------------------------------- scraping with the fake site


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


def test_scrape_listings_can_start_from_another_page(fake_site):
    scrape_listings(max_pages=2, start_page=2, sleep=lambda seconds: None)
    assert [call['params'] for call in fake_site] == [{'page': 2}, {'page': 3}]


def test_scrape_listings_keeps_a_car_seen_on_two_pages_once(monkeypatch):
    monkeypatch.setattr(requests.Session, 'get', lambda self, *args, **kwargs: FakeResponse(FIXTURE))
    rows = scrape_listings(max_pages=2, sleep=lambda seconds: None)
    # The 2 cars with an Id are kept once; the car without Id cannot be recognised, it stays
    assert sorted(str(row['Id']) for row in rows) == ['1036829', '1036830', 'None', 'None']


def test_scrape_listings_calls_on_page_after_each_page(fake_site):
    calls = []
    scrape_listings(
        max_pages=5,
        sleep=lambda seconds: None,
        on_page=lambda page, page_rows, all_rows: calls.append((page, len(page_rows), len(all_rows))),
    )
    assert calls == [(1, 3, 3), (2, 3, 6)]


def test_scrape_listings_error_tells_the_page_to_continue_from(monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        page = (params or {}).get('page', 1)
        return FakeResponse(fixture_page(page) if page < 3 else '', 500 if page == 3 else 200)

    monkeypatch.setattr(requests.Session, 'get', fake_get)
    saved = []
    with pytest.raises(ScrapingStopped) as error:
        scrape_listings(
            max_pages=10, sleep=lambda seconds: None, on_page=lambda page, rows, all_rows: saved.append(page)
        )
    assert error.value.next_page == 3
    assert isinstance(error.value.__cause__, requests.HTTPError)
    assert saved == [1, 2]  # the first two pages were given to on_page before the error


def test_scrape_listings_error_on_the_first_page(monkeypatch):
    monkeypatch.setattr(requests.Session, 'get', lambda self, *args, **kwargs: FakeResponse('', 404))
    with pytest.raises(ScrapingStopped) as error:
        scrape_listings(max_pages=1)
    assert error.value.next_page == 1


def test_scrape_listings_404_after_the_first_page_is_the_end_of_the_listing(monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        page = (params or {}).get('page', 1)
        return FakeResponse(fixture_page(1), 200) if page == 1 else FakeResponse('', 404)

    monkeypatch.setattr(requests.Session, 'get', fake_get)
    assert len(scrape_listings(max_pages=5, sleep=lambda seconds: None)) == 3


# ---------------------------------------------------------------- retries, with a real (local) server


class FlakyServer:
    """A local web server whose first answers are errors (statuses), then the fixture page."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests += 1
                status = outer.statuses.pop(0) if outer.statuses else 200
                body = (FIXTURE if status == 200 else 'error').encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # no noise in the tests

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}/'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def flaky_server():
    servers = []

    def start(statuses):
        servers.append(FlakyServer(statuses))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def test_retries_when_the_server_is_busy(flaky_server):
    server = flaky_server([503, 429])  # two errors, then it works
    rows = scrape_listings(max_pages=1, session=make_session(backoff_factor=0), url=server.url)
    assert len(rows) == 3
    assert server.requests == 3


def test_gives_up_after_several_errors(flaky_server):
    server = flaky_server([503] * 10)
    with pytest.raises(ScrapingStopped) as error:
        scrape_listings(max_pages=1, session=make_session(backoff_factor=0), url=server.url)
    assert error.value.next_page == 1
    assert server.requests == aramis_scraper.RETRY_TOTAL + 1  # the first try + the retries


def test_does_not_retry_an_error_that_will_not_go_away(flaky_server):
    server = flaky_server([404])
    with pytest.raises(ScrapingStopped):
        scrape_listings(max_pages=1, session=make_session(backoff_factor=0), url=server.url)
    assert server.requests == 1


# ---------------------------------------------------------------- saving and loading


def test_save_listings_creates_folder_and_keeps_empty_values(tmp_path):
    destination = tmp_path / 'data' / 'cars.csv'  # 'data' does not exist yet
    df = save_listings(parse_listing_page(FIXTURE), str(destination))
    assert list(df.columns) == COLUMNS
    saved = pd.read_csv(destination)
    assert len(saved) == 3
    assert saved.loc[0, 'Prix'] == 6999
    assert pd.isna(saved.loc[2, 'Prix'])
    assert pd.isna(saved.loc[2, 'Id'])


def test_save_listings_without_vehicle(tmp_path):
    df = save_listings([], str(tmp_path / 'empty.csv'))
    assert df.empty
    assert list(df.columns) == COLUMNS


def test_save_listings_keeps_one_row_per_id(tmp_path):
    rows = parse_listing_page(FIXTURE) * 2  # every car twice
    df = save_listings(rows, str(tmp_path / 'cars.csv'))
    assert len(df) == 4  # 2 cars with an Id (once each) + the 2 copies of the car without Id


def test_load_listings_gives_back_what_was_saved(tmp_path):
    rows = parse_listing_page(FIXTURE)
    destination = str(tmp_path / 'cars.csv')
    save_listings(rows, destination)
    assert load_listings(destination) == rows


def test_load_listings_of_a_file_made_by_an_older_version(tmp_path):
    destination = tmp_path / 'old.csv'
    pd.DataFrame(
        [
            {
                'Id': 1,
                'Nom': 'Peugeot 208',
                'Carburant': 'Essence',
                'Boite de vitesse': 'Manuelle',
                'Km': 1000,
                'année': 2020,
                'Prix': 9000,
            }
        ]
    ).to_csv(destination, index=False)
    row = load_listings(str(destination))[0]
    assert row['Puissance'] is None and row['Motorisation'] is None and row['Finition'] is None
    assert row['Prix'] == 9000


def test_load_listings_of_a_missing_file(tmp_path):
    assert load_listings(str(tmp_path / 'nothing.csv')) == []


# ---------------------------------------------------------------- command line


def test_main_saves_after_every_page_and_prints_progress(fake_site, no_wait, tmp_path, capsys):
    destination = tmp_path / 'cars.csv'
    df = main(['--pages', '2', '--output', str(destination)])
    assert len(df) == 6
    assert len(pd.read_csv(destination)) == 6
    output = capsys.readouterr().out
    assert 'page 1: 3 vehicles (3 in total)' in output
    assert 'page 2: 3 vehicles (6 in total)' in output
    assert '6 vehicles saved' in output


def test_main_all_goes_until_the_end_of_the_listing(fake_site, no_wait, tmp_path):
    df = main(['--all', '--output', str(tmp_path / 'cars.csv')])
    assert len(df) == 6  # page 3 is empty


def test_main_error_keeps_the_pages_and_tells_how_to_continue(monkeypatch, no_wait, tmp_path, capsys):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        page = (params or {}).get('page', 1)
        return FakeResponse(fixture_page(page) if page < 3 else '', 500 if page == 3 else 200)

    monkeypatch.setattr(requests.Session, 'get', fake_get)
    destination = tmp_path / 'cars.csv'

    with pytest.raises(SystemExit) as stop:
        main(['--all', '--output', str(destination)])

    assert stop.value.code == 1
    assert len(pd.read_csv(destination)) == 6  # pages 1 and 2 are saved
    error = capsys.readouterr().err
    assert '--start-page 3 --append' in error
    assert str(destination) in error


def test_main_append_adds_to_the_file_without_duplicates(fake_site, no_wait, tmp_path):
    destination = str(tmp_path / 'cars.csv')
    main(['--pages', '1', '--output', destination])  # page 1
    df = main(['--pages', '2', '--start-page', '1', '--append', '--output', destination])  # pages 1 and 2 again
    # The 2 cars with an Id of page 1 are not added twice. The cars without Id cannot be
    # recognised, so they are added again: 3 (first run) + 4 (new in the second run) = 7 rows.
    ids = df['Id'].dropna().tolist()
    assert len(ids) == len(set(ids)) == 4
    assert len(df) == 7


def test_main_without_append_replaces_the_file(fake_site, no_wait, tmp_path):
    destination = str(tmp_path / 'cars.csv')
    main(['--pages', '2', '--output', destination])
    df = main(['--pages', '1', '--output', destination])
    assert len(df) == 3


@pytest.mark.parametrize(
    'engine, expected',
    [
        ('50 kWh - 136', 50.0),
        ('82kWh - 498', 82.0),
        ('80,5 Kwh - 204', 80.5),
        ('77.4 KWH AWD', 77.4),
        ('1.2 PureTech 68 BVM5', None),
        ('2008 Hybrid 145 ch e-DCS6', None),
        ('', None),
        (None, None),
        (float('nan'), None),
    ],
)
def test_parse_battery(engine, expected):
    assert parse_battery(engine) == expected
