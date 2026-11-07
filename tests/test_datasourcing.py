"""Tests of scripts/datasourcing.py (no network: requests.get is replaced)."""
import pandas as pd
import pytest
import requests

from scripts import datasourcing
from scripts.datasourcing import csv_saving, fetch_quotes

SITE = 'https://quotes.toscrape.com'


def quote_block(author, text):
    return (
        f'<div class="quote"><span class="text">{text}</span>'
        f'<span>by <small class="author">{author}</small></span></div>'
    )


def page(quotes, next_href=None):
    """Build a page like quotes.toscrape.com: some quotes and an optional 'next' link."""
    html = ''.join(quote_block(author, text) for author, text in quotes)
    if next_href:
        html += f'<nav><ul class="pager"><li class="next"><a href="{next_href}">Next</a></li></ul></nav>'
    return f'<html><body>{html}</body></html>'


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.content = text.encode('utf-8')
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} error')


@pytest.fixture
def fake_site(monkeypatch):
    """A site with 3 pages. Returns the list of the URLs that were requested."""
    pages = {
        f'{SITE}/': page([('Zola', 'Z1'), ('Albert', 'A1')], '/page/2/'),
        f'{SITE}/page/2/': page([('Marie', 'M1')], '/page/3/'),
        f'{SITE}/page/3/': page([('Bob', 'B1')]),
    }
    requested = []

    def fake_get(url, timeout=None):
        requested.append(url)
        assert timeout, 'every request must have a timeout'
        return FakeResponse(pages[url]) if url in pages else FakeResponse('', 404)

    monkeypatch.setattr(datasourcing.requests, 'get', fake_get)
    return requested


def test_fetch_quotes_follows_all_the_pages(fake_site):
    quotes = fetch_quotes(f'{SITE}/')
    assert quotes == [
        {'Author': 'Zola', 'Citation': 'Z1'},
        {'Author': 'Albert', 'Citation': 'A1'},
        {'Author': 'Marie', 'Citation': 'M1'},
        {'Author': 'Bob', 'Citation': 'B1'},
    ]
    assert fake_site == [f'{SITE}/', f'{SITE}/page/2/', f'{SITE}/page/3/']


def test_fetch_quotes_can_start_from_another_page(fake_site):
    quotes = fetch_quotes(f'{SITE}/page/2/')
    assert [q['Author'] for q in quotes] == ['Marie', 'Bob']


def test_fetch_quotes_ignores_blocks_without_author_or_text(monkeypatch):
    html = '<div class="quote"><span class="text">No author</span></div>' + quote_block('Albert', 'A1')
    monkeypatch.setattr(datasourcing.requests, 'get', lambda url, timeout=None: FakeResponse(html))
    assert fetch_quotes(f'{SITE}/') == [{'Author': 'Albert', 'Citation': 'A1'}]


def test_fetch_quotes_raises_on_http_error(fake_site):
    with pytest.raises(requests.HTTPError):
        fetch_quotes(f'{SITE}/unknown/')


def test_csv_saving_saves_and_returns_the_sorted_data(tmp_path):
    data = [{'Author': 'Zola', 'Citation': 'Z1'}, {'Author': 'Albert', 'Citation': 'A1'}]
    destination = tmp_path / 'quotes.csv'

    returned = csv_saving(data, str(destination))

    assert list(returned['Author']) == ['Albert', 'Zola']
    saved = pd.read_csv(destination)
    assert list(saved['Author']) == ['Albert', 'Zola']  # the file is sorted too


def test_csv_saving_does_not_scrape_again(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('csv_saving must not make any request')

    monkeypatch.setattr(datasourcing.requests, 'get', forbidden)
    csv_saving([{'Author': 'Albert', 'Citation': 'A1'}], str(tmp_path / 'quotes.csv'))


def test_csv_saving_creates_the_output_folder(tmp_path):
    destination = tmp_path / 'data' / 'quotes.csv'  # 'data' does not exist yet
    csv_saving([{'Author': 'Albert', 'Citation': 'A1'}], str(destination))
    assert destination.exists()


def test_csv_saving_with_no_data(tmp_path):
    returned = csv_saving([], str(tmp_path / 'empty.csv'))
    assert returned.empty
