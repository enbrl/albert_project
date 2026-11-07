"""Tests of the command line of scripts/main.py (no network: requests.get is replaced)."""
import pandas as pd
import pytest

from scripts import datasourcing
from scripts.main import DEFAULT_OUTPUT, main

SITE = 'https://quotes.toscrape.com'


class FakeResponse:
    status_code = 200

    def __init__(self, text):
        self.content = text.encode('utf-8')

    def raise_for_status(self):
        pass


def quote_block(author, text):
    return f'<div class="quote"><span class="text">{text}</span><small class="author">{author}</small></div>'


@pytest.fixture
def fake_site(monkeypatch):
    """A site with 2 pages. Returns the list of the URLs that were requested."""
    pages = {
        f'{SITE}/': quote_block('Zola', 'Z1') + '<li class="next"><a href="/page/2/">Next</a></li>',
        f'{SITE}/page/2/': quote_block('Albert', 'A1'),
    }
    requested = []

    def fake_get(url, timeout=None):
        requested.append(url)
        return FakeResponse(pages[url])

    monkeypatch.setattr(datasourcing.requests, 'get', fake_get)
    return requested


def test_without_argument_scrapes_the_default_site_into_the_default_file(fake_site, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the default file is a relative path
    df = main([])
    assert list(df['Author']) == ['Albert', 'Zola']  # sorted
    assert fake_site == [datasourcing.base_url, f'{SITE}/page/2/']
    assert list(pd.read_csv(tmp_path / DEFAULT_OUTPUT)['Author']) == ['Albert', 'Zola']


def test_url_argument_is_the_first_page(fake_site, tmp_path):
    df = main(['--url', f'{SITE}/page/2/', '--output', str(tmp_path / 'out.csv')])
    assert list(df['Author']) == ['Albert']
    assert fake_site == [f'{SITE}/page/2/']


def test_output_argument_creates_the_folders(fake_site, tmp_path):
    destination = tmp_path / 'some' / 'folder' / 'quotes.csv'
    main(['--output', str(destination)])
    assert destination.exists()


def test_never_asks_a_question(fake_site, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('main must not call input()')

    monkeypatch.setattr('builtins.input', forbidden)
    main(['--output', str(tmp_path / 'quotes.csv')])


def test_unknown_argument_is_refused(capsys):
    with pytest.raises(SystemExit) as error:
        main(['--unknown'])
    assert error.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


def test_help_shows_the_options(capsys):
    with pytest.raises(SystemExit) as error:
        main(['--help'])
    assert error.value.code == 0
    out = capsys.readouterr().out
    assert '--url' in out and '--output' in out
