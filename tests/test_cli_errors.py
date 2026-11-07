"""Tests of the messages shown by the command lines (scripts/cli.py and the scripts that use it)."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

from scripts import aramis_scraper, models, predict
from scripts import analysis as analysis_script
from scripts import datacleaning as cleaning_script
from scripts import main as quotes_script
from scripts.cli import UserError, friendly_message, positive_int, run

ROOT = Path(__file__).resolve().parent.parent
HEADER = 'Nom,Carburant,Boite de vitesse,Km,année,Prix\n'


def command(monkeypatch, capsys, main, *arguments):
    """Run a script like `python -m scripts.xxx arguments` and return (exit code, error text)."""
    monkeypatch.setattr(sys, 'argv', ['script', *arguments])
    with pytest.raises(SystemExit) as stop:
        run(main)
    return stop.value.code, capsys.readouterr().err


# ---------------------------------------------------------------- the helpers


def test_positive_int():
    assert positive_int('5') == 5
    assert positive_int('1') == 1
    for bad in ('0', '-3', 'abc', '2.5', ''):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int(bad)


def test_friendly_message_for_the_problems_the_user_can_fix():
    assert friendly_message(UserError('not enough cars')) == 'not enough cars'
    assert 'cars.csv' in friendly_message(FileNotFoundError(2, 'No such file', 'cars.csv'))
    assert 'folder' in friendly_message(IsADirectoryError(21, 'Is a directory', 'data'))
    assert 'Not allowed' in friendly_message(PermissionError(13, 'Permission denied', 'x.csv'))
    assert 'empty' in friendly_message(pd.errors.EmptyDataError('No columns to parse'))
    assert 'not a valid CSV' in friendly_message(pd.errors.ParserError('bad line'))
    assert 'Network problem' in friendly_message(requests.ConnectionError('no internet'))


def test_friendly_message_is_none_for_a_bug():
    assert friendly_message(KeyError('Prix')) is None
    assert friendly_message(ZeroDivisionError()) is None
    assert friendly_message(ValueError('a plain ValueError is not a user error')) is None


def test_run_shows_one_line_and_stops_with_code_1(capsys):
    def main():
        raise UserError('not enough cars')

    with pytest.raises(SystemExit) as stop:
        run(main)
    assert stop.value.code == 1
    assert capsys.readouterr().err == 'Error: not enough cars\n'


def test_run_does_not_hide_a_bug():
    def main():
        raise ZeroDivisionError('a real bug')

    with pytest.raises(ZeroDivisionError):
        run(main)


def test_run_returns_the_result_and_can_print_it(capsys):
    assert run(lambda: 42) == 42
    assert capsys.readouterr().out == ''
    run(lambda: 'the table', print_result=True)
    assert capsys.readouterr().out == 'the table\n'


# ---------------------------------------------------------------- files that do not exist or are wrong


@pytest.mark.parametrize('module', [cleaning_script, models, analysis_script])
def test_missing_input_file(monkeypatch, capsys, tmp_path, module):
    code, error = command(monkeypatch, capsys, module.main, str(tmp_path / 'nothing.csv'))
    assert code == 1
    assert error.startswith('Error: File not found:') and 'nothing.csv' in error
    assert 'Traceback' not in error


def test_empty_file(monkeypatch, capsys, tmp_path):
    empty = tmp_path / 'empty.csv'
    empty.write_text('')
    code, error = command(monkeypatch, capsys, cleaning_script.main, str(empty))
    assert code == 1 and 'empty' in error


def test_csv_without_the_needed_columns(monkeypatch, capsys, tmp_path):
    wrong = tmp_path / 'wrong.csv'
    wrong.write_text('a,b\n1,2\n')
    code, error = command(monkeypatch, capsys, cleaning_script.main, str(wrong))
    assert code == 1 and error.startswith('Error: Missing columns')


def test_analysis_of_a_file_without_car(monkeypatch, capsys, tmp_path):
    header_only = tmp_path / 'none.csv'
    header_only.write_text(HEADER)
    code, error = command(monkeypatch, capsys, analysis_script.main, str(header_only))
    assert code == 1
    assert error.startswith('Error: There is no car in') and 'ValueError' not in error


def test_not_enough_cars_to_train(monkeypatch, capsys, tmp_path):
    tiny = tmp_path / 'tiny.csv'
    tiny.write_text(HEADER + 'Peugeot 208,Essence,Manuelle,1000,2020,9000\n')
    code, error = command(monkeypatch, capsys, models.main, str(tiny), '--model', str(tmp_path / 'm.joblib'))
    assert code == 1
    assert error.startswith('Error: At least 20 cars are needed to train a model, got 1')
    assert not (tmp_path / 'm.joblib').exists()


# ---------------------------------------------------------------- scraper options and safety


@pytest.mark.parametrize(
    'option, value',
    [('--pages', '0'), ('--pages', '-3'), ('--pages', 'many'), ('--start-page', '0'), ('--start-page', '-1')],
)
def test_scraper_refuses_impossible_numbers(monkeypatch, capsys, tmp_path, option, value):
    destination = tmp_path / 'cars.csv'
    destination.write_text('keep me')
    monkeypatch.setattr(sys, 'argv', ['script', option, value, '--output', str(destination)])
    with pytest.raises(SystemExit) as stop:
        aramis_scraper.main()
    assert stop.value.code == 2  # the usual code of a wrong option
    err = capsys.readouterr().err
    assert 'must be at least 1' in err or 'not a whole number' in err
    assert destination.read_text() == 'keep me'  # nothing was written


def test_scraper_message_for_a_wrong_number(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['script', '--pages', '0', '--output', str(tmp_path / 'x.csv')])
    with pytest.raises(SystemExit):
        aramis_scraper.main()
    assert '0 is not valid: it must be at least 1' in capsys.readouterr().err


class EmptyPage:
    text = '<html><body></body></html>'
    status_code = 200

    def raise_for_status(self):
        pass


def test_scraper_never_replaces_a_good_file_by_an_empty_one(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(requests.Session, 'get', lambda self, *args, **kwargs: EmptyPage())
    destination = tmp_path / 'cars.csv'
    destination.write_text('precious data')
    code, error = command(monkeypatch, capsys, aramis_scraper.main, '--pages', '1', '--output', str(destination))
    assert code == 1
    assert 'No vehicle was found' in error and 'was not changed' in error
    assert destination.read_text() == 'precious data'


def test_scraper_without_vehicle_does_not_create_an_empty_file(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(requests.Session, 'get', lambda self, *args, **kwargs: EmptyPage())
    destination = tmp_path / 'new.csv'
    code, _ = command(monkeypatch, capsys, aramis_scraper.main, '--pages', '1', '--output', str(destination))
    assert code == 1
    assert not destination.exists()


# ---------------------------------------------------------------- the other scripts


def test_quotes_script_with_an_address_that_is_not_a_url(monkeypatch, capsys, tmp_path):
    code, error = command(
        monkeypatch, capsys, quotes_script.main, '--url', 'not-a-url', '--output', str(tmp_path / 'q.csv')
    )
    assert code == 1 and error.startswith('Error: Network problem')


@pytest.mark.parametrize('year', ['1700', '2999', '0'])
def test_predict_refuses_an_impossible_year(monkeypatch, capsys, year):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'script',
            '--name',
            'Peugeot 208',
            '--fuel',
            'Essence',
            '--gearbox',
            'Manuelle',
            '--km',
            '1000',
            '--year',
            year,
        ],
    )
    with pytest.raises(SystemExit) as stop:
        predict.main()
    assert stop.value.code == 2
    assert '--year must be between 1990' in capsys.readouterr().err


def test_predict_refuses_an_empty_name(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        'argv',
        ['script', '--name', '   ', '--fuel', 'Essence', '--gearbox', 'Manuelle', '--km', '1000', '--year', '2020'],
    )
    with pytest.raises(SystemExit) as stop:
        predict.main()
    assert stop.value.code == 2


# ---------------------------------------------------------------- the real command, from a terminal


@pytest.mark.parametrize('module', ['scripts.datacleaning', 'scripts.models', 'scripts.analysis'])
def test_python_m_shows_a_message_not_a_traceback(module, tmp_path):
    done = subprocess.run(
        [sys.executable, '-m', module, str(tmp_path / 'nothing.csv')],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 1
    assert done.stderr.startswith('Error: File not found:')
    assert 'Traceback' not in done.stderr
