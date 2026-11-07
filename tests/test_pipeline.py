"""Tests of scripts/pipeline.py."""

import os
from pathlib import Path

import pandas as pd
import pytest
import requests

from scripts import analysis, aramis_scraper, datacleaning, models, pipeline
from tests.test_models import make_cars

FIXTURE = (Path(__file__).parent / 'fixtures' / 'aramis_listing.html').read_text(encoding='utf-8')


class Recorder:
    """Replaces the main function of a script: remembers that it was called, with which arguments."""

    def __init__(self, name, calls, fail_with=None):
        self.name, self.calls, self.fail_with = name, calls, fail_with

    def __call__(self, arguments):
        self.calls.append((self.name, list(arguments)))
        if self.fail_with is not None:
            raise self.fail_with


@pytest.fixture
def calls(monkeypatch):
    """All the steps are replaced by recorders; returns the list of the calls, in order."""
    recorded = []
    for name, module in [('scrape', aramis_scraper), ('clean', datacleaning), ('report', analysis), ('train', models)]:
        monkeypatch.setattr(module, 'main', Recorder(name, recorded))
    return recorded


def names(calls):
    return [name for name, _ in calls]


# ---------------------------------------------------------------- the order and the arguments


def test_the_four_steps_run_in_order_with_the_files_of_the_readme(calls, capsys):
    pipeline.main([])
    assert calls == [
        ('scrape', ['--pages', '5', '--output', os.path.join('data', 'cars_raw.csv')]),
        ('clean', [os.path.join('data', 'cars_raw.csv'), '--output', os.path.join('data', 'cars_clean.csv')]),
        ('report', [os.path.join('data', 'cars_clean.csv'), '--output', 'reports']),
        ('train', [os.path.join('data', 'cars_clean.csv'), '--kind', 'best', '--model', models.DEFAULT_MODEL_PATH]),
    ]
    output = capsys.readouterr().out
    for number in (1, 2, 3, 4):
        assert f'=== Step {number}/4' in output
    assert 'Done in' in output and 'report.html' in output and 'cars_clean.csv' in output


def test_all_scrapes_the_whole_listing(calls):
    pipeline.main(['--all'])
    assert calls[0][1][:1] == ['--all'] and '--pages' not in calls[0][1]


def test_pages_option(calls):
    pipeline.main(['--pages', '12'])
    assert calls[0][1][:2] == ['--pages', '12']


def test_skip_scrape_starts_from_the_file_already_there(calls, tmp_path):
    (tmp_path / 'cars_raw.csv').write_text('anything')
    pipeline.main(['--skip-scrape', '--data-dir', str(tmp_path)])
    assert names(calls) == ['clean', 'report', 'train']


def test_no_report(calls):
    pipeline.main(['--no-report'])
    assert names(calls) == ['scrape', 'clean', 'train']


def test_kind_folders_and_model_are_given_to_the_steps(calls, tmp_path):
    pipeline.main(
        [
            '--kind',
            'ridge',
            '--data-dir',
            str(tmp_path / 'd'),
            '--reports-dir',
            str(tmp_path / 'r'),
            '--model',
            str(tmp_path / 'm.joblib'),
        ]
    )
    by_name = dict(calls)
    assert by_name['scrape'][-1] == str(tmp_path / 'd' / 'cars_raw.csv')
    assert by_name['report'][-1] == str(tmp_path / 'r')
    assert by_name['train'] == [
        str(tmp_path / 'd' / 'cars_clean.csv'),
        '--kind',
        'ridge',
        '--model',
        str(tmp_path / 'm.joblib'),
    ]


# ---------------------------------------------------------------- wrong options


def test_skip_scrape_without_a_file_is_refused(calls, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        pipeline.main(['--skip-scrape', '--data-dir', str(tmp_path)])
    assert stop.value.code == 2
    assert 'cars_raw.csv' in capsys.readouterr().err and calls == []


@pytest.mark.parametrize(
    'arguments', [['--pages', '0'], ['--pages', '-2'], ['--pages', '3', '--all'], ['--kind', 'magic']]
)
def test_wrong_options_are_refused(calls, arguments):
    with pytest.raises(SystemExit) as stop:
        pipeline.main(arguments)
    assert stop.value.code == 2 and calls == []  # nothing was run


# ---------------------------------------------------------------- a step that fails


def test_a_step_that_stops_the_script_stops_the_pipeline(monkeypatch, calls, capsys):
    monkeypatch.setattr(analysis, 'main', Recorder('report', calls, fail_with=SystemExit(1)))
    with pytest.raises(SystemExit) as stop:
        pipeline.main([])
    assert stop.value.code == 1
    assert names(calls) == ['scrape', 'clean', 'report']  # the model is not trained
    error = capsys.readouterr().err
    assert 'Stopped at step 3/4 (Make the report and the charts)' in error
    assert 'The files of the steps before were kept' in error and '--skip-scrape' in error


def test_an_error_names_the_step_and_the_problem(monkeypatch, calls, capsys):
    monkeypatch.setattr(datacleaning, 'main', Recorder('clean', calls, fail_with=ValueError('Missing columns')))
    with pytest.raises(SystemExit) as stop:
        pipeline.main([])
    assert stop.value.code == 1 and names(calls) == ['scrape', 'clean']
    assert 'Stopped at step 2/4 (Clean the data): ValueError: Missing columns' in capsys.readouterr().err


def test_a_failure_of_the_first_step_does_not_suggest_skip_scrape(monkeypatch, calls, capsys):
    monkeypatch.setattr(aramis_scraper, 'main', Recorder('scrape', calls, fail_with=SystemExit(1)))
    with pytest.raises(SystemExit):
        pipeline.main([])
    error = capsys.readouterr().err
    assert 'Stopped at step 1/4' in error and '--skip-scrape' not in error and 'were kept' not in error
    assert names(calls) == ['scrape']


def test_exit_code_zero_is_not_a_failure(monkeypatch, calls):
    monkeypatch.setattr(datacleaning, 'main', Recorder('clean', calls, fail_with=SystemExit(0)))
    pipeline.main([])
    assert names(calls) == ['scrape', 'clean', 'report', 'train']


def test_debug_variable_shows_the_whole_error(monkeypatch, calls):
    monkeypatch.setenv('PIPELINE_DEBUG', '1')
    monkeypatch.setattr(datacleaning, 'main', Recorder('clean', calls, fail_with=KeyError('Prix')))
    with pytest.raises(KeyError):
        pipeline.main([])


# ---------------------------------------------------------------- the real scripts, one after the other


def test_the_real_steps_work_together_on_a_scraped_file(tmp_path, capsys):
    raw = tmp_path / 'data' / 'cars_raw.csv'
    raw.parent.mkdir()
    cars = make_cars(80, seed=2)
    cars.insert(0, 'Id', range(1, 81))
    cars.to_csv(raw, index=False)
    model = tmp_path / 'models' / 'price.joblib'

    made = pipeline.main(
        [
            '--skip-scrape',
            '--data-dir',
            str(raw.parent),
            '--reports-dir',
            str(tmp_path / 'reports'),
            '--model',
            str(model),
            '--kind',
            'ridge',
        ]
    )

    assert (raw.parent / 'cars_clean.csv').exists() and model.exists()
    assert len(pd.read_csv(raw.parent / 'cars_clean.csv')) == 80
    assert (tmp_path / 'reports' / 'report.html').exists()
    assert made[-1].endswith('report.html')
    assert models.load_model(str(model)) is not None
    assert 'Done in' in capsys.readouterr().out


class FixturePage:
    status_code = 200
    text = FIXTURE

    def raise_for_status(self):
        pass


def test_the_whole_pipeline_stops_cleanly_when_there_are_too_few_cars(monkeypatch, tmp_path, capsys):
    """The scraped fixture has only 2 usable cars: the model cannot be trained and the message says so."""
    monkeypatch.setattr(requests.Session, 'get', lambda self, *args, **kwargs: FixturePage())
    with pytest.raises(SystemExit) as stop:
        pipeline.main(
            [
                '--pages',
                '1',
                '--data-dir',
                str(tmp_path / 'data'),
                '--reports-dir',
                str(tmp_path / 'reports'),
                '--model',
                str(tmp_path / 'm.joblib'),
            ]
        )
    assert stop.value.code == 1
    error = capsys.readouterr().err
    assert 'Stopped at step 4/4 (Train and save the price model)' in error and 'At least 20 cars' in error
    assert (tmp_path / 'data' / 'cars_raw.csv').exists() and (tmp_path / 'data' / 'cars_clean.csv').exists()
    assert not (tmp_path / 'm.joblib').exists()
