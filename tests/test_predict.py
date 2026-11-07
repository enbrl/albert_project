"""Tests of the command line scripts/predict.py."""
import re

import pytest

from scripts import predict
from scripts.models import save_model, train_final_model
from tests.test_models import make_cars


@pytest.fixture(scope='module')
def model_file(tmp_path_factory):
    path = tmp_path_factory.mktemp('model') / 'price.joblib'
    save_model(train_final_model(make_cars()), str(path))
    return str(path)


def run(model_file, *arguments):
    return predict.main(['--model', model_file, *arguments])


def test_prints_an_estimated_price(model_file, capsys):
    price = run(model_file, '--name', 'Peugeot 208', '--fuel', 'Essence', '--gearbox', 'Manuelle',
                '--km', '60000', '--year', '2019')
    assert 3_000 < price < 20_000
    assert re.fullmatch(r'Estimated price: \d[\d ]* €\n', capsys.readouterr().out)


def test_labels_are_cleaned_like_in_datacleaning(model_file):
    clean = run(model_file, '--name', 'Peugeot 208', '--fuel', 'Diesel', '--gearbox', 'Automatique',
                '--km', '60000', '--year', '2019')
    messy = run(model_file, '--name', '  PEUGEOT   208 ', '--fuel', 'diesel', '--gearbox', 'Auto.',
                '--km', '60000', '--year', '2019')
    assert messy == pytest.approx(clean)


def test_newer_and_less_driven_car_is_more_expensive(model_file):
    recent = run(model_file, '--name', 'Renault Clio', '--fuel', 'Essence', '--gearbox', 'Manuelle',
                 '--km', '15000', '--year', '2024')
    old = run(model_file, '--name', 'Renault Clio', '--fuel', 'Essence', '--gearbox', 'Manuelle',
              '--km', '170000', '--year', '2013')
    assert recent > old


def test_missing_model_gives_a_clear_message(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        predict.main(['--model', str(tmp_path / 'nothing.joblib'), '--name', 'Peugeot 208', '--fuel', 'Essence',
                      '--gearbox', 'Manuelle', '--km', '1000', '--year', '2020'])
    assert stop.value.code == 1
    assert 'python -m scripts.models' in capsys.readouterr().err


def test_negative_mileage_is_refused(model_file, capsys):
    with pytest.raises(SystemExit) as stop:
        run(model_file, '--name', 'Peugeot 208', '--fuel', 'Essence', '--gearbox', 'Manuelle',
            '--km', '-5', '--year', '2020')
    assert stop.value.code == 2


def test_required_options(model_file):
    with pytest.raises(SystemExit) as stop:
        run(model_file, '--name', 'Peugeot 208')
    assert stop.value.code == 2
