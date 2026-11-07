"""Tests of the command line scripts/predict.py."""

import re

import pytest

from scripts import predict
from scripts.models import save_model, train_final_model
from tests.test_models import make_cars, make_cars_with_battery, make_cars_with_power


@pytest.fixture(scope='module')
def model_file(tmp_path_factory):
    path = tmp_path_factory.mktemp('model') / 'price.joblib'
    save_model(train_final_model(make_cars()), str(path))
    return str(path)


def run(model_file, *arguments):
    return predict.main(['--model', model_file, *arguments])


def test_prints_an_estimated_price(model_file, capsys):
    price = run(
        model_file,
        '--name',
        'Peugeot 208',
        '--fuel',
        'Essence',
        '--gearbox',
        'Manuelle',
        '--km',
        '60000',
        '--year',
        '2019',
    )
    assert 3_000 < price < 20_000
    assert re.fullmatch(r'Estimated price: \d[\d ]* €\n', capsys.readouterr().out)


def test_labels_are_cleaned_like_in_datacleaning(model_file):
    clean = run(
        model_file,
        '--name',
        'Peugeot 208',
        '--fuel',
        'Diesel',
        '--gearbox',
        'Automatique',
        '--km',
        '60000',
        '--year',
        '2019',
    )
    messy = run(
        model_file,
        '--name',
        '  PEUGEOT   208 ',
        '--fuel',
        'diesel',
        '--gearbox',
        'Auto.',
        '--km',
        '60000',
        '--year',
        '2019',
    )
    assert messy == pytest.approx(clean)


def test_newer_and_less_driven_car_is_more_expensive(model_file):
    recent = run(
        model_file,
        '--name',
        'Renault Clio',
        '--fuel',
        'Essence',
        '--gearbox',
        'Manuelle',
        '--km',
        '15000',
        '--year',
        '2024',
    )
    old = run(
        model_file,
        '--name',
        'Renault Clio',
        '--fuel',
        'Essence',
        '--gearbox',
        'Manuelle',
        '--km',
        '170000',
        '--year',
        '2013',
    )
    assert recent > old


def test_missing_model_gives_a_clear_message(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        predict.main(
            [
                '--model',
                str(tmp_path / 'nothing.joblib'),
                '--name',
                'Peugeot 208',
                '--fuel',
                'Essence',
                '--gearbox',
                'Manuelle',
                '--km',
                '1000',
                '--year',
                '2020',
            ]
        )
    assert stop.value.code == 1
    assert 'python -m scripts.models' in capsys.readouterr().err


def test_unreadable_model_gives_a_clear_message(tmp_path, capsys):
    broken = tmp_path / 'broken.joblib'
    broken.write_bytes(b'this is not a model')
    with pytest.raises(SystemExit) as stop:
        predict.main(
            [
                '--model',
                str(broken),
                '--name',
                'Peugeot 208',
                '--fuel',
                'Essence',
                '--gearbox',
                'Manuelle',
                '--km',
                '1000',
                '--year',
                '2020',
            ]
        )
    assert stop.value.code == 1
    error = capsys.readouterr().err
    assert 'cannot be read' in error and 'train it again' in error.lower()


def test_negative_mileage_is_refused(model_file, capsys):
    with pytest.raises(SystemExit) as stop:
        run(
            model_file,
            '--name',
            'Peugeot 208',
            '--fuel',
            'Essence',
            '--gearbox',
            'Manuelle',
            '--km',
            '-5',
            '--year',
            '2020',
        )
    assert stop.value.code == 2


def test_required_options(model_file):
    with pytest.raises(SystemExit) as stop:
        run(model_file, '--name', 'Peugeot 208')
    assert stop.value.code == 2


@pytest.fixture(scope='module')
def model_with_power_file(tmp_path_factory):
    path = tmp_path_factory.mktemp('model_power') / 'price.joblib'
    save_model(train_final_model(make_cars_with_power()), str(path))
    return str(path)


def test_power_makes_the_estimate_change(model_with_power_file):
    arguments = [
        '--name',
        'Opel Astra',
        '--fuel',
        'Essence',
        '--gearbox',
        'Manuelle',
        '--km',
        '60000',
        '--year',
        '2019',
    ]
    weak = run(model_with_power_file, *arguments, '--power', '65')
    strong = run(model_with_power_file, *arguments, '--power', '190')
    no_power = run(model_with_power_file, *arguments)  # a model that knows the power still works without it
    assert strong > weak + 1500
    assert no_power > 0


def test_power_is_ignored_by_a_model_that_does_not_know_it(model_file):
    arguments = [
        '--name',
        'Opel Astra',
        '--fuel',
        'Essence',
        '--gearbox',
        'Manuelle',
        '--km',
        '60000',
        '--year',
        '2019',
    ]
    assert run(model_file, *arguments, '--power', '150') == pytest.approx(run(model_file, *arguments))


@pytest.mark.parametrize('power', ['0', '-90'])
def test_impossible_power_is_refused(model_file, power):
    with pytest.raises(SystemExit) as stop:
        run(
            model_file,
            '--name',
            'Peugeot 208',
            '--fuel',
            'Essence',
            '--gearbox',
            'Manuelle',
            '--km',
            '1000',
            '--year',
            '2020',
            '--power',
            power,
        )
    assert stop.value.code == 2


@pytest.fixture(scope='module')
def model_with_battery_file(tmp_path_factory):
    path = tmp_path_factory.mktemp('model_battery') / 'price.joblib'
    save_model(train_final_model(make_cars_with_battery(600)), str(path))
    return str(path)


def test_battery_makes_the_estimate_change(model_with_battery_file):
    arguments = ['--name', 'Opel Astra', '--fuel', 'Électrique', '--gearbox', 'Automatique', '--km', '40000']
    arguments += ['--year', '2022']
    small = run(model_with_battery_file, *arguments, '--battery', '40')
    large = run(model_with_battery_file, *arguments, '--battery', '100')
    assert large > small + 1500
    assert run(model_with_battery_file, *arguments) > 0  # the battery is optional


@pytest.mark.parametrize('battery', ['0', '-50'])
def test_impossible_battery_is_refused(model_file, battery, capsys):
    arguments = ['--name', 'Opel Astra', '--fuel', 'Électrique', '--gearbox', 'Automatique', '--km', '40000']
    arguments += ['--year', '2022', '--battery', battery]
    with pytest.raises(SystemExit) as stop:
        run(model_file, *arguments)
    assert stop.value.code == 2
    assert '--battery must be positive' in capsys.readouterr().err
