"""Tests of scripts/estimator.py (what the web application needs, without Streamlit)."""

import json

import pandas as pd
import pytest

from scripts.estimator import (
    FUELS,
    SIMILAR_COLUMNS,
    estimate,
    fuel_options,
    load_cars,
    market_summary,
    model_names,
    similar_cars,
    typical_error,
)
from scripts.models import save_model, train_final_model
from tests.test_models import make_cars, make_cars_with_battery, make_cars_with_power


@pytest.fixture(scope='module')
def cars():
    return make_cars(300, seed=4)


@pytest.fixture(scope='module')
def model(cars):
    return train_final_model(cars, kind='ridge')


# ---------------------------------------------------------------- the file of cars


def test_load_cars(cars, tmp_path):
    path = tmp_path / 'cars.csv'
    cars.to_csv(path, index=False)
    assert len(load_cars(str(path))) == 300


@pytest.mark.parametrize('content', ['', 'a,b\n1,2\n', 'Nom,Carburant,Boite de vitesse,Km,année,Prix\n'])
def test_load_cars_unusable_files_give_none(tmp_path, content):
    path = tmp_path / 'bad.csv'
    path.write_text(content)
    assert load_cars(str(path)) is None


def test_load_cars_of_nothing(tmp_path):
    assert load_cars(str(tmp_path / 'nothing.csv')) is None
    assert load_cars('') is None
    assert load_cars(None) is None
    assert load_cars(str(tmp_path)) is None  # a folder


# ---------------------------------------------------------------- choices


def test_model_names_are_sorted_and_unique(cars):
    assert model_names(cars) == ['Opel Astra', 'Peugeot 208', 'Peugeot 3008', 'Renault Clio']
    assert model_names(None) == []


def test_fuel_options(cars):
    assert fuel_options(cars) == ['Essence', 'Diesel']  # the ones of the data, in the usual order
    assert fuel_options(None) == FUELS
    assert fuel_options(cars.assign(Carburant='Ethanol')) == FUELS  # unknown fuels: the usual list


# ---------------------------------------------------------------- similar cars


def test_similar_cars_have_the_same_name_and_are_the_closest_first(cars):
    result = similar_cars(cars, 'Peugeot 208', 2019, 60_000, count=5)
    assert list(result.columns) == SIMILAR_COLUMNS and len(result) == 5
    assert set(result['Nom']) == {'Peugeot 208'}
    distance = (result['année'] - 2019).abs() * 20_000 + (result['Km'] - 60_000).abs()
    assert list(distance) == sorted(distance)
    assert distance.iloc[0] <= distance.iloc[-1]


def test_similar_cars_name_is_cleaned_like_in_the_data(cars):
    assert len(similar_cars(cars, '  PEUGEOT   208 ', 2019, 60_000)) == 5


def test_similar_cars_without_match():
    cars = make_cars(50)
    for name in ('Ferrari F40', '', '   '):
        result = similar_cars(cars, name, 2019, 60_000)
        assert result.empty and list(result.columns) == SIMILAR_COLUMNS
    assert similar_cars(None, 'Peugeot 208', 2019, 60_000).empty


def test_similar_cars_fewer_than_asked():
    two = pd.DataFrame(
        {
            'Nom': ['Peugeot 208'] * 2,
            'Carburant': ['Essence'] * 2,
            'Boite de vitesse': ['Manuelle'] * 2,
            'Km': [1000, 2000],
            'année': [2020, 2021],
            'Prix': [9000, 9500],
        }
    )
    assert len(similar_cars(two, 'Peugeot 208', 2020, 1000, count=5)) == 2


def test_market_summary(cars):
    summary = market_summary(cars, 'peugeot 208')
    same = cars[cars['Nom'] == 'Peugeot 208']['Prix']
    assert summary == {
        'cars': len(same),
        'min': int(same.min()),
        'median': float(same.median()),
        'max': int(same.max()),
    }
    assert market_summary(cars, 'Ferrari F40') is None
    assert market_summary(cars, '') is None
    assert market_summary(None, 'Peugeot 208') is None


# ---------------------------------------------------------------- the estimate


def test_estimate_cleans_the_labels(model):
    clean = estimate(model, 'Peugeot 208', 'Diesel', 'Automatique', 60_000, 2019)
    messy = estimate(model, '  PEUGEOT   208 ', 'diesel', 'Auto.', 60_000, 2019)
    assert messy == pytest.approx(clean) and clean > 0


def test_estimate_power_zero_means_unknown(model):
    arguments = ('Opel Astra', 'Essence', 'Manuelle', 60_000, 2019)
    assert estimate(model, *arguments, power=0) == pytest.approx(estimate(model, *arguments))


def test_estimate_uses_the_power_when_the_model_knows_it():
    model = train_final_model(make_cars_with_power(300, seed=4), kind='ridge')
    arguments = ('Opel Astra', 'Essence', 'Manuelle', 60_000, 2019)
    assert estimate(model, *arguments, power=190) > estimate(model, *arguments, power=65) + 1000


# ---------------------------------------------------------------- the error of the model


def test_typical_error(model, tmp_path):
    path = str(tmp_path / 'price.joblib')
    save_model(model, path)
    assert typical_error(path) is None  # nothing about errors was saved
    (tmp_path / 'price.json').write_text(json.dumps({'metrics': {'mae': 1556.4}}))
    assert typical_error(path) == pytest.approx(1556.4)


@pytest.mark.parametrize('content', ['not json', '[]', '{"metrics": null}', '{"metrics": {"mae": "high"}}', '{}'])
def test_typical_error_of_unusable_information(tmp_path, content):
    (tmp_path / 'price.json').write_text(content)
    assert typical_error(str(tmp_path / 'price.joblib')) is None


def test_typical_error_without_file(tmp_path):
    assert typical_error(str(tmp_path / 'nothing.joblib')) is None


def test_estimate_uses_the_battery_when_the_model_knows_it():
    model = train_final_model(make_cars_with_battery(450, seed=4), kind='ridge')
    arguments = ('Opel Astra', 'Électrique', 'Automatique', 40_000, 2022)
    assert estimate(model, *arguments, battery=100) > estimate(model, *arguments, battery=40) + 1000
    assert estimate(model, *arguments, battery=0) == pytest.approx(estimate(model, *arguments))  # 0 means unknown
