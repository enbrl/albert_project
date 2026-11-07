"""Tests of scripts/models.py, on made-up cars whose price follows a known rule."""
import numpy as np
import pandas as pd
import pytest

from scripts.models import (
    load_model,
    main,
    predict_price,
    prepare_features,
    save_model,
    train_and_evaluate,
    train_final_model,
)

NAMES = {'Peugeot 208': 9000, 'Renault Clio': 8500, 'Peugeot 3008': 18000, 'Opel Astra': 11000}


def make_cars(n=400, seed=0):
    """Price = base price of the model - 0.07 EUR per km - 700 EUR per year of age (+ diesel bonus + noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        name = rng.choice(list(NAMES))
        year = int(rng.integers(2012, 2026))
        km = int(rng.integers(5_000, 200_000))
        fuel = str(rng.choice(['Essence', 'Diesel']))
        gearbox = str(rng.choice(['Manuelle', 'Automatique']))
        price = NAMES[name] + 9000 - 0.07 * km - 700 * (2026 - year) + (500 if fuel == 'Diesel' else 0)
        price += rng.normal(0, 300)
        rows.append({'Nom': name, 'Carburant': fuel, 'Boite de vitesse': gearbox, 'Km': km, 'année': year, 'Prix': int(price)})
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def cars():
    return make_cars()


@pytest.fixture(scope='module')
def trained(cars):
    return train_and_evaluate(cars)


def test_prepare_features_adds_the_brand():
    features = prepare_features(make_cars(5))
    assert 'Marque' in features.columns
    assert set(features['Marque']) <= {'Peugeot', 'Renault', 'Opel'}


def test_prepare_features_requires_the_columns():
    with pytest.raises(ValueError, match='Missing columns'):
        prepare_features(pd.DataFrame({'Nom': ['Peugeot 208']}))


def test_model_learns_the_price_rule(trained):
    _, metrics = trained
    assert metrics['n_train'] == 320 and metrics['n_test'] == 80
    assert metrics['r2'] > 0.9
    assert metrics['mae'] < metrics['baseline_mae'] / 3  # much better than always guessing the mean
    assert metrics['rmse'] >= metrics['mae']


def test_training_is_reproducible(cars, trained):
    _, metrics = trained
    _, again = train_and_evaluate(cars)
    assert again == pytest.approx(metrics)  # tiny rounding differences can come from the parallel computation


def test_cheaper_when_older_and_with_more_km(trained):
    model, _ = trained
    recent = predict_price(model, 'Peugeot 208', 'Essence', 'Manuelle', km=20_000, year=2024)
    old = predict_price(model, 'Peugeot 208', 'Essence', 'Manuelle', km=180_000, year=2013)
    assert recent > old + 3000


def test_bigger_model_is_more_expensive(trained):
    model, _ = trained
    suv = predict_price(model, 'Peugeot 3008', 'Essence', 'Manuelle', km=60_000, year=2019)
    city_car = predict_price(model, 'Renault Clio', 'Essence', 'Manuelle', km=60_000, year=2019)
    assert suv > city_car + 5000


def test_unknown_labels_do_not_crash(trained):
    model, _ = trained
    price = predict_price(model, 'Marque Inconnue X1', 'Hydrogène', 'Robot', km=50_000, year=2020)
    assert 0 < price < 40_000


def test_too_few_cars_are_refused():
    with pytest.raises(ValueError, match='At least'):
        train_and_evaluate(make_cars(10))


def test_save_and_load_give_the_same_predictions(cars, tmp_path):
    model = train_final_model(cars)
    destination = tmp_path / 'models' / 'price.joblib'  # 'models' does not exist yet
    save_model(model, str(destination))
    loaded = load_model(str(destination))
    args = ('Opel Astra', 'Diesel', 'Automatique', 90_000, 2017)
    assert predict_price(loaded, *args) == pytest.approx(predict_price(model, *args))


def test_main_trains_and_saves(cars, tmp_path, capsys):
    csv = tmp_path / 'cars_clean.csv'
    cars.to_csv(csv, index=False)
    destination = tmp_path / 'out' / 'model.joblib'

    metrics = main([str(csv), '--model', str(destination)])

    assert destination.exists()
    assert metrics['r2'] > 0.9
    assert 'Mean error' in capsys.readouterr().out
