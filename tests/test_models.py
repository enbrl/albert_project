"""Tests of scripts/models.py, on made-up cars whose price follows a known rule."""

import numpy as np
import pandas as pd
import pytest

from scripts.models import (
    MODEL_KINDS,
    add_trim_family,
    best_kind,
    build_model,
    compare_models,
    feature_columns,
    feature_importance,
    load_model,
    main,
    predict_price,
    prepare_features,
    save_model,
    target_encoder,
    train_and_evaluate,
    train_final_model,
)

# Value of each model when it is new
NAMES = {'Peugeot 208': 16000, 'Renault Clio': 14000, 'Peugeot 3008': 30000, 'Opel Astra': 18000}


def make_cars(n=400, seed=0):
    """
    Price = value when new, losing 10 % per year of age and 25 % per 100 000 km (+ 3 % for diesel, + noise).

    It is the kind of rule a real market follows (a price is never negative).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        name = rng.choice(list(NAMES))
        year = int(rng.integers(2012, 2026))
        km = int(rng.integers(5_000, 200_000))
        fuel = str(rng.choice(['Essence', 'Diesel']))
        gearbox = str(rng.choice(['Manuelle', 'Automatique']))
        price = NAMES[name] * 0.9 ** (2026 - year) * (1 - 0.25 * km / 100_000) * (1.03 if fuel == 'Diesel' else 1)
        price *= rng.normal(1, 0.03)
        rows.append(
            {'Nom': name, 'Carburant': fuel, 'Boite de vitesse': gearbox, 'Km': km, 'année': year, 'Prix': int(price)}
        )
    return pd.DataFrame(rows)


def make_cars_with_power(n=400, seed=0):
    """The same cars, with a power (in horsepower) that changes the price by 0.5 % per horsepower around 100."""
    cars = make_cars(n, seed)
    power = np.random.default_rng(seed + 1).integers(60, 200, size=len(cars))
    cars['Puissance'] = power
    cars['Prix'] = (cars['Prix'] * (1 + 0.005 * (power - 100))).astype(int)
    return cars


def make_cars_with_battery(n=400, seed=0):
    """Every third car is electric, with a battery of 40 to 100 kWh that changes its price by 1 % per kWh around 60."""
    cars = make_cars_with_power(n, seed)
    battery = np.random.default_rng(seed + 2).integers(40, 101, size=len(cars)).astype(float)
    electric = np.arange(len(cars)) % 3 == 0
    cars['Carburant'] = np.where(electric, 'Électrique', cars['Carburant'])
    cars['Batterie'] = np.where(electric, battery, np.nan)
    cars['Prix'] = np.where(electric, cars['Prix'] * (1 + 0.01 * (battery - 60)), cars['Prix']).astype(int)
    return cars


# The price of each family of trim, compared with the basic one
TRIM_FAMILIES = {'Active': 0.9, 'Active Pack': 0.9, 'Allure': 1.0, 'Allure Pack': 1.0, 'GT': 1.2, 'GT Line': 1.2}


def make_cars_with_trim(n=400, seed=0):
    """The same cars with a trim: an 'Active' is 10 % cheaper than an 'Allure', a 'GT' 20 % more expensive."""
    cars = make_cars_with_power(n, seed)
    trims = np.random.default_rng(seed + 3).choice(list(TRIM_FAMILIES), size=len(cars))
    cars['Finition'] = trims
    cars['Prix'] = (cars['Prix'] * np.array([TRIM_FAMILIES[trim] for trim in trims])).astype(int)
    return cars


@pytest.fixture(scope='module')
def cars():
    return make_cars()


@pytest.fixture(scope='module')
def trained(cars):
    return train_and_evaluate(cars)


@pytest.fixture(scope='module')
def cars_200():
    return make_cars(200, seed=3)


@pytest.fixture(scope='module')
def comparison(cars_200):
    """The comparison of all the models takes a few seconds: it is made once for all the tests."""
    return compare_models(cars_200)


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
    assert suv > city_car + 3000  # the real difference is about 6 500 EUR, the model finds 4 800 to 6 500


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

    metrics = main([str(csv), '--kind', 'forest', '--model', str(destination)])

    assert destination.exists()
    assert metrics['r2'] > 0.9
    assert 'Mean error' in capsys.readouterr().out


# ---------------------------------------------------------------- kinds of models, comparison


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_every_kind_of_model_learns_the_price_rule(cars, kind):
    model, metrics = train_and_evaluate(cars, kind=kind)
    assert metrics['r2'] > 0.8, kind
    assert metrics['mae'] < metrics['baseline_mae'] / 2
    assert 0 < metrics['mape'] < 20
    price = predict_price(model, 'Peugeot 208', 'Essence', 'Manuelle', km=60_000, year=2019)
    assert 3_000 < price < 20_000


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match='Unknown kind'):
        build_model('magic')


def test_compare_models_gives_a_sorted_table_with_the_average_last(comparison):
    table = comparison
    assert set(table['model']) == {'average', *MODEL_KINDS}
    assert list(table['mae']) == sorted(table['mae'])
    assert table.iloc[-1]['model'] == 'average'  # always answering the mean price is the worst
    assert set(table.columns) == {'model', 'mae', 'mae_std', 'rmse', 'r2', 'mape'}
    assert table[table['model'] != 'average']['r2'].min() > 0.8
    assert table[table['model'] == 'average'].iloc[0]['r2'] == pytest.approx(0, abs=0.05)


def test_compare_models_is_reproducible(cars_200):
    pd.testing.assert_frame_equal(
        compare_models(cars_200, kinds=('ridge',)), compare_models(cars_200, kinds=('ridge',))
    )


def test_best_kind_never_answers_average(comparison):
    table = comparison
    assert best_kind(table) in MODEL_KINDS
    assert best_kind(table) == table[table['model'] != 'average'].iloc[0]['model']


def test_compare_models_needs_enough_cars():
    with pytest.raises(ValueError, match='At least'):
        compare_models(make_cars(10))


# ---------------------------------------------------------------- the labels of the boosting


def test_the_boosting_reads_brand_fuel_and_gearbox_as_categories(cars):
    model = train_final_model(cars, kind='boosting')
    boosting = model.named_steps['model'].regressor_
    categories = list(boosting.feature_names_in_[boosting.is_categorical_])
    assert categories == ['Marque', 'Carburant', 'Boite de vitesse']  # not the mileage, the year or the model


def test_the_boosting_replaces_the_model_by_its_mean_price(cars):
    model = train_final_model(cars, kind='boosting')
    preprocessing = model.named_steps['preprocessing']
    names = pd.DataFrame({'Nom': ['Peugeot 3008', 'Renault Clio']})
    encoded = np.asarray(preprocessing.named_transformers_['names'].transform(names))[:, 0]
    assert encoded[0] > encoded[1] * 1.5  # a 3008 (30 000 EUR new) is worth much more than a Clio (14 000 EUR)


def test_the_boosting_gives_the_same_model_at_every_run(cars):
    car = cars.head(5)
    first = train_final_model(cars, kind='boosting').predict(prepare_features(car))
    second = train_final_model(cars, kind='boosting').predict(prepare_features(car))
    np.testing.assert_allclose(first, second)


def test_the_boosting_works_without_any_label(cars):
    model = build_model('boosting', 42, ['Km', 'année']).fit(cars[['Km', 'année']], cars['Prix'])
    assert model.predict(cars[['Km', 'année']].head(1))[0] > 0


@pytest.mark.parametrize('version', ['1.9.1', '1.10.0', '2.0.0'])
def test_the_target_encoder_takes_its_seed_in_the_cross_validation_from_scikit_learn_1_9(version):
    encoder = target_encoder(7, version)
    assert encoder.cv.random_state == 7 and encoder.cv.shuffle


@pytest.mark.parametrize('version', ['1.5.0', '1.8.2'])
def test_the_target_encoder_takes_its_seed_directly_before_scikit_learn_1_9(version):
    encoder = target_encoder(7, version)
    assert encoder.random_state == 7 and encoder.shuffle and encoder.cv == 5


# ---------------------------------------------------------------- power


def test_power_is_used_when_the_data_has_it():
    cars = make_cars_with_power()
    assert feature_columns(cars)[-1] == 'Puissance'
    assert feature_columns(make_cars(5)) == ['Marque', 'Nom', 'Carburant', 'Boite de vitesse', 'Km', 'année']
    assert 'Marque' in prepare_features(cars).columns and 'Puissance' in prepare_features(cars).columns


def test_power_is_ignored_when_most_values_are_missing():
    cars = make_cars_with_power(100)
    cars.loc[cars.index[:70], 'Puissance'] = np.nan  # 70 % missing
    assert 'Puissance' not in feature_columns(cars)


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_a_more_powerful_car_is_more_expensive(kind):
    model = train_final_model(make_cars_with_power(), kind=kind)
    args = ('Opel Astra', 'Essence', 'Manuelle', 60_000, 2019)
    weak = predict_price(model, *args, power=65)
    strong = predict_price(model, *args, power=190)
    assert strong > weak + 1500


def test_the_power_given_to_a_model_trained_without_it_is_ignored(cars):
    model = train_final_model(cars)
    args = ('Opel Astra', 'Essence', 'Manuelle', 60_000, 2019)
    assert predict_price(model, *args, power=150) == pytest.approx(predict_price(model, *args))


def test_unknown_power_is_replaced_by_the_median():
    cars = make_cars_with_power()
    cars.loc[cars.index[::5], 'Puissance'] = np.nan  # 20 % missing: still used
    model, metrics = train_and_evaluate(cars)
    assert metrics['r2'] > 0.85
    price = predict_price(model, 'Opel Astra', 'Essence', 'Manuelle', 60_000, 2019)  # no power given
    assert 3_000 < price < 25_000


# ---------------------------------------------------------------- battery


def test_battery_is_used_when_the_electric_cars_have_it():
    cars = make_cars_with_battery()
    assert feature_columns(cars)[-2:] == ['Puissance', 'Batterie']
    assert 'Batterie' in prepare_features(cars).columns


def test_battery_is_judged_on_the_electric_cars_not_on_all_the_cars():
    cars = make_cars_with_battery()
    assert cars['Batterie'].notna().mean() < 0.5  # two thirds of the cars have no battery, and that is normal
    assert 'Batterie' in feature_columns(cars)


def test_battery_is_ignored_when_most_electric_cars_have_no_value():
    cars = make_cars_with_battery()
    electric = cars.index[cars['Carburant'] == 'Électrique']
    cars.loc[electric[: int(len(electric) * 0.7)], 'Batterie'] = np.nan
    assert 'Batterie' not in feature_columns(cars)


def test_battery_is_ignored_when_there_are_too_few_electric_cars():
    cars = make_cars_with_battery()
    electric = cars.index[cars['Carburant'] == 'Électrique']
    cars.loc[electric[5:], 'Carburant'] = 'Essence'  # 5 electric cars left
    assert 'Batterie' not in feature_columns(cars)


def test_battery_is_ignored_when_the_data_has_no_fuel_column():
    cars = make_cars_with_battery().drop(columns=['Carburant'])
    assert 'Batterie' not in feature_columns(cars)


# The forest shares its trees between many columns, so with 600 cars it takes less from the battery (+555 EUR for
# +60 kWh, against +4 500 EUR for the boosting and +7 300 EUR for the linear model): it only has to go the right way
MINIMUM_RISE = {'forest': 200, 'boosting': 1500, 'ridge': 1500}


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_an_electric_car_with_a_bigger_battery_is_more_expensive(kind):
    model = train_final_model(make_cars_with_battery(600), kind=kind)
    args = ('Opel Astra', 'Électrique', 'Automatique', 40_000, 2022)
    small = predict_price(model, *args, battery=40)
    large = predict_price(model, *args, battery=100)
    assert large > small + MINIMUM_RISE[kind]


def test_the_battery_given_to_a_model_trained_without_it_is_ignored(cars):
    model = train_final_model(cars)
    args = ('Opel Astra', 'Électrique', 'Automatique', 60_000, 2019)
    assert predict_price(model, *args, battery=80) == pytest.approx(predict_price(model, *args))


def test_an_unknown_battery_still_gives_a_price():
    model = train_final_model(make_cars_with_battery(), kind='boosting')
    price = predict_price(model, 'Opel Astra', 'Électrique', 'Automatique', 40_000, 2022)
    assert 3_000 < price < 40_000


# ---------------------------------------------------------------- trim


def test_the_family_of_a_trim_is_its_first_word_in_small_letters():
    cars = pd.DataFrame({'Finition': ['Allure Pack', 'GT Line', 'gt', '   ', '', None, np.nan]})
    families = add_trim_family(cars)['Gamme']
    assert families[:3].tolist() == ['allure', 'gt', 'gt']
    assert families[3:].isna().all()  # an empty trim is an unknown family


def test_without_trim_the_cars_are_unchanged(cars):
    assert add_trim_family(cars) is cars
    assert 'Gamme' not in feature_columns(cars)


def test_a_family_already_given_is_kept():
    cars = pd.DataFrame({'Finition': ['Allure Pack'], 'Gamme': ['other']})
    assert add_trim_family(cars)['Gamme'].tolist() == ['other']


def test_the_family_is_used_when_most_cars_have_a_trim():
    cars = make_cars_with_trim()
    assert feature_columns(cars)[-1] == 'Gamme'
    assert 'Gamme' in prepare_features(cars).columns
    cars.loc[cars.index[: 70 * len(cars) // 100], 'Finition'] = None  # 70 % unknown
    assert 'Gamme' not in feature_columns(cars)


# The forest shares its trees between many columns and takes less from the trim (+5 % for +33 % with 600 cars)
MINIMUM_TRIM_RISE = {'forest': 1.03, 'boosting': 1.1, 'ridge': 1.1}


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_a_better_equipped_car_is_more_expensive(kind):
    model = train_final_model(make_cars_with_trim(600), kind=kind)
    args = ('Opel Astra', 'Essence', 'Manuelle', 60_000, 2019)
    basic = predict_price(model, *args, trim='Active')
    sport = predict_price(model, *args, trim='GT Line')
    assert sport > basic * MINIMUM_TRIM_RISE[kind]
    # only the first word counts, whatever the case
    assert predict_price(model, *args, trim='gt pack') == pytest.approx(sport)


def test_an_unknown_or_missing_trim_still_gives_a_price():
    model = train_final_model(make_cars_with_trim(), kind='boosting')
    args = ('Opel Astra', 'Essence', 'Manuelle', 60_000, 2019)
    assert 3_000 < predict_price(model, *args, trim='Never Seen') < 40_000
    assert 3_000 < predict_price(model, *args) < 40_000  # no trim at all: it must not crash


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_every_model_trained_with_the_trim_works_without_it(kind):
    model = train_final_model(make_cars_with_trim(), kind=kind)
    assert predict_price(model, 'Opel Astra', 'Essence', 'Manuelle', 60_000, 2019, trim=None) > 0


def test_the_trim_given_to_a_model_trained_without_it_is_ignored(cars):
    model = train_final_model(cars)
    args = ('Opel Astra', 'Essence', 'Manuelle', 60_000, 2019)
    assert predict_price(model, *args, trim='GT Line') == pytest.approx(predict_price(model, *args))


def test_the_boosting_accepts_more_families_than_it_has_categories():
    cars = make_cars_with_trim(600)
    cars['Finition'] = [f'Family{number}' for number in range(len(cars))]  # 600 families, the limit is 255
    model = train_final_model(cars, kind='boosting')
    assert predict_price(model, 'Opel Astra', 'Essence', 'Manuelle', 60_000, 2019, trim='Family3') > 0


# ---------------------------------------------------------------- importance


def test_feature_importance_finds_what_matters():
    table = feature_importance(make_cars_with_power(), n_repeats=3)
    assert list(table.columns) == ['column', 'importance', 'std']
    assert list(table['importance']) == sorted(table['importance'], reverse=True)
    assert set(table['column']) == {'Marque', 'Nom', 'Carburant', 'Boite de vitesse', 'Km', 'année', 'Puissance'}
    importance = table.set_index('column')['importance']
    assert importance['Puissance'] > 300  # the price depends on the power...
    assert importance['année'] > importance['Boite de vitesse']  # ...and on the age, not on the gearbox
    assert abs(importance['Boite de vitesse']) < 100


# ---------------------------------------------------------------- command line


def test_main_compares_then_saves_the_best_by_default(cars_200, tmp_path, capsys):
    csv = tmp_path / 'cars_clean.csv'
    cars_200.to_csv(csv, index=False)
    destination = tmp_path / 'out' / 'best.joblib'
    main([str(csv), '--model', str(destination)])
    output = capsys.readouterr().out
    assert 'Comparison' in output and 'average' in output and 'Model: ' in output
    assert load_model(str(destination)) is not None


def test_main_compare_only_does_not_save(cars_200, tmp_path, capsys):
    csv = tmp_path / 'cars_clean.csv'
    cars_200.to_csv(csv, index=False)
    destination = tmp_path / 'never.joblib'
    table = main([str(csv), '--compare', '--model', str(destination)])
    assert not destination.exists()
    assert 'Comparison' in capsys.readouterr().out
    assert 'model' in table.columns


def test_main_with_a_chosen_kind_does_not_compare(cars_200, tmp_path, capsys):
    csv = tmp_path / 'cars_clean.csv'
    cars_200.to_csv(csv, index=False)
    destination = tmp_path / 'ridge.joblib'
    main([str(csv), '--kind', 'ridge', '--model', str(destination)])
    output = capsys.readouterr().out
    assert 'Comparison' not in output and 'Model: ridge' in output
    assert destination.exists()


def test_main_unknown_kind_is_refused(cars_200, tmp_path):
    csv = tmp_path / 'cars_clean.csv'
    cars_200.to_csv(csv, index=False)
    with pytest.raises(SystemExit) as stop:
        main([str(csv), '--kind', 'magic'])
    assert stop.value.code == 2
