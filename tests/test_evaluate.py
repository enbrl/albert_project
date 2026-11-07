"""Tests of scripts/evaluate.py: the error of each group of cars."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.cli import UserError, run
from scripts.evaluate import (
    TITLES,
    add_segments,
    age_labels,
    errors_by_group,
    errors_by_model_rarity,
    evaluate,
    format_report,
    main,
    predict_out_of_fold,
    to_json,
)
from scripts.models import build_model, feature_columns, prepare_features
from tests.test_models import make_cars_with_power


@pytest.fixture(scope='module')
def cars():
    return make_cars_with_power(300, seed=6)


@pytest.fixture(scope='module')
def result(cars):
    return evaluate(cars, kind='ridge')


def small_cars():
    """6 cars with known prices and years, to check the arithmetic by hand."""
    return pd.DataFrame(
        {
            'Nom': ['Peugeot 208'] * 4 + ['Rare Car'] * 2,
            'Marque': ['Peugeot'] * 4 + ['Rare'] * 2,
            'Carburant': ['Essence', 'Essence', 'Diesel', 'Diesel', 'Essence', 'Essence'],
            'Km': [1000, 20000, 50000, 90000, 5000, 6000],
            'année': [2026, 2025, 2022, 2018, 2025, 2025],
            'Prix': [10000, 10000, 10000, 10000, 20000, 20000],
        }
    )


# ---------------------------------------------------------------- the estimates


def test_every_car_gets_an_estimate_from_a_model_that_did_not_learn_from_it(cars):
    estimates = predict_out_of_fold(cars, kind='ridge')
    assert list(estimates.index) == list(cars.index) and len(estimates) == 300
    # honest (cross-validation): a little worse than the same model judged on the cars it learnt from
    columns = feature_columns(cars)
    fitted = build_model('ridge', 42, columns).fit(prepare_features(cars, columns), cars['Prix'])
    in_sample = np.abs(fitted.predict(prepare_features(cars, columns)) - cars['Prix']).mean()
    assert np.abs(estimates - cars['Prix']).mean() >= in_sample


def test_the_estimates_are_the_same_at_every_run(cars):
    pd.testing.assert_series_equal(predict_out_of_fold(cars, kind='ridge'), predict_out_of_fold(cars, kind='ridge'))


def test_too_few_cars_are_refused():
    with pytest.raises(UserError, match='At least 20 cars'):
        predict_out_of_fold(small_cars())


# ---------------------------------------------------------------- the groups


def test_every_car_is_in_one_group_of_each_segment(cars):
    segments = add_segments(cars)
    for column in ('fuel', 'age', 'price', 'mileage', 'brand'):
        assert segments[column].notna().all(), column


def test_the_ages_are_counted_from_the_newest_car_not_from_today():
    segments = add_segments(small_cars())
    assert list(segments['age'].astype(str)) == ['2026 (newest)', '2025', '2020 to 2022', 'before 2020', '2025', '2025']
    assert age_labels(2026) == ['2026 (newest)', '2025', '2023 to 2024', '2020 to 2022', 'before 2020']
    assert age_labels(2030)[0] == '2030 (newest)'  # the labels follow the data


def test_the_mileage_groups():
    segments = add_segments(small_cars())
    assert list(segments['mileage'].astype(str)) == [
        'under 5 000 km',
        '5 000 to 30 000 km',
        '30 000 to 80 000 km',
        '80 000 km or more',
        '5 000 to 30 000 km',
        '5 000 to 30 000 km',
    ]


def test_the_price_quarters_have_the_same_size_even_when_prices_are_equal(cars):
    sizes = add_segments(cars)['price'].value_counts()
    assert sorted(sizes) == [75, 75, 75, 75]
    # many cars at the same price must not make the quarters fail
    same = cars.assign(Prix=10_000)
    assert add_segments(same)['price'].notna().all()


def test_the_brand_is_the_first_word_when_the_column_is_missing(cars):
    assert 'Marque' not in cars.columns
    assert set(add_segments(cars)['brand']) == {'Peugeot', 'Renault', 'Opel'}


# ---------------------------------------------------------------- the arithmetic of a group


def test_the_error_of_a_group_is_computed_by_hand():
    frame = small_cars()
    estimates = pd.Series(
        [11000, 9000, 10000, 12000, 18000, 21000], index=frame.index
    )  # errors +1000 -1000 0 +2000 -2000 +1000
    table = errors_by_group(add_segments(frame), estimates, 'fuel', min_cars=1).set_index('group')
    essence, diesel = table.loc['Essence'], table.loc['Diesel']
    # Essence: errors +1000, -1000, -2000, +1000 on prices 10000, 10000, 20000, 20000
    assert essence['cars'] == 4
    assert essence['mae'] == pytest.approx(1250)
    assert essence['mape'] == pytest.approx((10 + 10 + 10 + 5) / 4)
    assert essence['bias'] == pytest.approx(-250)  # negative: it estimates too low
    # Diesel: errors 0 and +2000 on prices 10000, 10000
    assert diesel['cars'] == 2 and diesel['mae'] == pytest.approx(1000) and diesel['bias'] == pytest.approx(1000)


def test_a_group_with_few_cars_is_left_out():
    frame = small_cars()
    estimates = frame['Prix'] + 100
    assert list(errors_by_group(add_segments(frame), estimates, 'fuel', min_cars=3)['group']) == ['Essence']
    assert errors_by_group(add_segments(frame), estimates, 'fuel', min_cars=10).empty


def test_fuel_and_brand_are_sorted_by_error_the_worst_first():
    frame = small_cars()
    estimates = frame['Prix'] + np.array([100, 100, 5000, 5000, 100, 100])  # Diesel is the badly predicted fuel
    table = errors_by_group(add_segments(frame), estimates, 'fuel', min_cars=1)
    assert list(table['group']) == ['Diesel', 'Essence']
    assert list(table['mape']) == sorted(table['mape'], reverse=True)


def test_age_price_and_mileage_keep_their_natural_order():
    frame = small_cars()
    estimates = frame['Prix'] + np.array(
        [5000, 100, 100, 100, 100, 100]
    )  # the newest group is the worst, yet it stays first
    table = errors_by_group(add_segments(frame), estimates, 'age', min_cars=1)
    assert list(table['group']) == ['2026 (newest)', '2025', '2020 to 2022', 'before 2020']
    prices = errors_by_group(
        add_segments(small_cars().assign(Prix=range(1000, 7000, 1000))), estimates, 'price', min_cars=1
    )
    assert list(prices['group']) == ['cheapest quarter', 'second quarter', 'third quarter', 'most expensive quarter']


def test_rare_models_are_told_apart_from_common_ones():
    frame = small_cars()  # 'Peugeot 208' has 4 cars, 'Rare Car' has 2
    estimates = frame['Prix'] + np.array([100, 100, 100, 100, 3000, 3000])
    table = errors_by_model_rarity(add_segments(frame), estimates, min_cars=3).set_index('group')
    assert set(table.index) == {'rare model (under 3 cars)', 'common model (3+ cars)'}
    assert table.loc['rare model (under 3 cars)', 'cars'] == 2
    assert table.loc['rare model (under 3 cars)', 'mae'] == pytest.approx(3000)
    assert table.loc['common model (3+ cars)', 'mae'] == pytest.approx(100)


# ---------------------------------------------------------------- the whole evaluation


def test_the_groups_add_up_to_the_overall_error(cars):
    result = evaluate(cars, kind='ridge', min_cars=1)  # no group is left out
    overall = result['overall']
    assert overall['cars'] == 300
    for column in ('fuel', 'age', 'price', 'mileage', 'brand', 'model_rarity'):
        table = result[column]
        assert table['cars'].sum() == 300, column
        assert (table['cars'] * table['mae']).sum() / 300 == pytest.approx(overall['mae']), column
        assert (table['cars'] * table['bias']).sum() / 300 == pytest.approx(overall['bias']), column


def test_the_model_is_good_on_this_made_up_market(result):
    assert result['overall']['mae'] < 1500 and result['overall']['mape'] < 15


def test_the_result_has_every_segment(result):
    assert set(result) == {'overall', *TITLES}


def test_the_report_has_one_table_per_segment(result):
    text = format_report(result, 'ridge')
    assert text.startswith('Model: ridge. Cars: 300.')
    for title in TITLES.values():
        assert title in text
    assert 'bias' in text and 'negative: estimates too low' in text


def test_the_json_can_be_saved_and_read_back(result):
    data = json.loads(json.dumps(to_json(result, 'ridge')))
    assert data['model'] == 'ridge' and data['overall']['cars'] == 300
    assert set(data) == {'model', 'overall', *TITLES}
    assert {'group', 'cars', 'mae', 'mape', 'bias'} <= set(data['fuel'][0])


# ---------------------------------------------------------------- the command line


def test_main_prints_the_report_and_saves_the_json(cars, tmp_path, capsys):
    csv = tmp_path / 'cars_clean.csv'
    cars.to_csv(csv, index=False)
    destination = tmp_path / 'reports' / 'evaluation.json'  # the folder does not exist yet
    main([str(csv), '--kind', 'ridge', '--output', str(destination)])
    assert 'By fuel:' in capsys.readouterr().out
    assert json.loads(destination.read_text())['overall']['cars'] == 300


def test_main_can_leave_out_more_groups(cars, tmp_path, capsys):
    csv = tmp_path / 'cars_clean.csv'
    cars.to_csv(csv, index=False)
    main([str(csv), '--kind', 'ridge', '--min-cars', '1000'])
    assert 'Empty DataFrame' in capsys.readouterr().out  # every group has fewer than 1000 cars


@pytest.mark.parametrize('value', ['0', '-3', 'many'])
def test_main_refuses_an_impossible_min_cars(cars, tmp_path, value):
    csv = tmp_path / 'cars_clean.csv'
    cars.to_csv(csv, index=False)
    with pytest.raises(SystemExit) as stop:
        main([str(csv), '--min-cars', value])
    assert stop.value.code == 2


def test_a_missing_file_gives_a_message_not_a_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr('sys.argv', ['evaluate', str(tmp_path / 'nothing.csv')])
    with pytest.raises(SystemExit) as stop:
        run(main)
    assert stop.value.code == 1
    assert capsys.readouterr().err.startswith('Error: File not found:')
