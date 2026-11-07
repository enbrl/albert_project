"""Tests of scripts/model_card.py: the model card is written from measures, never from numbers typed by hand."""

import re

import numpy as np
import pandas as pd
import pytest

from scripts.cli import run
from scripts.evaluate import evaluate
from scripts.model_card import (
    BIGGEST_BRANDS,
    MODEL_NAMES,
    WORST_BRANDS,
    brand_rows,
    count,
    describe_data,
    euros,
    main,
    measure,
    percent,
    signed,
    worst_cars,
    write_card,
)
from tests.test_models import make_cars, make_cars_with_trim

VERSIONS = {'python': '3.13.0', 'scikit-learn': '1.9.1', 'pandas': '3.1.1', 'numpy': '2.5.4'}


def make_listing(n=300, seed=8):
    """Made-up cars with every column of the scraper: power, trim, and electric cars with a battery."""
    cars = make_cars_with_trim(n, seed)
    electric = np.arange(len(cars)) % 4 == 0
    cars['Carburant'] = np.where(electric, 'Électrique', cars['Carburant'])
    cars['Batterie'] = np.where(electric & (np.arange(len(cars)) % 8 == 0), 50.0, np.nan)
    cars.loc[cars.index[:15], 'Puissance'] = np.nan  # 15 cars without power
    cars.loc[cars.index[1], ['Carburant', 'Batterie']] = ['Hybride rechargeable', 10.0]  # not counted as electric
    cars.loc[cars.index[-6:], 'Nom'] = [f'Opel Rare{number // 2}' for number in range(6)]  # 3 rare models, 2 cars each
    return cars


@pytest.fixture(scope='module')
def cars():
    return make_listing()


@pytest.fixture(scope='module')
def results(cars):
    return measure(cars)


@pytest.fixture(scope='module')
def card(results):
    return write_card(results, '2026-11-07', VERSIONS)


# ---------------------------------------------------------------- how numbers are written


def test_numbers_are_written_like_in_the_rest_of_the_card():
    assert euros(12345.6) == '12 346 €'
    assert count(2693) == '2 693'
    assert signed(-921.4) == '-921 €' and signed(340) == '+340 €'
    assert percent(60, 2693) == '2.2 %' and percent(1, 0) == '0.0 %'


# ---------------------------------------------------------------- the measures


def test_the_data_is_described_from_the_cars(cars):
    data = describe_data(cars)
    assert data['cars'] == 300 and data['models'] == cars['Nom'].nunique()
    assert data['price_min'] == cars['Prix'].min() and data['price_max'] == cars['Prix'].max()
    assert data['power'] == 285
    assert data['families'] == 3  # Active, Allure and GT: the first word of the trims
    assert data['electric'] == 75 and data['battery_electric'] == 38


def test_the_optional_columns_are_described_only_when_the_data_has_them():
    data = describe_data(make_cars(50))
    assert not {'power', 'families', 'electric'} & set(data)


def test_the_worst_cars_are_the_biggest_errors_first(cars):
    estimates = (
        cars['Prix'] + np.where(np.arange(len(cars)) == 7, 9000, 0) + np.where(np.arange(len(cars)) == 3, -5000, 0)
    )
    worst = worst_cars(cars, pd.Series(estimates, index=cars.index), number=2)
    assert [car['estimate'] - car['price'] for car in worst] == [9000, -5000]


def test_the_brands_shown_are_the_biggest_and_the_worst():
    table = pd.DataFrame(
        {
            'group': [f'Brand{number}' for number in range(20)],
            'cars': list(range(100, 80, -1)),  # Brand0 has the most cars
            'mape': list(range(20)),  # Brand19 is the worst estimated
            'mae': 1000.0,
            'bias': 0.0,
        }
    )
    groups = [row['group'] for row in brand_rows(table)]
    assert len(groups) == BIGGEST_BRANDS + WORST_BRANDS
    assert groups[0] == 'Brand19' and 'Brand0' in groups and 'Brand12' not in groups  # Brand12: neither


def test_evaluate_uses_the_estimates_it_is_given(cars):
    result = evaluate(cars, kind='ridge', estimates=cars['Prix'] + 100)
    assert result['overall']['mae'] == pytest.approx(100) and result['overall']['bias'] == pytest.approx(100)


# ---------------------------------------------------------------- the card


def test_the_card_has_every_section(card):
    for title in ('## Summary', '## Intended use', '## Data', '## Model', '## Where the model is wrong'):
        assert title in card
    for title in ('## Limits and risks', '## Reproduce', '**By price**', '**By fuel**', '**By brand**'):
        assert title in card


def test_the_numbers_of_the_card_are_the_measured_ones(card, results):
    overall = results['evaluation']['overall']
    assert f'wrong by **{euros(overall["mae"])} on average, {overall["mape"]:.1f} % of the price**' in card
    average = results['comparison'].set_index('model').loc['average']
    assert f'would be wrong by {euros(average["mae"])}' in card
    best = results['comparison'].iloc[0]  # the lowest error: the model kept, like python -m scripts.models does
    assert best['model'] == results['kind']
    assert f'**{MODEL_NAMES[best["model"]]}, kept** | **{euros(best["mae"])}**' in card
    assert card.count(', kept**') == 1
    for row in results['importance'].to_dict('records')[:3]:
        assert euros(row['importance']) in card


def test_the_card_gives_the_date_and_the_versions(card):
    assert card.count('2026-11-07') == 4  # the top, the data, the limits and the command to reproduce
    assert 'Python 3.13.0, scikit-learn 1.9.1, pandas 3.1.1 and numpy 2.5.4' in card


def test_the_optional_rows_follow_the_data(card):
    assert '| Power | found in the text of the engine for 95.0 % of the cars' in card
    assert '**Powers that cannot be read**: 15 cars (5.0 %)' in card
    assert 'the family (`Allure Pack` -> `allure`: 3 families)' in card
    assert 'for 38 of the 75 electric cars' in card


def test_a_card_without_optional_columns_does_not_mention_them():
    text = write_card(measure(make_cars(120, seed=2)), '2026-11-07', VERSIONS)
    for word in ('| Power |', '| Trim |', '| Battery |', 'Powers that cannot be read'):
        assert word not in text


def test_the_rare_models_are_compared_with_the_common_ones(card, results):
    rarity = results['evaluation']['model_rarity'].set_index('group')
    rare, common = rarity.loc['rare model (under 5 cars)'], rarity.loc['common model (5+ cars)']
    assert rare['cars'] == 6
    words = 'too low' if rare['bias'] < 0 else 'too high'
    assert f'error **{rare["mae"] / common["mae"]:.1f} times** that of a common model, and {words}' in card
    assert '6 cars of the data are in this case' in card


def test_the_bias_is_said_in_words_with_its_sign(card, results):
    prices = results['evaluation']['price'].set_index('group')
    cheap = prices.loc['cheapest quarter', 'bias']
    words = 'too low' if cheap < 0 else 'too high'
    assert f'the cheapest quarter {words} by {euros(abs(cheap))}' in card


def test_the_boosting_note_is_only_for_the_boosting(results):
    note = 'reads brand, fuel and gearbox as categories'
    assert note in write_card(dict(results, kind='boosting'), '2026-11-07', VERSIONS)
    assert note not in write_card(dict(results, kind='ridge'), '2026-11-07', VERSIONS)


def test_a_chosen_model_can_be_described_instead_of_the_best(cars):
    assert measure(cars.head(120), kind='forest')['kind'] == 'forest'


def test_no_number_is_left_to_fill(card):
    assert '@' not in card and 'nan' not in card.lower().replace('financ', '')
    assert not re.search(r'\{[a-z_]+\}', card)  # no format placeholder forgotten


# ---------------------------------------------------------------- the command line


def test_main_writes_the_card(tmp_path, capsys):
    csv = tmp_path / 'cars_clean.csv'
    make_listing(150, seed=3).to_csv(csv, index=False)
    output = tmp_path / 'docs' / 'card.md'  # the folder does not exist yet
    text = main([str(csv), '--scraped', '2026-10-01', '--output', str(output)])
    assert output.read_text(encoding='utf-8') == text
    assert 'data scraped on **2026-10-01**' in text
    assert 'Model card written in' in capsys.readouterr().out


@pytest.mark.parametrize('value', ['yesterday', '2026-13-01', '07/11/2026'])
def test_main_refuses_a_date_that_is_not_a_date(tmp_path, value, capsys):
    with pytest.raises(SystemExit) as stop:
        main([str(tmp_path / 'cars.csv'), '--scraped', value])
    assert stop.value.code == 2
    assert 'is not a date like 2026-11-07' in capsys.readouterr().err


def test_a_missing_file_gives_a_message_not_a_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr('sys.argv', ['model_card', str(tmp_path / 'nothing.csv')])
    with pytest.raises(SystemExit) as stop:
        run(main)
    assert stop.value.code == 1
    assert capsys.readouterr().err.startswith('Error: File not found:')
