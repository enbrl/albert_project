"""Tests of scripts/datacleaning.py."""
import pandas as pd
import pytest

from scripts.datacleaning import (
    COLUMNS,
    UNKNOWN,
    clean_cars,
    main,
    normalize_fuel,
    normalize_gearbox,
    normalize_name,
    normalize_text,
    to_number,
)


def make_df(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_to_number():
    assert to_number('57 844 km') == 57844
    assert to_number('18 690 €') == 18690
    assert to_number('18 690\xa0€') == 18690
    assert to_number(19160) == 19160
    assert to_number(2021.0) == 2021
    assert to_number('2021') == 2021


def test_to_number_without_value():
    assert to_number(None) is None
    assert to_number(float('nan')) is None
    assert to_number('') is None
    assert to_number('inconnu') is None


def test_normalize_name():
    assert normalize_name('  PEUGEOT   308 ') == 'Peugeot 308'
    assert normalize_name('Citroen C4') == 'Citroen C4'
    assert normalize_name('MG ZS') == 'MG ZS'
    assert normalize_name('MINI MINI') == 'Mini Mini'
    assert normalize_name('') is None
    assert normalize_name(None) is None


@pytest.mark.parametrize('raw, expected', [
    ('Essence', 'Essence'),
    ('Diesel', 'Diesel'),
    ('Électrique', 'Électrique'),
    ('electrique', 'Électrique'),
    ('Hybride essence', 'Hybride'),
    ('Mild Hybrid Essence', 'Hybride'),
    ('Hybride rechargeable essence', 'Hybride rechargeable'),
    ('GPL', 'GPL'),
    (None, UNKNOWN),
    ('', UNKNOWN),
])
def test_normalize_fuel(raw, expected):
    assert normalize_fuel(raw) == expected


@pytest.mark.parametrize('raw, expected', [
    ('Manuelle', 'Manuelle'),
    ('Automatique', 'Automatique'),
    ('Auto.', 'Automatique'),
    ('Double embrayage / DCT', 'Automatique'),
    (None, UNKNOWN),
    ('???', UNKNOWN),
])
def test_normalize_gearbox(raw, expected):
    assert normalize_gearbox(raw) == expected


def test_clean_cars_harmonises_the_messy_rows_of_the_notebooks():
    # Rows copied from the outputs of the notebooks: numbers and labels in different formats
    df = make_df([
        ['OPEL Astra', 'Essence', 'Manuelle', 19160, '2022', 19999],
        ['Citroen C4', 'Diesel', 'Automatique', '57 844 km', '2021', '18 690 €'],
        ['Renault Captur', 'Diesel', 'Double embrayage / DCT', '104 582 km', 2016, '11 690 €'],
    ])
    cleaned = clean_cars(df, current_year=2026)
    assert len(cleaned) == 3
    assert list(cleaned['Km']) == [19160, 57844, 104582]
    assert list(cleaned['Prix']) == [19999, 18690, 11690]
    assert list(cleaned['année']) == [2022, 2021, 2016]
    assert list(cleaned['Boite de vitesse']) == ['Manuelle', 'Automatique', 'Automatique']
    assert list(cleaned['Nom']) == ['Opel Astra', 'Citroen C4', 'Renault Captur']
    assert list(cleaned['Marque']) == ['Opel', 'Citroen', 'Renault']
    for column in ('Km', 'année', 'Prix'):
        assert cleaned[column].dtype == 'int64'


def test_clean_cars_removes_duplicates_after_harmonisation():
    df = make_df([
        ['PEUGEOT 308', 'Essence', 'Manuelle', 79049, 2019, 16999],
        ['Peugeot 308', 'Essence', 'Manuelle', '79 049 km', '2019', '16 999 €'],
    ])
    assert len(clean_cars(df, current_year=2026)) == 1


def test_clean_cars_keeps_one_row_per_id():
    df = make_df([
        ['Peugeot 208', 'Essence', 'Manuelle', 50000, 2019, 10000],
        ['Peugeot 208', 'Essence', 'Manuelle', 50000, 2019, 9900],  # same car, the price changed
        ['Renault Clio', 'Diesel', 'Manuelle', 30000, 2021, 12000],
        ['Renault Clio', 'Diesel', 'Manuelle', 30000, 2021, 12000],  # no Id: only exact copies are removed
    ])
    df.insert(0, 'Id', [1036829, 1036829, None, None])
    cleaned = clean_cars(df, current_year=2026)
    assert len(cleaned) == 2
    assert list(cleaned['Prix']) == [10000, 12000]  # the first copy is kept
    assert cleaned.columns[0] == 'Id'


def test_clean_cars_without_id_column_has_no_id_column():
    cleaned = clean_cars(make_df([['Peugeot 208', 'Essence', 'Manuelle', 50000, 2019, 10000]]), current_year=2026)
    assert 'Id' not in cleaned.columns


def test_clean_cars_cleans_the_engine_columns_when_present():
    df = make_df([
        ['Peugeot 208', 'Essence', 'Manuelle', 50000, 2019, 10000],
        ['Peugeot 208', 'Essence', 'Manuelle', 51000, 2019, 10100],
        ['Peugeot 208', 'Essence', 'Manuelle', 52000, 2019, 10200],
        ['Peugeot 208', 'Essence', 'Manuelle', 53000, 2019, 10300],
    ])
    df['Motorisation'] = ['  1.2   PureTech 68 ', '', None, '50 kWh - 136']
    df['Finition'] = ['Like', 'GT  Line', None, ' ']
    df['Puissance'] = [68, '110 ch', 5, 99999]  # 5 and 99999 are not possible powers
    cleaned = clean_cars(df, current_year=2026)
    assert len(cleaned) == 4  # an impossible power does not remove the car
    assert list(cleaned['Motorisation']) == ['1.2 PureTech 68', None, None, '50 kWh - 136']
    assert list(cleaned['Finition']) == ['Like', 'GT Line', None, None]
    assert cleaned['Puissance'].tolist()[:2] == [68, 110]
    assert cleaned['Puissance'].isna().tolist() == [False, False, True, True]


def test_clean_cars_without_engine_columns_has_no_engine_columns():
    cleaned = clean_cars(make_df([['Peugeot 208', 'Essence', 'Manuelle', 50000, 2019, 10000]]), current_year=2026)
    assert not {'Motorisation', 'Finition', 'Puissance'} & set(cleaned.columns)


def test_normalize_text():
    assert normalize_text('  GT   Line ') == 'GT Line'
    assert normalize_text('   ') is None
    assert normalize_text(None) is None
    assert normalize_text(float('nan')) is None


def test_clean_cars_removes_incomplete_and_impossible_rows():
    df = make_df([
        ['Peugeot 208', 'Essence', 'Manuelle', 50000, 2019, 10000],  # valid
        [None, 'Essence', 'Manuelle', 50000, 2019, 10000],  # no name
        ['Peugeot 208', 'Essence', 'Manuelle', None, 2019, 10100],  # no mileage
        ['Peugeot 208', 'Essence', 'Manuelle', 50001, 2019, None],  # no price
        ['Peugeot 208', 'Essence', 'Manuelle', 50002, 2019, 0],  # price 0
        ['Peugeot 208', 'Essence', 'Manuelle', 50003, 1850, 10200],  # year too old
        ['Peugeot 208', 'Essence', 'Manuelle', 50004, 2040, 10300],  # year in the future
        ['Peugeot 208', 'Essence', 'Manuelle', 9999999, 2019, 10400],  # impossible mileage
    ])
    cleaned = clean_cars(df, current_year=2026)
    assert len(cleaned) == 1
    assert cleaned.loc[0, 'Prix'] == 10000


def test_clean_cars_keeps_rows_with_unknown_fuel_or_gearbox():
    df = make_df([['Peugeot 208', None, None, 50000, 2019, 10000]])
    cleaned = clean_cars(df, current_year=2026)
    assert cleaned.loc[0, 'Carburant'] == UNKNOWN
    assert cleaned.loc[0, 'Boite de vitesse'] == UNKNOWN


def test_clean_cars_does_not_modify_the_input():
    df = make_df([['PEUGEOT 308', 'Essence', 'Manuelle', '79 049 km', '2019', '16 999 €']])
    before = df.copy()
    clean_cars(df, current_year=2026)
    pd.testing.assert_frame_equal(df, before)


def test_clean_cars_requires_the_columns():
    with pytest.raises(ValueError, match='Missing columns'):
        clean_cars(pd.DataFrame({'Nom': ['Peugeot 208']}))


def test_main_merges_files_and_creates_the_output_folder(tmp_path):
    first = tmp_path / 'first.csv'
    second = tmp_path / 'second.csv'
    make_df([['Peugeot 208', 'Essence', 'Manuelle', 50000, 2019, 10000]]).to_csv(first, index=False)
    make_df([
        ['PEUGEOT 208', 'Essence', 'Manuelle', '50 000 km', '2019', '10 000 €'],  # same car as the first file
        ['Renault Clio', 'Diesel', 'Manuelle', 30000, 2021, 12000],
    ]).to_csv(second, index=False)

    destination = tmp_path / 'data' / 'clean.csv'  # 'data' does not exist yet
    cleaned = main([str(first), str(second), '-o', str(destination)])

    assert len(cleaned) == 2
    saved = pd.read_csv(destination)
    assert sorted(saved['Nom']) == ['Peugeot 208', 'Renault Clio']
    assert 'Marque' in saved.columns
