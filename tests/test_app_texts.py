"""Tests of scripts/app_texts.py: the texts of the web application in two languages (no Streamlit needed)."""

import string

import pandas as pd
import pytest

from scripts.app_texts import COLUMNS, DEFAULT_LANGUAGE, LABELS, LANGUAGES, TEXTS, label, text, translate_table
from scripts.estimator import FUELS, GEARBOXES, SIMILAR_COLUMNS


def placeholders(sentence):
    return {name for _, name, _, _ in string.Formatter().parse(sentence) if name}


def test_both_languages_have_the_same_texts():
    assert set(LANGUAGES) == set(TEXTS) == {'en', 'fr'}
    assert set(TEXTS['en']) == set(TEXTS['fr'])


@pytest.mark.parametrize('key', sorted(TEXTS['en']))
def test_a_text_has_the_same_placeholders_in_both_languages(key):
    assert placeholders(TEXTS['en'][key]) == placeholders(TEXTS['fr'][key]), key


@pytest.mark.parametrize('key', sorted(TEXTS['en']))
def test_no_text_is_empty_and_none_is_left_in_english_in_the_french_texts(key):
    assert TEXTS['en'][key].strip() and TEXTS['fr'][key].strip()
    assert TEXTS['en'][key] != TEXTS['fr'][key], f'{key} is the same in both languages'


def test_the_default_language_is_english():
    assert DEFAULT_LANGUAGE == 'en'


def test_text_fills_the_placeholders():
    assert text('en', 'no_model', path='models/x.joblib').startswith('There is no model in `models/x.joblib`.')
    assert text('fr', 'no_model', path='models/x.joblib').startswith("Il n'y a pas de modèle dans `models/x.joblib`.")
    assert text('fr', 'similar_text', cars=12, low='5 000 €', high='9 000 €', median='7 000 €') == (
        '12 voitures, de 5 000 € à 9 000 € (médiane 7 000 €). Les plus proches de la vôtre :'
    )


def test_text_without_values_is_given_as_it_is():
    assert text('en', 'submit') == 'Estimate the price'
    assert text('fr', 'submit') == 'Estimer le prix'


def test_an_unknown_language_falls_back_to_english():
    assert text('de', 'submit') == 'Estimate the price'


def test_a_missing_value_for_a_placeholder_is_an_error_not_a_silent_blank():
    with pytest.raises(KeyError):
        text('en', 'no_model', other='x')


# ---------------------------------------------------------------- the words of the data


@pytest.mark.parametrize('value', FUELS + GEARBOXES)
def test_every_fuel_and_gearbox_has_an_english_label(value):
    assert value in LABELS['en'], f'{value} has no English label'
    assert label('en', value) != '' and label('fr', value) == value  # French: as in the data


def test_an_unknown_value_is_shown_as_it_is():
    assert label('en', 'Éthanol e85') == 'Éthanol e85'
    assert label('de', 'Essence') == 'Essence'


@pytest.mark.parametrize('language', ['en', 'fr'])
def test_every_column_of_the_table_has_a_header(language):
    assert set(SIMILAR_COLUMNS) <= set(COLUMNS[language])


def sample_table():
    return pd.DataFrame(
        {
            'Nom': ['Peugeot 208'],
            'année': [2019],
            'Km': [60000],
            'Carburant': ['Essence'],
            'Boite de vitesse': ['Manuelle'],
            'Prix': [9500],
        }
    )


def test_the_table_is_translated_into_english():
    shown = translate_table('en', sample_table())
    assert list(shown.columns) == ['Model', 'Year', 'Mileage (km)', 'Fuel', 'Gearbox', 'Price (€)']
    assert shown.loc[0, 'Fuel'] == 'Petrol' and shown.loc[0, 'Gearbox'] == 'Manual'
    assert shown.loc[0, 'Model'] == 'Peugeot 208' and shown.loc[0, 'Price (€)'] == 9500  # the numbers are not touched


def test_the_table_is_translated_into_french():
    shown = translate_table('fr', sample_table())
    assert list(shown.columns) == ['Modèle', 'Année', 'Kilométrage (km)', 'Carburant', 'Boîte de vitesse', 'Prix (€)']
    assert shown.loc[0, 'Carburant'] == 'Essence'


def test_translating_does_not_change_the_data():
    table = sample_table()
    translate_table('en', table)
    assert list(table.columns)[0] == 'Nom' and table.loc[0, 'Carburant'] == 'Essence'


def test_an_empty_table_can_be_translated():
    empty = pd.DataFrame(columns=SIMILAR_COLUMNS)
    assert list(translate_table('en', empty).columns)[0] == 'Model' and translate_table('fr', empty).empty
