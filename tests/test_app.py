"""Tests of the web application app.py, with the testing tool of Streamlit.

They are skipped when Streamlit is not installed (it is only needed for the application:
pip install -r requirements-app.txt).
"""

import json
from pathlib import Path

import pytest

pytest.importorskip('streamlit', reason='Streamlit is not installed (pip install -r requirements-app.txt)')
from streamlit.testing.v1 import AppTest  # noqa: E402

from scripts.models import save_model, train_final_model  # noqa: E402
from tests.test_models import make_cars, make_cars_with_power  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / 'app.py')


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp('app')
    cars = make_cars(300, seed=4)
    cars.to_csv(folder / 'cars.csv', index=False)
    save_model(train_final_model(cars, kind='ridge'), str(folder / 'price.joblib'))
    (folder / 'price.json').write_text(json.dumps({'metrics': {'mae': 1556.0}}))
    power_cars = make_cars_with_power(300, seed=4)
    power_cars.to_csv(folder / 'power_cars.csv', index=False)
    save_model(train_final_model(power_cars, kind='ridge'), str(folder / 'power.joblib'))
    return folder


def start(model, cars=''):
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    app.sidebar.text_input[0].set_value(str(model))
    app.sidebar.text_input[1].set_value(str(cars))
    return app.run()


def test_without_a_model_the_app_explains_what_to_do(tmp_path):
    app = start(tmp_path / 'nothing.joblib')
    assert len(app.error) == 1
    assert 'There is no model' in app.error[0].value and 'python -m scripts.models' in app.error[0].value
    assert len(app.button) == 0  # the form is not shown


def test_a_model_that_cannot_be_read(tmp_path):
    broken = tmp_path / 'broken.joblib'
    broken.write_bytes(b'broken')
    app = start(broken)
    assert 'cannot be read' in app.error[0].value and 'train it again' in app.error[0].value


def test_the_title_and_the_form_are_shown(files):
    app = start(files / 'price.joblib', files / 'cars.csv')
    assert not app.exception
    assert app.title[0].value.endswith('Used car price')
    assert len(app.button) == 1 and app.button[0].label == 'Estimate the price'
    assert app.selectbox[0].label == 'Brand and model' and 'Peugeot 208' in app.selectbox[0].options
    assert app.selectbox[0].value == 'Peugeot 208'


def test_estimate_with_the_cars_file(files):
    app = start(files / 'price.joblib', files / 'cars.csv')
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].label == 'Estimated price'
    assert app.metric[0].value.endswith(' €') and app.metric[0].value[0].isdigit()
    assert any(
        'wrong by 1 560 € on average' in caption.value for caption in app.caption
    )  # saved with the model, rounded
    assert 'Cars like this in the data: Peugeot 208' in app.subheader[0].value
    assert len(app.dataframe) == 1 and len(app.dataframe[0].value) == 5  # the 5 closest cars


def test_a_newer_car_with_fewer_kilometres_costs_more(files):
    def price(year, km):
        app = start(files / 'price.joblib', files / 'cars.csv')
        app.number_input[0].set_value(km)
        app.number_input[1].set_value(year)
        app.button[0].click().run()
        return int(app.metric[0].value.replace(' €', '').replace(' ', '').replace(' ', ''))

    assert price(2024, 20_000) > price(2013, 180_000) + 3000


def test_without_the_cars_file_the_name_is_typed_and_nothing_is_compared(files):
    app = start(files / 'price.joblib')
    assert len(app.main.selectbox) == 2  # fuel and gearbox: the name is typed (the language is in the sidebar)
    assert app.text_input[0].label == 'Brand and model'  # the main page comes before the sidebar in the list
    app.text_input[0].set_value('Renault Clio')
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value.endswith(' €')
    assert len(app.dataframe) == 0 and len(app.subheader) == 0


def test_a_name_that_the_model_does_not_know_still_gives_an_estimate(files):
    app = start(files / 'price.joblib')
    app.text_input[0].set_value('Ferrari F40')
    app.button[0].click().run()
    assert not app.exception and app.metric[0].value.endswith(' €')  # the model ignores unknown labels
    assert len(app.dataframe) == 0


def test_an_empty_name_is_refused(files):
    app = start(files / 'price.joblib')
    app.text_input[0].set_value('   ')
    app.button[0].click().run()
    assert len(app.warning) == 1 and 'brand and the model' in app.warning[0].value
    assert len(app.metric) == 0


def test_the_power_changes_the_estimate_when_the_model_knows_it(files):
    def price(power):
        app = start(files / 'power.joblib', files / 'power_cars.csv')
        app.selectbox[0].set_value('Opel Astra')
        app.number_input[2].set_value(power)
        app.button[0].click().run()
        return int(app.metric[0].value.replace(' €', '').replace(' ', '').replace(' ', ''))

    assert price(190) > price(65) + 1000


# ---------------------------------------------------------------- the two languages


def choose_french(app):
    app.sidebar.selectbox[0].set_value('fr').run()  # the first selectbox of the sidebar is the language
    return app


def test_the_application_starts_in_english_with_a_choice_of_language(files):
    app = start(files / 'price.joblib', files / 'cars.csv')
    language = app.sidebar.selectbox[0]
    assert (
        language.label == 'Language / Langue'
        and language.value == 'en'
        and list(language.options) == ['English', 'Français']
    )
    assert app.title[0].value.endswith('Used car price')


def test_in_french_every_text_of_the_page_is_in_french(files):
    app = choose_french(start(files / 'price.joblib', files / 'cars.csv'))
    assert not app.exception
    assert app.title[0].value == "🚗 Prix d'une voiture d'occasion"
    assert app.button[0].label == 'Estimer le prix'
    assert [s.label for s in app.main.selectbox] == ['Marque et modèle', 'Carburant', 'Boîte de vitesse']
    assert [n.label for n in app.number_input][:2] == ['Kilométrage (km)', 'Année de première mise en circulation']
    assert app.sidebar.text_input[0].label == 'Modèle entraîné'


def test_in_english_the_fuels_and_gearboxes_are_in_english_but_the_values_stay_the_data_labels(files):
    app = start(files / 'price.joblib', files / 'cars.csv')
    fuel, gearbox = app.main.selectbox[1], app.main.selectbox[2]
    assert list(fuel.options) == ['Petrol', 'Diesel']  # the fuels of the made-up data
    assert list(gearbox.options) == ['Manual', 'Automatic']
    assert fuel.value == 'Essence' and gearbox.value == 'Manuelle'  # what the model knows


def test_in_french_the_fuels_are_shown_as_in_the_data(files):
    app = choose_french(start(files / 'price.joblib', files / 'cars.csv'))
    assert list(app.main.selectbox[1].options) == ['Essence', 'Diesel']
    assert list(app.main.selectbox[2].options) == ['Manuelle', 'Automatique']


def test_the_result_is_in_french_and_the_estimate_is_the_same_in_both_languages(files):
    english = start(files / 'price.joblib', files / 'cars.csv')
    english.button[0].click().run()
    french = choose_french(start(files / 'price.joblib', files / 'cars.csv'))
    french.button[0].click().run()
    assert french.metric[0].label == 'Prix estimé'
    assert french.metric[0].value == english.metric[0].value  # the language changes the words, never the estimate
    assert any('ce modèle se trompe de 1 560 € en moyenne' in caption.value for caption in french.caption)
    assert french.subheader[0].value == 'Voitures comparables dans les données : Peugeot 208'
    table = french.dataframe[0].value
    assert list(table.columns) == ['Modèle', 'Année', 'Kilométrage (km)', 'Carburant', 'Boîte de vitesse', 'Prix (€)']


def test_the_table_of_an_english_page_has_english_headers_and_values(files):
    app = start(files / 'price.joblib', files / 'cars.csv')
    app.button[0].click().run()
    table = app.dataframe[0].value
    assert list(table.columns) == ['Model', 'Year', 'Mileage (km)', 'Fuel', 'Gearbox', 'Price (€)']
    assert set(table['Fuel']) <= {'Petrol', 'Diesel'} and set(table['Gearbox']) <= {'Manual', 'Automatic'}


def test_the_messages_when_there_is_no_model_are_in_french_too(tmp_path):
    app = start(tmp_path / 'nothing.joblib')
    choose_french(app)
    assert "Il n'y a pas de modèle" in app.error[0].value and 'python -m scripts.models' in app.error[0].value


def test_the_message_for_a_model_that_cannot_be_read_is_in_french_too(tmp_path):
    broken = tmp_path / 'broken.joblib'
    broken.write_bytes(b'broken')
    app = choose_french(start(broken))
    assert 'ne peut pas être lu' in app.error[0].value and 'entraînez-le de nouveau' in app.error[0].value


def test_an_empty_name_is_refused_in_french(files):
    app = choose_french(start(files / 'price.joblib'))
    app.text_input[0].set_value('   ')
    app.button[0].click().run()
    assert app.warning[0].value == 'Écrivez la marque et le modèle de la voiture.'


def test_switching_language_keeps_what_the_user_typed_and_chose(files):
    # Streamlit identifies a widget by its label unless it has a key: without keys, a new language would forget everything.
    # (The fields of a form only count once the button is pressed: that is how Streamlit forms work, in any language.)
    app = start(files / 'price.joblib', files / 'cars.csv')
    app.main.selectbox[0].set_value('Opel Astra')
    app.main.selectbox[1].set_value('Diesel')
    app.number_input[0].set_value(123_000)
    app.number_input[1].set_value(2015)
    app.number_input[2].set_value(120)
    app.button[0].click().run()
    price = app.metric[0].value

    choose_french(app)

    assert app.sidebar.text_input[0].value == str(
        files / 'price.joblib'
    )  # the model and data files typed in the sidebar
    assert app.sidebar.text_input[1].value == str(files / 'cars.csv')
    assert app.main.selectbox[0].value == 'Opel Astra' and app.main.selectbox[1].value == 'Diesel'
    assert [n.value for n in app.number_input][:3] == [123_000, 2015, 120]
    assert len(app.error) == 0 and len(app.button) == 1  # the form is still there, the model was not lost
    app.button[0].click().run()
    assert app.metric[0].value == price  # the same car, the same estimate
