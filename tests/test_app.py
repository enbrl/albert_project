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
    assert any('wrong by 1 560 € on average' in caption.value for caption in app.caption)  # saved with the model, rounded
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
    assert len(app.selectbox) == 2  # fuel and gearbox: the name is typed
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
