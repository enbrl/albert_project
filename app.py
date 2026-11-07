"""Web application: estimate the price of a used car.

Start it from the root of the project, after training the model with
``python -m scripts.models data/cars_clean.csv``:

    pip install -r requirements-app.txt
    streamlit run app.py

It opens in your browser, in English or in French (the choice is in the sidebar). The logic is in
scripts/estimator.py and the texts in scripts/app_texts.py.
"""

import datetime
import os

import streamlit as st

from scripts.analysis import euros
from scripts.app_texts import DEFAULT_LANGUAGE, LANGUAGES, label, text, translate_table
from scripts.datacleaning import FIRST_YEAR
from scripts.estimator import (
    GEARBOXES,
    estimate,
    fuel_options,
    load_cars,
    market_summary,
    model_names,
    similar_cars,
    typical_error,
)
from scripts.models import DEFAULT_MODEL_PATH, load_model

# Every widget has a `key`: Streamlit otherwise identifies a widget by its label, which changes with the language, and the
# user would lose what was typed (the paths of the files, the car) when switching language.
# The first Streamlit command: it runs before the language is chosen, so the title of the tab is the same in both languages
st.set_page_config(page_title=f'{text("en", "page_title")} / {text("fr", "page_title")}', page_icon='🚗')

with st.sidebar:
    language = st.selectbox(
        'Language / Langue',
        list(LANGUAGES),
        index=list(LANGUAGES).index(DEFAULT_LANGUAGE),
        format_func=LANGUAGES.get,
        key='language',
    )


def t(key, **values):
    """A text of the application in the chosen language."""
    return text(language, key, **values)


st.title(t('title'))
st.caption(t('caption'))

with st.sidebar:
    st.header(t('files'))
    model_path = st.text_input(t('model_file'), DEFAULT_MODEL_PATH, key='model_path')
    data_path = st.text_input(t('data_file'), 'data/cars_clean.csv', help=t('data_help'), key='data_path')
    st.caption(t('make_files'))


@st.cache_resource(show_spinner=False)
def get_model(path, modified):
    """The model of a file (read again if the file changed: `modified` is its date)."""
    return load_model(path)


@st.cache_data(show_spinner=False)
def get_cars(path, modified):
    return load_cars(path)


def modified(path):
    return os.path.getmtime(path) if os.path.isfile(path) else None


if not os.path.isfile(model_path):
    st.error(t('no_model', path=model_path))
    st.stop()
try:
    model = get_model(model_path, modified(model_path))
except Exception as error:  # whatever the reason, say that the file cannot be read
    st.error(t('unreadable', path=model_path, kind=type(error).__name__))
    st.stop()

cars = get_cars(data_path, modified(data_path))
names = model_names(cars)
last_year = datetime.date.today().year + 1

with st.form('car'):
    if names:
        name = st.selectbox(
            t('name'), names, index=names.index('Peugeot 208') if 'Peugeot 208' in names else 0, key='name_choice'
        )
    else:
        name = st.text_input(t('name'), 'Peugeot 208', help=t('name_help'), key='name_typed')
    left, right = st.columns(2)
    fuel = left.selectbox(t('fuel'), fuel_options(cars), format_func=lambda value: label(language, value), key='fuel')
    gearbox = right.selectbox(t('gearbox'), GEARBOXES, format_func=lambda value: label(language, value), key='gearbox')
    km = left.number_input(t('km'), min_value=0, max_value=1_000_000, value=60_000, step=1_000, key='km')
    year = right.number_input(t('year'), min_value=FIRST_YEAR, max_value=last_year, value=2019, step=1, key='year')
    power = st.number_input(t('power'), min_value=0, max_value=1500, value=0, step=5, help=t('power_help'), key='power')
    battery = st.number_input(
        t('battery'), min_value=0, max_value=300, value=0, step=5, help=t('battery_help'), key='battery'
    )
    submitted = st.form_submit_button(t('submit'), type='primary')

if submitted:
    if not name.strip():
        st.warning(t('empty_name'))
        st.stop()
    price = estimate(model, name, fuel, gearbox, int(km), int(year), int(power), int(battery))
    st.metric(t('estimated_price'), euros(round(price, -2)))
    mean_error = typical_error(model_path)
    if mean_error:
        st.caption(t('mean_error', error=euros(round(mean_error, -1))))

    summary = market_summary(cars, name)
    if summary:
        st.subheader(t('similar_title', name=name))
        st.write(
            t(
                'similar_text',
                cars=summary['cars'],
                low=euros(summary['min']),
                high=euros(summary['max']),
                median=euros(summary['median']),
            )
        )
        st.dataframe(translate_table(language, similar_cars(cars, name, int(year), int(km))), hide_index=True)
