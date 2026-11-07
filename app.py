"""Web application: estimate the price of a used car.

Start it from the root of the project, after training the model with
``python -m scripts.models data/cars_clean.csv``:

    pip install -r requirements-app.txt
    streamlit run app.py

It opens in your browser. The logic is in scripts/estimator.py.
"""

import datetime
import os

import streamlit as st

from scripts.analysis import euros
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

st.set_page_config(page_title='Used car price', page_icon='🚗')
st.title('🚗 Used car price')
st.caption('Estimate the price of a used car from a model trained on the aramisauto.com listing.')

with st.sidebar:
    st.header('Files')
    model_path = st.text_input('Trained model', DEFAULT_MODEL_PATH)
    data_path = st.text_input(
        'Cleaned cars (optional)',
        'data/cars_clean.csv',
        help='Gives the list of the models and the cars to compare with.',
    )
    st.caption('Make them with `python -m scripts.pipeline`.')


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
    st.error(f'There is no model in `{model_path}`. Train it first: `python -m scripts.models data/cars_clean.csv`')
    st.stop()
try:
    model = get_model(model_path, modified(model_path))
except Exception as error:  # whatever the reason, say that the file cannot be read
    st.error(
        f'The model in `{model_path}` cannot be read ({type(error).__name__}). It was probably saved with another '
        'version of scikit-learn: train it again with `python -m scripts.models data/cars_clean.csv`.'
    )
    st.stop()

cars = get_cars(data_path, modified(data_path))
names = model_names(cars)
last_year = datetime.date.today().year + 1

with st.form('car'):
    if names:
        name = st.selectbox('Brand and model', names, index=names.index('Peugeot 208') if 'Peugeot 208' in names else 0)
    else:
        name = st.text_input('Brand and model', 'Peugeot 208', help='For example Renault Clio')
    left, right = st.columns(2)
    fuel = left.selectbox('Fuel', fuel_options(cars))
    gearbox = right.selectbox('Gearbox', GEARBOXES)
    km = left.number_input('Mileage (km)', min_value=0, max_value=1_000_000, value=60_000, step=1_000)
    year = right.number_input(
        'Year of first registration', min_value=FIRST_YEAR, max_value=last_year, value=2019, step=1
    )
    power = st.number_input(
        'Power (horsepower), 0 if you do not know',
        min_value=0,
        max_value=1500,
        value=0,
        step=5,
        help='It makes the estimate more precise when the model knows the power.',
    )
    submitted = st.form_submit_button('Estimate the price', type='primary')

if submitted:
    if not name.strip():
        st.warning('Write the brand and the model of the car.')
        st.stop()
    price = estimate(model, name, fuel, gearbox, int(km), int(year), int(power))
    st.metric('Estimated price', euros(round(price, -2)))
    mean_error = typical_error(model_path)
    if mean_error:
        st.caption(f'On cars it had not seen, this model is wrong by {euros(round(mean_error, -1))} on average.')

    summary = market_summary(cars, name)
    if summary:
        st.subheader(f'Cars like this in the data: {name}')
        st.write(
            f'{summary["cars"]} cars, from {euros(summary["min"])} to {euros(summary["max"])} '
            f'(median {euros(summary["median"])}). The closest to yours:'
        )
        st.dataframe(similar_cars(cars, name, int(year), int(km)), hide_index=True)
