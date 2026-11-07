"""What the web application (app.py) needs, without Streamlit: easy to test and to use elsewhere.

- the labels to choose from (brands and models, fuels, gearboxes)
- the cars of the data that look like the one we estimate
- the price of the model, with the labels cleaned like in scripts/datacleaning.py
"""

import json
import os

import pandas as pd

from scripts.datacleaning import normalize_fuel, normalize_gearbox, normalize_name
from scripts.models import predict_price

FUELS = ['Essence', 'Diesel', 'Électrique', 'Hybride', 'Hybride rechargeable', 'GPL']
GEARBOXES = ['Manuelle', 'Automatique']
SIMILAR_COLUMNS = ['Nom', 'année', 'Km', 'Carburant', 'Boite de vitesse', 'Prix']


def load_cars(path):
    """
    The cleaned cars of a CSV file (see scripts/datacleaning.py), or None if there is no usable file.

    The application works without it, the file only gives the list of the models and cars to compare with.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        cars = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    needed = {'Nom', 'Carburant', 'Boite de vitesse', 'Km', 'année', 'Prix'}
    return cars if needed <= set(cars.columns) and not cars.empty else None


def model_names(cars):
    """The names (brand and model) to choose from, sorted; empty if there are no cars."""
    return [] if cars is None else sorted(cars['Nom'].dropna().unique())


def fuel_options(cars):
    """The fuels to choose from: the ones of the cars if there are some, else the usual ones."""
    if cars is None:
        return list(FUELS)
    found = [fuel for fuel in FUELS if fuel in set(cars['Carburant'])]
    return found or list(FUELS)


def similar_cars(cars, name, year, km, count=5):
    """
    The cars of the data with the same name, the closest to this year and mileage first.

    Parameters:
    - cars: the cleaned cars (or None)
    - name: brand and model, for example 'Peugeot 208'
    - year, km: the year and the mileage of the car
    - count: how many cars to return

    Returns:
    - A DataFrame with the columns SIMILAR_COLUMNS (empty if there is no such car)
    """
    if cars is None:
        return pd.DataFrame(columns=SIMILAR_COLUMNS)
    same = cars[cars['Nom'].str.lower() == normalize_name(name).lower()] if normalize_name(name) else cars.iloc[0:0]
    if same.empty:
        return pd.DataFrame(columns=SIMILAR_COLUMNS)
    distance = (same['année'] - year).abs() * 20_000 + (same['Km'] - km).abs()  # a year counts as 20 000 km
    return same.loc[distance.sort_values().index[:count], SIMILAR_COLUMNS].reset_index(drop=True)


def market_summary(cars, name):
    """The number of cars of this name in the data and their lowest, median and highest price, or None."""
    if cars is None or not normalize_name(name):
        return None
    prices = cars.loc[cars['Nom'].str.lower() == normalize_name(name).lower(), 'Prix']
    if prices.empty:
        return None
    return {
        'cars': int(len(prices)),
        'min': int(prices.min()),
        'median': float(prices.median()),
        'max': int(prices.max()),
    }


def typical_error(model_path):
    """The mean error (in euros) measured when the model was trained, if it was saved next to it, else None."""
    try:
        with open(os.path.splitext(model_path)[0] + '.json', encoding='utf-8') as file:
            error = json.load(file)['metrics']['mae']
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return float(error) if isinstance(error, (int, float)) else None


def estimate(model, name, fuel, gearbox, km, year, power=None):
    """
    The price (in euros) that the model estimates. The labels are cleaned like in the data ('Auto.' = Automatique).

    A power of 0 or None means unknown.
    """
    return predict_price(
        model,
        normalize_name(name),
        normalize_fuel(fuel),
        normalize_gearbox(gearbox),
        km,
        year,
        power=power or None,
    )
