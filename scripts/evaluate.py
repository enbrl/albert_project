"""Where is the price model wrong? The error for each kind of car, on cars the model has not seen.

``python -m scripts.models`` gives one number for the whole dataset. A single number hides that a model can be
good on a Peugeot 208 and bad on a rare model or on the most expensive cars. This script cuts the cars in groups
(fuel, age, price, mileage, brand, rare or common model) and gives the error of each group.

Usage (from the root of the project):

    python -m scripts.evaluate data/cars_clean.csv
    python -m scripts.evaluate data/cars_clean.csv --kind ridge --output reports/evaluation.json

How: the cars are cut in 5 parts (cross-validation); each car is predicted by a model that did not learn from it,
so every car has an honest prediction. The groups are then compared. For each group:

- ``cars``: how many cars
- ``mae``: the mean error, in euros (the lower the better)
- ``mape``: the mean error in % of the price
- ``bias``: the mean of (estimate - real price), in euros. Negative: the model estimates too low; positive: too high.

A group of few cars gives a noisy number: groups with fewer than ``--min-cars`` cars are left out.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from scripts.cli import UserError, positive_int, run
from scripts.models import DEFAULT_KIND, MIN_ROWS, MODEL_KINDS, build_model, feature_columns, prepare_features

# A group (or a model) with fewer cars than this is "rare": the model has seen too few of them to learn them
MIN_CARS = 5

MILEAGE_LIMITS = [
    0,
    5_000,
    30_000,
    80_000,
    float('inf'),
]  # each group includes its first limit: 5 000 km is in '5 000 to 30 000'
MILEAGE_LABELS = ['under 5 000 km', '5 000 to 30 000 km', '30 000 to 80 000 km', '80 000 km or more']
AGE_LIMITS = [-1, 0, 1, 3, 6, float('inf')]

# These segments have a natural order (newest to oldest, cheapest to most expensive...): it is kept.
# The others (fuel, brand) are sorted by error, the least well predicted first.
ORDERED_SEGMENTS = ('age', 'price', 'mileage')


def predict_out_of_fold(df, kind=DEFAULT_KIND, n_splits=5, random_state=42):
    """
    A price estimate for every car, made by a model that did not learn from it (cross-validation).

    Returns:
    - A Series of estimates, with the index of df
    """
    if len(df) < MIN_ROWS:
        raise UserError(f'At least {MIN_ROWS} cars are needed to evaluate a model, got {len(df)}')
    columns = feature_columns(df)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    estimates = cross_val_predict(
        build_model(kind, random_state, columns), prepare_features(df, columns), df['Prix'], cv=splitter
    )
    return pd.Series(estimates, index=df.index, name='estimate')


def age_labels(last_year):
    """The labels of the age groups, from the newest year of the data (so they do not change with the date of the day)."""
    return [
        f'{last_year} (newest)',
        f'{last_year - 1}',
        f'{last_year - 3} to {last_year - 2}',
        f'{last_year - 6} to {last_year - 4}',
        f'before {last_year - 6}',
    ]


def add_segments(df):
    """
    The same cars with the columns that say which group each one belongs to.

    Returns:
    - A new DataFrame with the columns 'fuel', 'age', 'price', 'mileage', 'brand' and 'model_rarity'
    """
    cars = df.copy()
    last_year = int(cars['année'].max())
    cars['fuel'] = cars['Carburant']
    cars['age'] = pd.cut(last_year - cars['année'], AGE_LIMITS, labels=age_labels(last_year))
    # rank first: cars with the same price (ties) must not make two quarters of the same limit
    cars['price'] = pd.qcut(
        cars['Prix'].rank(method='first'),
        4,
        labels=['cheapest quarter', 'second quarter', 'third quarter', 'most expensive quarter'],
    )
    cars['mileage'] = pd.cut(cars['Km'], MILEAGE_LIMITS, labels=MILEAGE_LABELS, right=False)
    cars['brand'] = cars['Marque'] if 'Marque' in cars.columns else cars['Nom'].str.split().str[0]
    return cars


def errors_by_group(cars, estimates, column, min_cars=MIN_CARS):
    """
    The error of each group of one segment (a column of add_segments): in their natural order for age, price and mileage,
    else the least well predicted first.

    Parameters:
    - cars: the DataFrame of add_segments
    - estimates: the out-of-fold estimates (predict_out_of_fold)
    - column: the segment, for example 'fuel'
    - min_cars: the groups with fewer cars are left out

    Returns:
    - A DataFrame with one row per group: group, cars, mae, mape, bias
    """
    error = estimates - cars['Prix']
    table = (
        pd.DataFrame(
            {'group': cars[column], 'abs': error.abs(), 'pct': error.abs() / cars['Prix'] * 100, 'bias': error}
        )
        .groupby('group', observed=True)
        .agg(cars=('abs', 'size'), mae=('abs', 'mean'), mape=('pct', 'mean'), bias=('bias', 'mean'))
        .reset_index()
    )
    table = table[table['cars'] >= min_cars]
    if column in ORDERED_SEGMENTS:
        return table.reset_index(drop=True)
    return table.sort_values('mape', ascending=False).reset_index(drop=True)


def errors_by_model_rarity(cars, estimates, min_cars=MIN_CARS):
    """
    The error of the models the data knows well against the models it knows little.

    A model (brand and model, for example 'Peugeot 208') with fewer than min_cars cars in the data is rare: the
    estimate for it can only lean on the brand, the year and the mileage.

    Returns:
    - A DataFrame with the two groups: group, cars, mae, mape, bias
    """
    sizes = cars.groupby('Nom')['Prix'].transform('size')
    labelled = cars.assign(
        model_rarity=np.where(
            sizes < min_cars, f'rare model (under {min_cars} cars)', f'common model ({min_cars}+ cars)'
        )
    )
    return errors_by_group(labelled, estimates, 'model_rarity', min_cars=1)


def evaluate(df, kind=DEFAULT_KIND, n_splits=5, random_state=42, min_cars=MIN_CARS, estimates=None):
    """
    The error of the model overall and for every group of cars.

    Parameters:
    - estimates: the out-of-fold estimates if they are already made (predict_out_of_fold), else they are made here

    Returns:
    - A dictionary: 'overall' (cars, mae, mape, bias), and one DataFrame per segment: 'fuel', 'age', 'price',
      'mileage', 'brand', 'model_rarity'
    """
    if estimates is None:
        estimates = predict_out_of_fold(df, kind, n_splits, random_state)
    cars = add_segments(df)
    error = estimates - df['Prix']
    result = {
        'overall': {
            'cars': int(len(df)),
            'mae': float(error.abs().mean()),
            'mape': float((error.abs() / df['Prix']).mean() * 100),
            'bias': float(error.mean()),
        }
    }
    for column in ('fuel', 'age', 'price', 'mileage', 'brand'):
        result[column] = errors_by_group(cars, estimates, column, min_cars)
    result['model_rarity'] = errors_by_model_rarity(cars, estimates, min_cars)
    return result


TITLES = {
    'fuel': 'By fuel',
    'age': 'By age (years since the newest car of the data)',
    'price': 'By price',
    'mileage': 'By mileage',
    'brand': 'By brand',
    'model_rarity': 'By how well the data knows the model',
}


def format_report(result, kind):
    """The text printed by the command: the overall error, then one table per segment."""
    overall = result['overall']
    lines = [
        f'Model: {kind}. Cars: {overall["cars"]}.',
        f'Overall: mean error {overall["mae"]:.0f} EUR ({overall["mape"]:.1f} % of the price), '
        f'bias {overall["bias"]:+.0f} EUR (negative: estimates too low).',
    ]
    for column, title in TITLES.items():
        table = result[column].round({'mae': 0, 'mape': 1, 'bias': 0})
        lines += ['', f'{title}:', table.to_string(index=False)]
    return '\n'.join(lines)


def to_json(result, kind):
    """The same results as plain Python data, to be saved as JSON."""
    data = {'model': kind, 'overall': {key: round(value, 3) for key, value in result['overall'].items()}}
    for column in TITLES:
        data[column] = result[column].round(3).to_dict('records')
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description='Where is the price model wrong? The error of each group of cars')
    parser.add_argument('data', help='cleaned CSV file (made by scripts/datacleaning.py)')
    parser.add_argument(
        '--kind', default=DEFAULT_KIND, choices=MODEL_KINDS, help=f'model to evaluate (default: {DEFAULT_KIND})'
    )
    parser.add_argument(
        '--min-cars',
        type=positive_int,
        default=MIN_CARS,
        help=f'groups with fewer cars are left out (default: {MIN_CARS})',
    )
    parser.add_argument('--output', help='also save the results in this JSON file')
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    result = evaluate(df, kind=args.kind, min_cars=args.min_cars)
    print(format_report(result, args.kind))
    if args.output:
        folder = os.path.dirname(args.output)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as file:
            json.dump(to_json(result, args.kind), file, indent=2, ensure_ascii=False)
            file.write('\n')
        print(f'\nSaved in {args.output}')
    return result


if __name__ == '__main__':
    run(main)
