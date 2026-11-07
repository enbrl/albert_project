"""Estimate the price of one used car from the command line.

Usage (from the root of the project), after training the model with
``python -m scripts.models data/cars_clean.csv``:

    python -m scripts.predict --name "Peugeot 208" --fuel Essence --gearbox Manuelle --km 60000 --year 2019

The labels are cleaned like in scripts/datacleaning.py: 'PEUGEOT 208', 'peugeot 208'
and 'Peugeot 208' are the same car, and 'Auto.' means 'Automatique'.
"""

import argparse

from scripts.analysis import euros
from scripts.datacleaning import normalize_fuel, normalize_gearbox, normalize_name
from scripts.models import DEFAULT_MODEL_PATH, load_model, predict_price


def main(argv=None):
    parser = argparse.ArgumentParser(description='Estimate the price of a used car')
    parser.add_argument('--name', required=True, help='brand and model, for example "Peugeot 208"')
    parser.add_argument('--fuel', required=True, help='Essence, Diesel, Électrique, Hybride...')
    parser.add_argument('--gearbox', required=True, help='Manuelle or Automatique')
    parser.add_argument('--km', required=True, type=int, help='mileage in km')
    parser.add_argument('--year', required=True, type=int, help='year of the first registration')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help=f'trained model (default: {DEFAULT_MODEL_PATH})')
    args = parser.parse_args(argv)

    if args.km < 0:
        parser.error('--km must be positive')

    try:
        model = load_model(args.model)
    except FileNotFoundError:
        parser.exit(1, f'No model in {args.model}. Train it first: python -m scripts.models data/cars_clean.csv\n')

    price = predict_price(
        model,
        normalize_name(args.name),
        normalize_fuel(args.fuel),
        normalize_gearbox(args.gearbox),
        args.km,
        args.year,
    )
    print(f'Estimated price: {euros(price)}')
    return price


if __name__ == '__main__':
    main()
