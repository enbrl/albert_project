"""Estimate the price of one used car from the command line.

Usage (from the root of the project), after training the model with
``python -m scripts.models data/cars_clean.csv``:

    python -m scripts.predict --name "Peugeot 208" --fuel Essence --gearbox Manuelle --km 60000 --year 2019

The power in horsepower is optional (``--power 100``): it makes the estimate more precise when the
model was trained with it (the power is the column that matters most for the price). For an electric car,
``--battery 50`` gives the capacity of the battery in kWh, with the same effect, and ``--trim "Allure Pack"`` the trim.

The labels are cleaned like in scripts/datacleaning.py: 'PEUGEOT 208', 'peugeot 208'
and 'Peugeot 208' are the same car, and 'Auto.' means 'Automatique'.
"""

import argparse
import datetime
import sys

from scripts.analysis import euros
from scripts.cli import run
from scripts.datacleaning import FIRST_YEAR, normalize_fuel, normalize_gearbox, normalize_name
from scripts.models import DEFAULT_MODEL_PATH, compatibility_problem, load_model, predict_price, read_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description='Estimate the price of a used car')
    parser.add_argument('--name', required=True, help='brand and model, for example "Peugeot 208"')
    parser.add_argument('--fuel', required=True, help='Essence, Diesel, Électrique, Hybride...')
    parser.add_argument('--gearbox', required=True, help='Manuelle or Automatique')
    parser.add_argument('--km', required=True, type=int, help='mileage in km')
    parser.add_argument('--year', required=True, type=int, help='year of the first registration')
    parser.add_argument('--power', type=int, help='power in horsepower (optional, used if the model knows the power)')
    parser.add_argument(
        '--battery',
        type=float,
        help='capacity of the battery in kWh (optional, electric cars, used if the model knows it)',
    )
    parser.add_argument('--trim', help='trim, for example "Allure Pack" (optional, used if the model knows the trims)')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help=f'trained model (default: {DEFAULT_MODEL_PATH})')
    args = parser.parse_args(argv)

    if args.km < 0:
        parser.error('--km must be positive')
    if args.power is not None and args.power <= 0:
        parser.error('--power must be positive')
    if args.battery is not None and args.battery <= 0:
        parser.error('--battery must be positive')
    last_year = datetime.date.today().year + 1
    if not FIRST_YEAR <= args.year <= last_year:
        parser.error(f'--year must be between {FIRST_YEAR} and {last_year}')
    if not args.name.strip():
        parser.error('--name must not be empty')

    try:
        model = load_model(args.model)
    except FileNotFoundError:
        parser.exit(1, f'No model in {args.model}. Train it first: python -m scripts.models data/cars_clean.csv\n')
    except Exception as error:  # whatever the reason, a file that cannot be read is reported the same way
        # A model file can only be read with the version of scikit-learn that made it (or the file is broken)
        reason = (
            compatibility_problem(read_metadata(args.model))
            or 'it was probably saved with another version of scikit-learn'
        )
        parser.exit(
            1,
            f'The model in {args.model} cannot be read ({type(error).__name__}): {reason}. '
            'Train it again with python -m scripts.models data/cars_clean.csv\n',
        )
    problem = compatibility_problem(read_metadata(args.model))
    if problem:
        print(f'Warning: {problem}', file=sys.stderr)  # the model could be read, but the result may differ

    price = predict_price(
        model,
        normalize_name(args.name),
        normalize_fuel(args.fuel),
        normalize_gearbox(args.gearbox),
        args.km,
        args.year,
        power=args.power,
        battery=args.battery,
        trim=args.trim,
    )
    print(f'Estimated price: {euros(price)}')
    return price


if __name__ == '__main__':
    run(main)
