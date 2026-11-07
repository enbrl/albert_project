# training models for the project and saving the model
"""Train, compare, evaluate, save and use models that estimate the price of a used car.

The input is the cleaned file made by scripts/datacleaning.py, with the columns
Nom, Carburant, Boite de vitesse, Km, année, Prix (and optionally Marque and
Puissance, the power in horsepower, and Batterie, the capacity in kWh of the electric cars: they are used when
the file has them).

Usage (from the root of the project):

    python -m scripts.models data/cars_clean.csv                  # the best model is chosen and saved
    python -m scripts.models data/cars_clean.csv --kind forest    # or boosting, ridge
    python -m scripts.models data/cars_clean.csv --compare        # only show the comparison
    python -m scripts.models --info models/price_model.joblib     # what a saved model is

How the models are compared: the cars are cut in 5 parts; each model learns on 4 parts and is
tested on the 5th, 5 times, so every car is used for the test once. The table gives the average
errors (the lower the better). A model that always answers the mean price is in the table too: the
others must do much better.

To use a model in Python:

    from scripts.models import load_model, predict_price
    model = load_model('models/price_model.joblib')
    predict_price(model, 'Peugeot 208', 'Essence', 'Manuelle', km=60000, year=2019)

The labels (fuel, gearbox...) must be the ones of the cleaned file.
"""

import argparse
import datetime
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from scripts.cli import UserError, run

BASE_CATEGORICAL_FEATURES = ['Marque', 'Nom', 'Carburant', 'Boite de vitesse']
NUMERIC_FEATURES = ['Km', 'année']
# Used only when the data has them (and enough of them)
OPTIONAL_NUMERIC_FEATURES = ['Puissance', 'Batterie']
OPTIONAL_CATEGORICAL_FEATURES = ['Gamme']  # the family of the trim, made from 'Finition' (see add_trim_family)
CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + OPTIONAL_CATEGORICAL_FEATURES
FEATURES = BASE_CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = 'Prix'

# The models that can be trained. 'average' only answers the mean price (to compare with).
MODEL_KINDS = ('forest', 'boosting', 'ridge')
# The best model on the 2 693 cars of the whole aramisauto listing (see the comparison with --compare)
DEFAULT_KIND = 'boosting'

DEFAULT_MODEL_PATH = 'models/price_model.joblib'

# Below this number of cars, the measured errors would mean nothing
MIN_ROWS = 20

# A column such as Puissance is used only if at least this share of the cars have it
MIN_SHARE_FILLED = 0.5
# Batterie (kWh) only exists for electric cars: it is used if at least this share of the electric cars have it,
# and there are at least MIN_ELECTRIC_CARS of them (below, there is nothing to learn from)
ELECTRIC = 'Électrique'
MIN_ELECTRIC_CARS = 10


def add_trim_family(df):
    """
    The same cars with the column 'Gamme': the first word of the trim, in small letters ('Allure Pack' -> 'allure').

    The 790 trims of the listing are too many to learn each one, but they come in families (Allure, Allure Pack,
    Allure Suréquipée...) that say how well the car is equipped. Without 'Finition', the cars are returned unchanged.
    """
    if 'Finition' not in df.columns or 'Gamme' in df.columns:
        return df
    families = [
        trim.split()[0].lower() if isinstance(trim, str) and trim.split() else np.nan for trim in df['Finition']
    ]
    # object, even when every trim is unknown: a column of NaN only would be read as numbers, not as labels
    return df.assign(Gamme=pd.Series(families, index=df.index, dtype=object))


def is_usable(df, column):
    """Does the data have enough of this optional column to learn from it?"""
    if column not in df.columns:
        return False
    if column == 'Batterie':  # a petrol car has no battery capacity: look at the electric cars only
        battery = df.loc[df['Carburant'] == ELECTRIC, column] if 'Carburant' in df.columns else df[column].iloc[:0]
        return len(battery) >= MIN_ELECTRIC_CARS and battery.notna().mean() >= MIN_SHARE_FILLED
    return df[column].notna().mean() >= MIN_SHARE_FILLED


def feature_columns(df):
    """The columns used by the model for this data: the usual ones, plus the optional ones the data has."""
    df = add_trim_family(df)
    optional = OPTIONAL_NUMERIC_FEATURES + OPTIONAL_CATEGORICAL_FEATURES
    return FEATURES + [column for column in optional if is_usable(df, column)]


def prepare_features(df, columns=None):
    """
    Return the columns used by the model (the brand is the first word of the name if 'Marque' is missing).

    Parameters:
    - df: DataFrame of cars
    - columns: the columns to return (by default: feature_columns(df))
    """
    features = add_trim_family(df.copy())
    if 'Marque' not in features.columns:
        features['Marque'] = features['Nom'].astype(str).str.split().str[0]
    columns = columns if columns is not None else feature_columns(features)
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise UserError(f'Missing columns: {missing}')
    return features[columns]


def build_model(kind=DEFAULT_KIND, random_state=42, columns=None):
    """
    Build a model that is not trained yet.

    Parameters:
    - kind: 'forest' (random forest), 'boosting' (gradient boosting) or 'ridge' (linear model);
      'average' is a model that always answers the mean price (to compare with)
    - random_state: seed, to get the same result at each run
    - columns: the columns the model will use (by default: FEATURES)

    The labels (brand, fuel...) are encoded as numbers and the missing numbers (an unknown power)
    are replaced by the median. Labels never seen during the training are ignored instead of
    raising an error. The linear model and the boosting learn the logarithm of the price: a mistake
    of 10 % counts the same on a cheap car and on an expensive one.

    The boosting is told which columns are labels: it groups brands, fuels and gearboxes as categories
    (Audi and Volvo together, the others apart) instead of cutting their numbers (Audi=1 < BMW=2 <
    ... < Volvo=37, an order that means nothing). The model (300 names, more than the 255 categories it
    accepts) is replaced by its mean price, learnt on other cars than the one encoded (TargetEncoder
    cuts the cars in parts). On the whole listing this lowers the mean error from 1 420 to 1 357 EUR.
    """
    columns = columns if columns is not None else FEATURES
    categorical = [c for c in columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in columns if c not in CATEGORICAL_FEATURES]

    if kind == 'average':
        return Pipeline([('model', DummyRegressor(strategy='mean'))])

    numbers = SimpleImputer(strategy='median', add_indicator=True)
    if kind == 'forest':
        labels = OneHotEncoder(handle_unknown='ignore')
        regressor = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    elif kind == 'boosting':
        return _build_boosting(categorical, numeric, numbers, random_state)
    elif kind == 'ridge':
        labels = OneHotEncoder(handle_unknown='ignore')
        numbers = Pipeline(
            [('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]
        )
        regressor = TransformedTargetRegressor(Ridge(alpha=1.0), func=np.log, inverse_func=np.exp)
    else:
        raise ValueError(f"Unknown kind '{kind}', choose one of {MODEL_KINDS}")

    preprocessing = ColumnTransformer([('labels', labels, categorical), ('numbers', numbers, numeric)])
    return Pipeline([('preprocessing', preprocessing), ('model', regressor)])


# The model has too many names to be a category of the boosting: it is replaced by its mean price
TARGET_ENCODED = ['Nom']
# The most categories the boosting accepts in one column (the rarer values are grouped in one)
MAX_CATEGORIES = 255


def target_encoder(random_state, version=None):
    """
    The TargetEncoder of the boosting, with a fixed seed (the same model at every run).

    scikit-learn 1.9 takes the seed in the cross-validation (``cv``) and will remove ``random_state`` in 1.11,
    but before 1.9 ``cv`` can only be a number: the right way depends on the installed version.
    """
    major, minor = (int(part) for part in (version or sklearn.__version__).split('.')[:2])
    if (major, minor) >= (1, 9):
        return TargetEncoder(target_type='continuous', cv=KFold(5, shuffle=True, random_state=random_state))
    return TargetEncoder(target_type='continuous', shuffle=True, random_state=random_state)


def _build_boosting(categorical, numeric, numbers, random_state):
    """The gradient boosting of build_model, with its labels as categories (see build_model)."""
    encoded = [c for c in categorical if c in TARGET_ENCODED]
    labels = [c for c in categorical if c not in TARGET_ENCODED]
    # A label never seen (or missing) becomes -1, which the boosting reads as 'missing'
    # The boosting takes at most 255 categories: the rarest values are grouped (the listing has 251 trim families)
    ordinal = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        encoded_missing_value=-1,
        max_categories=MAX_CATEGORIES,
    )
    preprocessing = ColumnTransformer(
        [
            ('labels', ordinal, labels),
            ('names', target_encoder(random_state), encoded),
            ('numbers', numbers, numeric),
        ],
        verbose_feature_names_out=False,  # the columns keep their names: the boosting finds the labels by name
    ).set_output(transform='pandas')
    regressor = TransformedTargetRegressor(
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            min_samples_leaf=5,
            categorical_features=labels,
            random_state=random_state,
        ),
        func=np.log,
        inverse_func=np.exp,
    )
    return Pipeline([('preprocessing', preprocessing), ('model', regressor)])


def _check_size(df):
    if len(df) < MIN_ROWS:
        raise UserError(f'At least {MIN_ROWS} cars are needed to train a model, got {len(df)}')


def compare_models(df, kinds=MODEL_KINDS, n_splits=5, random_state=42):
    """
    Compare the models with a cross-validation (see the top of this file).

    Parameters:
    - df: cleaned DataFrame (with the column Prix)
    - kinds: the models to compare
    - n_splits: number of parts the cars are cut in

    Returns:
    - A DataFrame with one row per model, the best first: 'mae' (mean error in euros) and 'mae_std'
      (how much it varies between the parts), 'rmse', 'r2' (1 is perfect) and 'mape' (mean error in
      % of the price). The row 'average' is the model that always answers the mean price.
    """
    _check_size(df)
    columns = feature_columns(df)
    features, prices = prepare_features(df, columns), df[TARGET]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
        'mape': 'neg_mean_absolute_percentage_error',
    }

    rows = []
    for kind in ('average', *kinds):
        scores = cross_validate(
            build_model(kind, random_state, columns), features, prices, cv=splitter, scoring=scoring
        )
        rows.append(
            {
                'model': kind,
                'mae': -scores['test_mae'].mean(),
                'mae_std': scores['test_mae'].std(),
                'rmse': -scores['test_rmse'].mean(),
                'r2': scores['test_r2'].mean(),
                'mape': -scores['test_mape'].mean() * 100,
            }
        )
    return pd.DataFrame(rows).sort_values('mae').reset_index(drop=True)


def best_kind(comparison):
    """The model with the lowest mean error in a table made by compare_models (the average model is left out)."""
    return comparison[comparison['model'] != 'average'].iloc[0]['model']


def train_and_evaluate(df, test_size=0.2, random_state=42, kind=DEFAULT_KIND):
    """
    Train the model on a part of the cars and measure its errors on the other part.

    Parameters:
    - df: cleaned DataFrame (see scripts/datacleaning.py), with the column Prix
    - test_size: share of the cars kept to measure the errors
    - random_state: seed, to get the same result at each run
    - kind: the model to train (see build_model)

    Returns:
    - (model, metrics): the model trained on the training part, and a dictionary with
      'mae' (mean error in euros), 'rmse', 'r2' (1 is perfect), 'mape' (mean error in % of the
      price), 'baseline_mae' (error of a model that always answers the mean price: ours must be
      much lower), 'n_train' and 'n_test'
    """
    _check_size(df)
    columns = feature_columns(df)
    features = prepare_features(df, columns)
    prices = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, prices, test_size=test_size, random_state=random_state
    )

    model = build_model(kind, random_state, columns)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        'mae': float(mean_absolute_error(y_test, predictions)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
        'r2': float(r2_score(y_test, predictions)),
        'mape': float(mean_absolute_percentage_error(y_test, predictions) * 100),
        'baseline_mae': float(mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))),
        'n_train': int(len(x_train)),
        'n_test': int(len(x_test)),
    }
    return model, metrics


def train_final_model(df, random_state=42, kind=DEFAULT_KIND):
    """Train the model on all the cars (to use after the evaluation)."""
    columns = feature_columns(df)
    model = build_model(kind, random_state, columns)
    model.fit(prepare_features(df, columns), df[TARGET])
    return model


def feature_importance(df, kind=DEFAULT_KIND, test_size=0.2, n_repeats=10, random_state=42):
    """
    Which columns matter most for the price?

    The model is trained on a part of the cars. Then, on the other part, the values of one column
    are shuffled: the more the error grows, the more the model needed this column.

    Returns:
    - A DataFrame with one row per column, the most important first: 'importance' is the increase
      of the mean error (in euros) when the column is shuffled, 'std' how much it varies
    """
    _check_size(df)
    columns = feature_columns(df)
    features = prepare_features(df, columns)
    x_train, x_test, y_train, y_test = train_test_split(
        features, df[TARGET], test_size=test_size, random_state=random_state
    )
    model = build_model(kind, random_state, columns).fit(x_train, y_train)
    result = permutation_importance(
        model, x_test, y_test, scoring='neg_mean_absolute_error', n_repeats=n_repeats, random_state=random_state
    )
    table = pd.DataFrame({'column': columns, 'importance': result.importances_mean, 'std': result.importances_std})
    return table.sort_values('importance', ascending=False).reset_index(drop=True)


def metadata_path(filename):
    """Where the information about a model is saved: next to the model, with .json (models/price.joblib -> models/price.json)."""
    return os.path.splitext(filename)[0] + '.json'


def library_versions():
    """The versions of what is needed to read a saved model again."""
    return {
        'python': '.'.join(str(part) for part in sys.version_info[:3]),
        'scikit-learn': sklearn.__version__,
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'joblib': joblib.__version__,
    }


def make_metadata(df, kind, metrics, columns):
    """
    The information that tells what a saved model is.

    Parameters:
    - df: the cars the model was trained on
    - kind: the kind of model ('boosting'...)
    - metrics: the dictionary of errors given by train_and_evaluate (measured on cars the model had not seen)
    - columns: the columns the model uses
    """
    return {
        'created': datetime.date.today().isoformat(),
        'kind': kind,
        'cars': int(len(df)),
        'columns': list(columns),
        'metrics': {key: (round(value, 3) if isinstance(value, float) else value) for key, value in metrics.items()},
    }


def save_model(model, filename=DEFAULT_MODEL_PATH, metadata=None):
    """
    Save the model in a file (the folder is created if needed), and the information about it next to it.

    The information (see make_metadata) is saved with the versions of scikit-learn, pandas...
    in a .json file: a model can only be read with the version of scikit-learn that made it.
    """
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, filename)
    information = {**(metadata or {}), 'versions': library_versions()}
    with open(metadata_path(filename), 'w', encoding='utf-8') as file:
        json.dump(information, file, indent=2, ensure_ascii=False)
        file.write('\n')


def read_metadata(filename=DEFAULT_MODEL_PATH):
    """The information saved with a model, or None if there is none (or it cannot be read)."""
    try:
        with open(metadata_path(filename), encoding='utf-8') as file:
            information = json.load(file)
    except (OSError, ValueError):
        return None
    return information if isinstance(information, dict) else None


def _minor(version):
    """'1.6.1' -> '1.6' (the first two numbers)."""
    return '.'.join(str(version).split('.')[:2])


def compatibility_problem(metadata):
    """
    A sentence if the model was saved with another version of scikit-learn than the installed one, else None.

    Only the first two numbers count: 1.6.1 and 1.6.2 are the same.
    """
    saved = ((metadata or {}).get('versions') or {}).get('scikit-learn')
    if saved and _minor(saved) != _minor(sklearn.__version__):
        return (
            f'this model was saved with scikit-learn {saved}, but scikit-learn {sklearn.__version__} is installed: '
            'train it again if the results look wrong'
        )
    return None


def describe_model(metadata):
    """The lines that tell what a saved model is, from its information."""
    metrics = metadata.get('metrics') or {}
    lines = [
        f'Model: {metadata.get("kind", "?")}, trained on {metadata.get("cars", "?")} cars on {metadata.get("created", "?")}',
        f'Columns: {", ".join(metadata.get("columns") or [])}',
    ]
    if 'mae' in metrics:
        lines.append(
            f'Mean error on cars it had not seen: {metrics["mae"]:.0f} EUR ({metrics.get("mape", 0):.1f} % of the price)'
        )
    versions = metadata.get('versions') or {}
    lines.append('Saved with: ' + ', '.join(f'{name} {version}' for name, version in versions.items()))
    problem = compatibility_problem(metadata)
    if problem:
        lines.append(f'Warning: {problem}')
    return lines


def load_model(filename=DEFAULT_MODEL_PATH):
    """Load a model saved with save_model. Only load files you made yourself."""
    return joblib.load(filename)


def predict_price(model, name, fuel, gearbox, km, year, power=None, battery=None, trim=None):
    """
    Estimate the price (in euros) of one used car.

    Parameters:
    - name: for example 'Peugeot 208'
    - fuel: for example 'Essence'
    - gearbox: 'Manuelle' or 'Automatique'
    - km: mileage
    - year: year of the first registration
    - power: power in horsepower; used only if the model was trained with the power (otherwise ignored)
    - battery: capacity of the battery in kWh (electric cars); used only if the model was trained with it
    - trim: the trim, for example 'Allure Pack' (only its first word counts); used only if the model was trained with it
    """
    car = pd.DataFrame(
        [
            {
                'Nom': name,
                'Carburant': fuel,
                'Boite de vitesse': gearbox,
                'Km': km,
                'année': year,
                'Puissance': np.nan if power is None else power,
                'Batterie': np.nan if battery is None else battery,
                'Finition': trim if trim else np.nan,
            }
        ]
    )
    columns = list(model.feature_names_in_)  # the columns the model was trained with
    return float(model.predict(prepare_features(car, columns))[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare, train and save the used-car price model')
    parser.add_argument('data', nargs='?', help='cleaned CSV file (made by scripts/datacleaning.py)')
    parser.add_argument(
        '--model', default=DEFAULT_MODEL_PATH, help=f'file where the model is saved (default: {DEFAULT_MODEL_PATH})'
    )
    parser.add_argument(
        '--kind',
        default='best',
        choices=('best', *MODEL_KINDS),
        help="model to train (default: 'best', the one with the lowest error in the comparison)",
    )
    parser.add_argument(
        '--compare', action='store_true', help='only show the comparison of the models, do not save anything'
    )
    parser.add_argument(
        '--info', metavar='MODEL', help='show what a saved model is (date, cars, error, versions) and stop'
    )
    args = parser.parse_args(argv)

    if args.info:
        metadata = read_metadata(args.info)
        if metadata is None:
            raise UserError(f'No information about the model {args.info} (looked in {metadata_path(args.info)})')
        print('\n'.join(describe_model(metadata)))
        return metadata
    if not args.data:
        parser.error('the data file is required (or use --info MODEL)')

    df = pd.read_csv(args.data)
    columns = feature_columns(df)
    print(f'{len(df)} cars, columns used: {", ".join(columns)}')

    comparison = None
    if args.compare or args.kind == 'best':
        comparison = compare_models(df)
        print('\nComparison (5-fold cross-validation, errors in euros except mape in %):')
        print(comparison.round({'mae': 0, 'mae_std': 0, 'rmse': 0, 'r2': 2, 'mape': 1}).to_string(index=False))
    if args.compare:
        return comparison

    kind = best_kind(comparison) if args.kind == 'best' else args.kind
    _, metrics = train_and_evaluate(df, kind=kind)
    print(f'\nModel: {kind}. Cars used to train: {metrics["n_train"]}, to test: {metrics["n_test"]}')
    print(
        f'Mean error: {metrics["mae"]:.0f} EUR, {metrics["mape"]:.1f} % of the price '
        f'(always guessing the mean price: {metrics["baseline_mae"]:.0f} EUR)'
    )
    print(f'RMSE: {metrics["rmse"]:.0f} EUR, R2: {metrics["r2"]:.2f}')

    model = train_final_model(df, kind=kind)  # the saved model learns from all the cars
    save_model(model, args.model, make_metadata(df, kind, metrics, columns))
    print(f'Model saved in {args.model}')
    return metrics


if __name__ == '__main__':
    run(main)
