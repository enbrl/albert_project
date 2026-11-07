#training models for the project and saving the model
"""Train, evaluate, save and use a model that estimates the price of a used car.

The input is the cleaned file made by scripts/datacleaning.py, with the columns
Nom, Carburant, Boite de vitesse, Km, année, Prix (and optionally Marque).

Usage (from the root of the project):

    python -m scripts.models data/cars_clean.csv --model models/price_model.joblib

The command trains the model, prints the errors measured on cars the model has
never seen, and saves the model. To use it in Python:

    from scripts.models import load_model, predict_price
    model = load_model('models/price_model.joblib')
    predict_price(model, 'Peugeot 208', 'Essence', 'Manuelle', km=60000, year=2019)

The labels (fuel, gearbox...) must be the ones of the cleaned file.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_FEATURES = ['Marque', 'Nom', 'Carburant', 'Boite de vitesse']
NUMERIC_FEATURES = ['Km', 'année']
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = 'Prix'

DEFAULT_MODEL_PATH = 'models/price_model.joblib'

# Below this number of cars, the measured errors would mean nothing
MIN_ROWS = 20


def prepare_features(df):
    """
    Return the columns used by the model (the brand is the first word of the name if 'Marque' is missing).
    """
    features = df.copy()
    if 'Marque' not in features.columns:
        features['Marque'] = features['Nom'].astype(str).str.split().str[0]
    missing = [column for column in FEATURES if column not in features.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    return features[FEATURES]


def build_model(random_state=42):
    """
    Build the (not trained) model: one-hot encoding of the labels + random forest.

    A random forest needs no scaling and handles the link between mileage, age and price well.
    Labels never seen during the training are ignored instead of raising an error.
    """
    preprocessing = ColumnTransformer([
        ('labels', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ('numbers', 'passthrough', NUMERIC_FEATURES),
    ])
    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    return Pipeline([('preprocessing', preprocessing), ('forest', forest)])


def train_and_evaluate(df, test_size=0.2, random_state=42):
    """
    Train the model on a part of the cars and measure its errors on the other part.

    Parameters:
    - df: cleaned DataFrame (see scripts/datacleaning.py), with the column Prix
    - test_size: share of the cars kept to measure the errors
    - random_state: seed, to get the same result at each run

    Returns:
    - (model, metrics): the model trained on the training part, and a dictionary with
      'mae' (mean error in euros), 'rmse', 'r2' (1 is perfect), 'baseline_mae' (error
      of a model that always answers the mean price: ours must be much lower),
      'n_train' and 'n_test'
    """
    if len(df) < MIN_ROWS:
        raise ValueError(f'At least {MIN_ROWS} cars are needed, got {len(df)}')

    features = prepare_features(df)
    prices = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(features, prices, test_size=test_size, random_state=random_state)

    model = build_model(random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        'mae': float(mean_absolute_error(y_test, predictions)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
        'r2': float(r2_score(y_test, predictions)),
        'baseline_mae': float(mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))),
        'n_train': int(len(x_train)),
        'n_test': int(len(x_test)),
    }
    return model, metrics


def train_final_model(df, random_state=42):
    """Train the model on all the cars (to use after the evaluation)."""
    model = build_model(random_state)
    model.fit(prepare_features(df), df[TARGET])
    return model


def save_model(model, filename=DEFAULT_MODEL_PATH):
    """Save the model in a file (the folder is created if needed)."""
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, filename)


def load_model(filename=DEFAULT_MODEL_PATH):
    """Load a model saved with save_model. Only load files you made yourself."""
    return joblib.load(filename)


def predict_price(model, name, fuel, gearbox, km, year):
    """
    Estimate the price (in euros) of one used car.

    Parameters:
    - name: for example 'Peugeot 208'
    - fuel: for example 'Essence'
    - gearbox: 'Manuelle' or 'Automatique'
    - km: mileage
    - year: year of the first registration
    """
    car = pd.DataFrame([{
        'Nom': name,
        'Carburant': fuel,
        'Boite de vitesse': gearbox,
        'Km': km,
        'année': year,
    }])
    return float(model.predict(prepare_features(car))[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train the used-car price model')
    parser.add_argument('data', help='cleaned CSV file (made by scripts/datacleaning.py)')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help=f'file where the model is saved (default: {DEFAULT_MODEL_PATH})')
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    _, metrics = train_and_evaluate(df)
    print(f"Cars used to train: {metrics['n_train']}, to test: {metrics['n_test']}")
    print(f"Mean error: {metrics['mae']:.0f} EUR (always guessing the mean price: {metrics['baseline_mae']:.0f} EUR)")
    print(f"RMSE: {metrics['rmse']:.0f} EUR, R2: {metrics['r2']:.2f}")

    model = train_final_model(df)  # the saved model learns from all the cars
    save_model(model, args.model)
    print(f'Model saved in {args.model}')
    return metrics


if __name__ == '__main__':
    main()
