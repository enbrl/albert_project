#data cleaning script
"""Clean the scraped used-car data so that it can be used by the model.

The input has the columns of the scrapers and of the notebooks:
Nom, Carburant, Boite de vitesse, Km, année, Prix.
The values can be messy (for example 57 844 km or 18 690 € in some rows and
19160 or 19999 in others, 'PEUGEOT 308' and 'Peugeot 308', 'Auto.' and
'Automatique').

Usage (from the root of the project):

    python -m scripts.datacleaning data/cars_raw.csv -o data/cars_clean.csv

Several input files can be given: they are merged and the duplicates are removed.
"""

import argparse
import datetime
import os
import re

import pandas as pd

COLUMNS = ['Nom', 'Carburant', 'Boite de vitesse', 'Km', 'année', 'Prix']

# Value used when the fuel or the gearbox is not given or not understood
UNKNOWN = 'Inconnu'

# Limits used to remove impossible values (typos, test listings...)
FIRST_YEAR = 1990
MAX_KM = 1_000_000


def to_number(value):
    """
    Convert '57 844 km', '18 690 €', 19160 or 2021.0 to an integer.

    Returns:
    - The integer, or None if the value is empty or has no digit
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    digits = re.sub(r'\D', '', str(value))  # keep only the digits (spaces, 'km', '€'... are removed)
    return int(digits) if digits else None


def normalize_name(value):
    """'  PEUGEOT   308 ' and 'peugeot 308' both become 'Peugeot 308' (short names like MG or C4 stay in capitals)."""
    if not isinstance(value, str) or not value.strip():
        return None
    words = value.split()
    return ' '.join(w if (w.isupper() and len(w) <= 3) else w.capitalize() for w in words)


def normalize_fuel(value):
    """Group the fuel labels: Essence, Diesel, Électrique, Hybride, Hybride rechargeable, GPL."""
    if not isinstance(value, str) or not value.strip():
        return UNKNOWN
    text = value.lower()
    if 'électrique' in text or 'electrique' in text:
        return 'Électrique'
    if 'rechargeable' in text or 'phev' in text:
        return 'Hybride rechargeable'
    if 'hybrid' in text:  # 'hybride essence', 'mild hybrid essence', 'hybrid diesel'...
        return 'Hybride'
    if 'diesel' in text:
        return 'Diesel'
    if 'essence' in text:
        return 'Essence'
    if 'gpl' in text:
        return 'GPL'
    return value.strip().capitalize()


def normalize_gearbox(value):
    """Group the gearbox labels in Manuelle or Automatique ('Auto.', 'Double embrayage / DCT'...)."""
    if not isinstance(value, str) or not value.strip():
        return UNKNOWN
    text = value.lower()
    if 'manu' in text or 'méca' in text:
        return 'Manuelle'
    if any(word in text for word in ('auto', 'dct', 'double embrayage', 'robotis', 'cvt')):
        return 'Automatique'
    return UNKNOWN


def clean_cars(df, current_year=None):
    """
    Clean a DataFrame of used cars.

    Steps: numbers converted to integers, labels harmonised, rows without name,
    mileage, year or price removed, impossible values removed, duplicates removed.

    Parameters:
    - df: DataFrame with at least the columns Nom, Carburant, Boite de vitesse, Km, année, Prix
      (and optionally 'Id', the number of the listing: a car with the same Id is kept once)
    - current_year: the latest accepted year (by default: next year)

    Returns:
    - A new DataFrame with the same columns plus 'Marque' (the first word of the name)
    """
    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if current_year is None:
        current_year = datetime.date.today().year + 1  # a car of the next model year can be on sale

    has_id = 'Id' in df.columns
    cleaned = df[(['Id'] if has_id else []) + COLUMNS].copy()
    if has_id:
        cleaned['Id'] = pd.array(cleaned['Id'].map(to_number), dtype='Int64')
    cleaned['Nom'] = cleaned['Nom'].map(normalize_name)
    cleaned['Carburant'] = cleaned['Carburant'].map(normalize_fuel)
    cleaned['Boite de vitesse'] = cleaned['Boite de vitesse'].map(normalize_gearbox)
    for column in ('Km', 'année', 'Prix'):
        cleaned[column] = cleaned[column].map(to_number)

    # The model needs a name, a mileage, a year and a price
    cleaned = cleaned.dropna(subset=['Nom', 'Km', 'année', 'Prix'])

    # Remove impossible values
    cleaned = cleaned[
        (cleaned['Prix'] > 0)
        & (cleaned['Km'] >= 0) & (cleaned['Km'] <= MAX_KM)
        & (cleaned['année'] >= FIRST_YEAR) & (cleaned['année'] <= current_year)
    ]

    cleaned = cleaned.astype({'Km': 'int64', 'année': 'int64', 'Prix': 'int64'})
    if has_id:  # the same Id is the same car, even if it was scraped twice
        cleaned = cleaned[cleaned['Id'].isna() | ~cleaned['Id'].duplicated()]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned['Marque'] = cleaned['Nom'].str.split().str[0]
    return cleaned


def main(argv=None):
    parser = argparse.ArgumentParser(description='Clean the scraped used-car data')
    parser.add_argument('inputs', nargs='+', help='one or several CSV files to clean')
    parser.add_argument('-o', '--output', default='data/cars_clean.csv', help='CSV file to write (default: data/cars_clean.csv)')
    args = parser.parse_args(argv)

    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    cleaned = clean_cars(raw)

    folder = os.path.dirname(args.output)
    if folder:
        os.makedirs(folder, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f'{len(raw)} rows read, {len(cleaned)} rows kept, saved in {args.output}')
    return cleaned


if __name__ == '__main__':
    main()
