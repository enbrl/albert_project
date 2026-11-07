# data cleaning script
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
import unicodedata

import pandas as pd

from scripts.aramis_scraper import parse_power
from scripts.cli import UserError, run

COLUMNS = ['Nom', 'Carburant', 'Boite de vitesse', 'Km', 'année', 'Prix']

# Value used when the fuel or the gearbox is not given or not understood
UNKNOWN = 'Inconnu'

# Optional columns of the scraper: kept and cleaned when they are in the file
ENGINE_COLUMNS = ['Motorisation', 'Finition']
POWER_COLUMN = 'Puissance'

# Limits used to remove impossible values (typos, test listings...)
FIRST_YEAR = 1990
MAX_KM = 1_000_000
MIN_POWER = 30
MAX_POWER = 1500


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


# The brands that need a rule: the name of the brand is not its first word, or it is spelled in several ways
# (the legacy files of the notebooks write "Citroen", the website writes "Citroën").
# canonical spelling -> the spellings found in the data, without accents and in lower case
BRAND_SPELLINGS = {
    'Alfa Romeo': ['alfa romeo'],
    'Aston Martin': ['aston martin'],
    'Land Rover': ['land rover'],
    'Rolls-Royce': ['rolls-royce', 'rolls royce'],
    'Lynk & Co': ['lynk & co', 'lynk&co'],
    'Mercedes': ['mercedes', 'mercedes-benz', 'mercedes benz'],
    'Citroën': ['citroen'],
    'Skoda': ['skoda'],
}


def without_accents(text):
    """'Citroën' -> 'Citroen'."""
    decomposed = unicodedata.normalize('NFKD', text)
    return ''.join(char for char in decomposed if not unicodedata.combining(char))


def split_brand(name):
    """
    Split a name into the brand and the model.

    'Alfa Romeo Tonale' -> ('Alfa Romeo', 'Tonale'), 'Citroen C4' -> ('Citroën', 'C4'),
    'Peugeot 208' -> ('Peugeot', '208'). By default the brand is the first word.
    """
    words = name.split()
    best = None  # (number of words of the spelling, canonical brand)
    for brand, spellings in BRAND_SPELLINGS.items():
        for spelling in spellings:
            count = len(spelling.split())
            matches = len(words) >= count and without_accents(' '.join(words[:count])).lower() == spelling
            if matches and (best is None or count > best[0]):
                best = (count, brand)
    if best is None:
        return (words[0], ' '.join(words[1:])) if words else ('', '')
    return best[1], ' '.join(words[best[0] :])


def brand_of(name):
    """The brand of a (clean) name: 'Alfa Romeo Tonale' -> 'Alfa Romeo'."""
    return split_brand(name)[0]


def normalize_name(value):
    """
    '  PEUGEOT   308 ' and 'peugeot 308' both become 'Peugeot 308' (short names like MG or C4 stay in capitals).

    The brand is written in only one way: 'CITROEN C4' becomes 'Citroën C4', 'Mercedes-Benz GLA' 'Mercedes GLA'.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    words = value.split()
    text = ' '.join(w if (w.isupper() and len(w) <= 3) else w.capitalize() for w in words)
    brand, model = split_brand(text)
    return f'{brand} {model}'.strip()


def normalize_text(value):
    """Remove the extra spaces of a text; an empty text becomes None."""
    if not isinstance(value, str):
        return None
    text = ' '.join(value.split())
    return text or None


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


def clean_cars_with_report(df, current_year=None):
    """
    Clean a DataFrame of used cars.

    Steps: numbers converted to integers, labels harmonised, rows without name,
    mileage, year or price removed, impossible values removed, duplicates removed.

    Parameters:
    - df: DataFrame with at least the columns Nom, Carburant, Boite de vitesse, Km, année, Prix
      (and optionally 'Id', the number of the listing: a car with the same Id is kept once)
    - current_year: the latest accepted year (by default: next year)

    Returns:
    - (cleaned, report): the new DataFrame and a dictionary that counts the rows ('read', then
      'missing' (no name, mileage, year or price), 'impossible' (impossible values), 'same_id' and
      'duplicates' that were removed, and 'kept'). The DataFrame has the same columns plus 'Marque'
      (the brand: 'Alfa Romeo' for 'Alfa Romeo Tonale'). The optional columns Motorisation, Finition
      and Puissance (the power in horsepower, empty when it is not possible) are cleaned and kept
      when the input has them.
    """
    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise UserError(f'Missing columns: {missing}')
    if current_year is None:
        current_year = datetime.date.today().year + 1  # a car of the next model year can be on sale

    has_id = 'Id' in df.columns
    optional = [column for column in ENGINE_COLUMNS + [POWER_COLUMN] if column in df.columns]
    cleaned = df[(['Id'] if has_id else []) + COLUMNS + optional].copy()
    if has_id:
        cleaned['Id'] = pd.array(cleaned['Id'].map(to_number), dtype='Int64')
    cleaned['Nom'] = cleaned['Nom'].map(normalize_name)
    cleaned['Carburant'] = cleaned['Carburant'].map(normalize_fuel)
    cleaned['Boite de vitesse'] = cleaned['Boite de vitesse'].map(normalize_gearbox)
    for column in ('Km', 'année', 'Prix'):
        cleaned[column] = cleaned[column].map(to_number)

    for column in ENGINE_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].map(normalize_text)
    if 'Motorisation' in cleaned.columns:
        # The power is read again from the text of the engine, so that a file made by an older version of the scraper is
        # repaired without scraping again (the first version took the '40' of the Audi '40 TFSIe 204' for the power).
        # Without the text of the engine (a file of the notebooks), the power already in the file is kept.
        stored = cleaned[POWER_COLUMN] if POWER_COLUMN in cleaned.columns else [None] * len(cleaned)
        cleaned[POWER_COLUMN] = [
            parse_power(engine, name) if isinstance(engine, str) else old
            for engine, name, old in zip(cleaned['Motorisation'], cleaned['Nom'], stored)  # noqa: B905 (strict= needs Python 3.10)
        ]
    if POWER_COLUMN in cleaned.columns:  # a power that is not possible is empty, the car is kept
        power = cleaned[POWER_COLUMN].map(to_number)
        power = power.where(power.between(MIN_POWER, MAX_POWER))
        cleaned[POWER_COLUMN] = pd.array(power, dtype='Int64')

    report = {'read': len(df)}

    # The model needs a name, a mileage, a year and a price
    count = len(cleaned)
    cleaned = cleaned.dropna(subset=['Nom', 'Km', 'année', 'Prix'])
    report['missing'] = count - len(cleaned)

    # Remove impossible values
    count = len(cleaned)
    cleaned = cleaned[
        (cleaned['Prix'] > 0)
        & (cleaned['Km'] >= 0)
        & (cleaned['Km'] <= MAX_KM)
        & (cleaned['année'] >= FIRST_YEAR)
        & (cleaned['année'] <= current_year)
    ]

    report['impossible'] = count - len(cleaned)

    cleaned = cleaned.astype({'Km': 'int64', 'année': 'int64', 'Prix': 'int64'})
    count = len(cleaned)
    if has_id:  # the same Id is the same car, even if it was scraped twice
        cleaned = cleaned[cleaned['Id'].isna() | ~cleaned['Id'].duplicated()]
    report['same_id'] = count - len(cleaned)
    count = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    report['duplicates'] = count - len(cleaned)
    cleaned['Marque'] = cleaned['Nom'].map(brand_of)
    report['kept'] = len(cleaned)
    return cleaned, report


def clean_cars(df, current_year=None):
    """Clean a DataFrame of used cars (see clean_cars_with_report, which also counts the removed rows)."""
    return clean_cars_with_report(df, current_year)[0]


REPORT_LINES = [
    ('missing', 'removed: no name, mileage, year or price'),
    ('impossible', 'removed: impossible values (price, mileage or year)'),
    ('same_id', 'removed: same listing Id seen twice'),
    ('duplicates', 'removed: identical rows'),
]


def format_report(report):
    """The lines that tell what the cleaning did, from the dictionary made by clean_cars_with_report."""
    lines = [f'{report["read"]} rows read']
    lines += [f'{report[key]:>7} {text}' for key, text in REPORT_LINES]
    lines.append(f'{report["kept"]} rows kept')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Clean the scraped used-car data')
    parser.add_argument('inputs', nargs='+', help='one or several CSV files to clean')
    parser.add_argument(
        '-o', '--output', default='data/cars_clean.csv', help='CSV file to write (default: data/cars_clean.csv)'
    )
    args = parser.parse_args(argv)

    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    cleaned, report = clean_cars_with_report(raw)

    folder = os.path.dirname(args.output)
    if folder:
        os.makedirs(folder, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(format_report(report))
    print(f'Saved in {args.output}')
    return cleaned


if __name__ == '__main__':
    run(main)
