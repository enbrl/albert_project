"""Write docs/model-card.md from the cleaned data: every number of the card is measured, none is typed by hand.

After a new scraping, the card says the truth again with one command (from the root of the project):

    python -m scripts.model_card data/cars_clean.csv
    python -m scripts.model_card data/cars_clean.csv --scraped 2026-11-07 --output docs/model-card.md

It compares the models, measures what matters most, evaluates the error by group of cars (scripts/evaluate.py) and
writes the card. It takes about a minute on the 2 700 cars of the whole listing.
"""

import argparse
import datetime
import os
import platform

import numpy as np
import pandas as pd
import sklearn

from scripts.cli import run
from scripts.evaluate import MIN_CARS, evaluate, predict_out_of_fold
from scripts.models import (
    ELECTRIC,
    add_trim_family,
    best_kind,
    compare_models,
    feature_columns,
    feature_importance,
)

DEFAULT_OUTPUT = 'docs/model-card.md'

# What the reader reads for each value of the data (the data is in French)
FUEL_NAMES = {
    'Essence': 'petrol',
    'Hybride': 'hybrid',
    'Diesel': 'diesel',
    'Électrique': 'electric',
    'Hybride rechargeable': 'plug-in hybrid',
    'GPL': 'LPG',
    'Éthanol e85': 'ethanol E85',
}
COLUMN_NAMES = {
    'Puissance': 'power',
    'année': 'year',
    'Km': 'mileage',
    'Nom': 'model',
    'Marque': 'brand',
    'Carburant': 'fuel',
    'Boite de vitesse': 'gearbox',
    'Gamme': 'family of the trim',
    'Batterie': 'battery',
}
MODEL_NAMES = {
    'boosting': 'boosting (gradient boosting)',
    'ridge': 'ridge (linear)',
    'forest': 'random forest',
    'average': 'always the average price',
}

# How many of the worst single errors the card lists, and how many brands it shows (the biggest and the worst)
WORST_CARS = 3
BIGGEST_BRANDS = 10
WORST_BRANDS = 5


def euros(value):
    """12345.6 -> '12 346 €' (a space every three digits, like the rest of the card)."""
    return f'{value:,.0f}'.replace(',', ' ') + ' €'


def count(value):
    """2693 -> '2 693'."""
    return f'{value:,}'.replace(',', ' ')


def signed(value):
    """-921.4 -> '-921 €', 340 -> '+340 €'."""
    return f'{value:+,.0f}'.replace(',', ' ') + ' €'


def percent(part, whole):
    return f'{100 * part / whole:.1f} %' if whole else '0.0 %'


def error_table(rows, title):
    """A table of the groups of scripts/evaluate.py: group, cars, mean error, % of the price, bias."""
    lines = [f'| {title} | Cars | Mean error | % of the price | Bias |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(
            f'| {row["group"]} | {count(int(row["cars"]))} | {euros(row["mae"])} | {row["mape"]:.1f} % | '
            f'{signed(row["bias"])} |'
        )
    return '\n'.join(lines)


def brand_rows(table):
    """The brands with the most cars and the worst estimated ones, the worst first."""
    biggest = set(table.sort_values('cars', ascending=False)['group'][:BIGGEST_BRANDS])
    worst = set(table.sort_values('mape', ascending=False)['group'][:WORST_BRANDS])
    return table[table['group'].isin(biggest | worst)].sort_values('mape', ascending=False).to_dict('records')


def describe_data(df):
    """The figures of the section Data: size, prices, years, fuels and how often the optional columns are filled."""
    cars = add_trim_family(df)
    facts = {
        'cars': len(cars),
        'brands': cars['Marque'].nunique() if 'Marque' in cars.columns else cars['Nom'].str.split().str[0].nunique(),
        'models': cars['Nom'].nunique(),
        'price_min': cars['Prix'].min(),
        'price_max': cars['Prix'].max(),
        'price_median': cars['Prix'].median(),
        'year_min': int(cars['année'].min()),
        'year_max': int(cars['année'].max()),
        'fuels': cars['Carburant'].value_counts(),
        'columns': list(df.columns),
    }
    if 'Puissance' in cars.columns:
        facts['power'] = int(cars['Puissance'].notna().sum())
    if 'Gamme' in cars.columns:
        facts['trims'] = int(cars['Finition'].notna().sum())
        facts['trim_names'] = cars['Finition'].nunique()
        facts['families'] = cars['Gamme'].nunique()
    if 'Batterie' in cars.columns:
        electric = cars['Carburant'] == ELECTRIC
        facts['electric'] = int(electric.sum())
        facts['battery_electric'] = int(cars.loc[electric, 'Batterie'].notna().sum())
    return facts


def worst_cars(df, estimates, number=WORST_CARS):
    """The cars the model is the most wrong about: name, price and estimate, the worst first."""
    error = (estimates - df['Prix']).abs()
    return [
        {'name': df.loc[index, 'Nom'], 'price': df.loc[index, 'Prix'], 'estimate': estimates[index]}
        for index in error.sort_values(ascending=False).index[:number]
    ]


def measure(df, kind=None, n_repeats=5):
    """
    Every number of the card, measured on the cleaned cars (the slow part: about a minute on the whole listing).

    The model described is the one ``python -m scripts.models`` keeps: the best of the comparison (or ``kind``).
    """
    comparison = compare_models(df)
    kind = kind or best_kind(comparison)
    estimates = predict_out_of_fold(df, kind)
    return {
        'data': describe_data(df),
        'comparison': comparison,
        'importance': feature_importance(df, kind=kind, n_repeats=n_repeats),
        'evaluation': evaluate(df, kind=kind, estimates=estimates),
        'worst': worst_cars(df, estimates),
        'features': feature_columns(df),
        'kind': kind,
    }


def _data_rows(data):
    fuels = ', '.join(f'{FUEL_NAMES.get(fuel, fuel)} {count(int(cars))}' for fuel, cars in data['fuels'].items())
    rows = [
        '| Source | the public used-car listing of aramisauto.com (`/voitures/occasion/`), a reconditioned-car dealer |',
        '| Collected | @SCRAPED@, one request every 5 seconds (the `Crawl-delay` of its `robots.txt`) |',
        f'| Size | {count(data["cars"])} cars, {data["brands"]} brands, {count(data["models"])} models; asking prices from '
        f'{euros(data["price_min"])} to {euros(data["price_max"])} (median {euros(data["price_median"])}); '
        f'years {data["year_min"]} to {data["year_max"]} |',
        f'| Fuel | {fuels} |',
        '| Columns | ' + ', '.join(f'`{column}`' for column in data['columns']) + ' |',
    ]
    if 'power' in data:
        rows.append(
            f'| Power | found in the text of the engine for {percent(data["power"], data["cars"])} of the cars '
            '(for the others it is empty and the model uses the median) |'
        )
    if 'families' in data:
        rows.append(
            f'| Trim | written for {percent(data["trims"], data["cars"])} of the cars ({count(data["trim_names"])} '
            f'trims); the model uses its first word, the family (`Allure Pack` -> `allure`: {data["families"]} families) |'
        )
    if 'electric' in data:
        rows.append(
            f'| Battery | the capacity in kWh, found in the text of the engine for {data["battery_electric"]} of the '
            f'{data["electric"]} electric cars (`50 kWh - 136`); a value outside 5 to 150 kWh is a typo and is emptied |'
        )
    return '\n'.join(['| | |', '|---|---|', *rows])


def _comparison_table(comparison, kind):
    lines = ['| Model | Mean error | % of the price | R² |', '|---|---:|---:|---:|']
    for row in comparison.to_dict('records'):
        name = MODEL_NAMES.get(row['model'], row['model'])
        error = euros(row['mae'])
        if row['model'] == kind:
            name, error = f'**{name}, kept**', f'**{error}**'
        lines.append(f'| {name} | {error} | {row["mape"]:.1f} % | {max(row["r2"], 0):.2f} |')
    return '\n'.join(lines)


def _boosting_note(kind, data):
    """How the boosting reads the labels (scripts/models.py: build_model), when it is the model kept."""
    if kind != 'boosting':
        return []
    return [
        'The boosting reads brand, fuel and gearbox as categories; the model of the car '
        f'({count(data["models"])} names, too many categories for it) is replaced by its mean price.',
        '',
    ]


def _importance_text(importance):
    return ', '.join(
        f'{COLUMN_NAMES.get(row["column"], row["column"])} {euros(row["importance"])}'
        for row in importance.to_dict('records')
    )


def _too(bias):
    return 'too low' if bias < 0 else 'too high'


def write_card(results, scraped, versions=None):
    """The text of the card (Markdown) from the results of measure()."""
    data, evaluation, kind = results['data'], results['evaluation'], results['kind']
    comparison = results['comparison'].set_index('model')
    overall = evaluation['overall']
    average = comparison.loc['average'] if 'average' in comparison.index else None
    rarity = evaluation['model_rarity'].set_index('group')
    rare = rarity[rarity.index.str.startswith('rare')]
    common = rarity[rarity.index.str.startswith('common')]
    prices = evaluation['price'].set_index('group')
    versions = versions or installed_versions()
    names = [COLUMN_NAMES.get(column, column) for column in results['features']]

    lines = [
        '# Model card: used-car price estimate',
        '',
        'What the price model is, what it was trained on, how wrong it is and where, and what it must not be used for.',
        f'This card is written by `python -m scripts.model_card` from the data scraped on **{scraped}**: every number is '
        'measured, none is typed by hand. Run it again after a new scraping.',
        '',
        '## Summary',
        '',
        f'The model ({kind}) estimates the **asking price of a used car** from its {", ".join(names)}. It was trained on '
        f'**{count(data["cars"])} cars of the aramisauto.com used-car listing**. Measured on cars it had not seen (5-fold '
        f'cross-validation), it is wrong by **{euros(overall["mae"])} on average, {overall["mape"]:.1f} % of the price**.',
    ]
    if average is not None:
        lines.append(
            f'Always answering the average price would be wrong by {euros(average["mae"])} ({average["mape"]:.1f} %).'
        )
    lines += [
        'It is a **learning project**, not a valuation tool.',
        '',
        '## Intended use',
        '',
        '- Showing the whole data-science chain on a real listing: scraping, cleaning, modelling, evaluation.',
        '- Getting an order of magnitude for a **common** car of the listing (a Peugeot 208, a Renault Clio...), next to',
        '  the real cars of the data that look like it (the web application shows the five closest).',
        '',
        '**Not intended for**: buying or selling decisions, insurance, taxes or any valuation that somebody relies on; cars',
        'that are not sold by this website; models that are rare in the data (see the limits); any use that treats the',
        'estimate as a price somebody will pay.',
        '',
        '## Data',
        '',
        _data_rows(data).replace('@SCRAPED@', scraped),
        '',
        'The CSV files are **not** in the repository (`.gitignore`). No personal data is collected: a listing has a number,',
        'a model and a price. Read the terms of use of the website before publishing the data.',
        '',
        '## Model',
        '',
        f'- **Features**: {", ".join(names)}. The condition of the car, its options and its colour are not in the listing',
        '  page, so they are not used.',
        '- **Target**: the asking price. The linear model and the boosting learn the **logarithm** of the price, so that a',
        '  mistake of 10 % counts the same on a cheap car and on an expensive one.',
        '- **Choice**: the models are compared on the same 5 parts of the cars (`python -m scripts.models --compare`):',
        '',
        _comparison_table(results['comparison'], kind),
        '',
        *_boosting_note(kind, data),
        f'- **What matters most** (how much the error grows when a column is shuffled): {_importance_text(results["importance"])}.',
        '',
        '## Where the model is wrong',
        '',
        'Cross-validation, the cars cut in groups (`python -m scripts.evaluate`). **Bias** is the mean of (estimate - real',
        f'price): negative means the model estimates too low. Groups of fewer than {MIN_CARS} cars are left out.',
        '',
        '**By how well the data knows the model**:',
        '',
        error_table(evaluation['model_rarity'].to_dict('records'), 'Model'),
    ]
    if len(rare) and len(common):
        rare_row, common_row = rare.iloc[0], common.iloc[0]
        lines += [
            '',
            f'A model with fewer than {MIN_CARS} cars in the data is estimated with an error '
            f'**{rare_row["mae"] / common_row["mae"]:.1f} times** that of a common model, and {_too(rare_row["bias"])} by '
            f'{euros(abs(rare_row["bias"]))} on average: the model has hardly seen it, so it leans on the brand, the year '
            f'and the mileage. {count(int(rare_row["cars"]))} cars of the data are in this case.',
        ]
    lines += ['', '**By price**:', '', error_table(evaluation['price'].to_dict('records'), 'Price')]
    if {'cheapest quarter', 'most expensive quarter'} <= set(prices.index):
        cheap, dear = prices.loc['cheapest quarter', 'bias'], prices.loc['most expensive quarter', 'bias']
        lines += [
            '',
            f'The most expensive quarter is estimated {_too(dear)} by {euros(abs(dear))} on average, the cheapest quarter '
            f'{_too(cheap)} by {euros(abs(cheap))}.',
        ]
    worst = '; '.join(
        f'a {car["name"]} at {euros(car["price"])} estimated near {euros(round(car["estimate"], -2))}'
        for car in results['worst']
    )
    lines += [
        f'The worst single errors: {worst}.',
        '',
        '**By fuel**:',
        '',
        error_table(evaluation['fuel'].to_dict('records'), 'Fuel'),
        '',
        f'**By brand** (the {BIGGEST_BRANDS} brands with the most cars, and the {WORST_BRANDS} worst):',
        '',
        error_table(brand_rows(evaluation['brand']), 'Brand'),
        '',
        '**By age and by mileage**:',
        '',
        error_table(evaluation['age'].to_dict('records'), 'Age'),
        '',
        error_table(evaluation['mileage'].to_dict('records'), 'Mileage'),
        '',
        '## Limits and risks',
        '',
        '- **Asking price, not sale price.** The data is what the dealer asks, not what a buyer pays.',
        f'- **One seller, one day.** All the cars come from one reconditioned-car dealer on {scraped}. Other sellers or',
        '  other months can have other prices; the model does not know that prices move with time.',
        '- **No condition, options or colour.** Two cars of the same model, year and mileage can have different prices for',
        '  reasons the model cannot see.',
        '- **Rare models** are a weak point (tables above). A tool that shows an estimate should also show how many similar',
        '  cars the data holds (the web application does).',
    ]
    if 'power' in data:
        missing = data['cars'] - data['power']
        lines.append(
            f'- **Powers that cannot be read**: {count(missing)} cars ({percent(missing, data["cars"])}) have no power in '
            'the text of their engine; the model uses the median for them.'
        )
    lines += [
        '- **Small groups are noisy**: a group of 20 cars gives an error that moves a lot from one scraping to the next. Do',
        '  not read more into them than they say.',
        '',
        '## Reproduce',
        '',
        '```bash',
        'python -m scripts.pipeline --all                     # scrape (about 10 minutes), clean, report, train',
        f'python -m scripts.model_card data/cars_clean.csv --scraped {scraped}   # this card',
        '```',
        '',
        f'Another scraping gives other cars and other numbers. Measured with Python {versions["python"]}, scikit-learn '
        f'{versions["scikit-learn"]}, pandas {versions["pandas"]} and numpy {versions["numpy"]}.',
        '',
    ]
    return '\n'.join(lines)


def installed_versions():
    return {
        'python': platform.python_version(),
        'scikit-learn': sklearn.__version__,
        'pandas': pd.__version__,
        'numpy': np.__version__,
    }


def date(text):
    """The date of the scraping, as YYYY-MM-DD (argparse type)."""
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a date like 2026-11-07") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Write the model card from the cleaned data (every number measured)')
    parser.add_argument('data', help='cleaned CSV file (made by scripts/datacleaning.py)')
    parser.add_argument(
        '--scraped', type=date, default=datetime.date.today().isoformat(), help='date of the scraping (default: today)'
    )
    parser.add_argument(
        '--output', default=DEFAULT_OUTPUT, help=f'where the card is written (default: {DEFAULT_OUTPUT})'
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    print(f'Measuring the models on {len(df)} cars (about a minute on the whole listing)...')
    text = write_card(measure(df), args.scraped)
    folder = os.path.dirname(args.output)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f'Model card written in {args.output}')
    return text


if __name__ == '__main__':
    run(main)
