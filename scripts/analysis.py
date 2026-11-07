"""Data analysis of the cleaned used-car data: statistics, charts and an HTML report.

It is the "data analysis interface" of the project: one command makes a folder
with the charts (PNG files) and a page ``report.html`` to open in a browser.

Usage (from the root of the project):

    python -m scripts.analysis data/cars_clean.csv --output reports/

The input is the file made by scripts/datacleaning.py (columns Nom, Carburant,
Boite de vitesse, Km, année, Prix, Marque). The charts use only matplotlib.
"""

import argparse
import html
import os

import pandas as pd
from matplotlib.figure import Figure

from scripts.cli import UserError, run

# Brands (or models) with fewer cars are left out of the averages: 2 cars say nothing
MIN_CARS = 5

# Colours of the fuels in the charts (the others are grey)
FUEL_COLORS = {
    'Essence': '#e69f00',
    'Diesel': '#555555',
    'Électrique': '#009e73',
    'Hybride': '#0072b2',
    'Hybride rechargeable': '#56b4e9',
    'GPL': '#cc79a7',
}
OTHER_COLOR = '#999999'
MAIN_COLOR = '#0072b2'


def euros(value):
    """12345.6 -> '12 346 €' (spaces between the thousands, as in French)."""
    return f'{value:,.0f}'.replace(',', ' ') + ' €'


# ---------------------------------------------------------------- statistics


def summary(df):
    """
    Main figures of a cleaned DataFrame.

    Returns:
    - A dictionary: number of cars, price (mean, median, min, max), mileage and year
      (median, min, max)
    """
    return {
        'cars': int(len(df)),
        'price_mean': float(df['Prix'].mean()),
        'price_median': float(df['Prix'].median()),
        'price_min': int(df['Prix'].min()),
        'price_max': int(df['Prix'].max()),
        'km_median': float(df['Km'].median()),
        'km_max': int(df['Km'].max()),
        'year_median': float(df['année'].median()),
        'year_min': int(df['année'].min()),
        'year_max': int(df['année'].max()),
    }


def price_by(df, column, min_cars=MIN_CARS):
    """
    Number of cars, median price and median mileage for each value of a column.

    Values with fewer than min_cars cars are left out. The result is sorted by median price
    (the most expensive first).
    """
    grouped = df.groupby(column).agg(cars=('Prix', 'size'), median_price=('Prix', 'median'), median_km=('Km', 'median'))
    grouped = grouped[grouped['cars'] >= min_cars]
    return grouped.sort_values('median_price', ascending=False).reset_index()


def price_by_year(df, min_cars=MIN_CARS):
    """Median price for each year (sorted by year), years with fewer than min_cars cars are left out."""
    grouped = df.groupby('année').agg(cars=('Prix', 'size'), median_price=('Prix', 'median'))
    return grouped[grouped['cars'] >= min_cars].sort_index().reset_index()


def correlations(df):
    """
    Correlation of the price with the mileage and the year (between -1 and 1), and with the power if present.

    A correlation near -1 or 1 means that the two numbers move together; near 0, that they are not linked.
    """
    columns = [c for c in ('Km', 'année', 'Puissance') if c in df.columns]
    return df[['Prix'] + columns].corr()['Prix'].drop('Prix').round(2)


# ---------------------------------------------------------------- charts


def _save(figure, filename):
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)
    figure.tight_layout()
    figure.savefig(filename, dpi=110)


def plot_price_distribution(df, filename):
    """Histogram of the prices."""
    figure = Figure(figsize=(7, 4))
    axes = figure.subplots()
    axes.hist(df['Prix'], bins=30, color=MAIN_COLOR, edgecolor='white')
    axes.axvline(df['Prix'].median(), color='black', linestyle='--', label=f'median: {euros(df["Prix"].median())}')
    axes.set_title('How much do the cars cost?')
    axes.set_xlabel('Price (€)')
    axes.set_ylabel('Number of cars')
    axes.legend()
    _save(figure, filename)


def plot_price_vs_km(df, filename):
    """Price against mileage, one colour per fuel."""
    figure = Figure(figsize=(7, 4.5))
    axes = figure.subplots()
    for fuel, cars in df.groupby('Carburant'):
        axes.scatter(cars['Km'], cars['Prix'], s=14, alpha=0.6, label=fuel, color=FUEL_COLORS.get(fuel, OTHER_COLOR))
    axes.set_title('The more kilometres, the cheaper')
    axes.set_xlabel('Mileage (km)')
    axes.set_ylabel('Price (€)')
    axes.legend(title='Fuel')
    _save(figure, filename)


def plot_price_by_year(df, filename):
    """Median price of each year of registration."""
    data = price_by_year(df)
    figure = Figure(figsize=(7, 4))
    axes = figure.subplots()
    axes.bar(data['année'].astype(str), data['median_price'], color=MAIN_COLOR)
    axes.set_title(f'Median price by year (years with at least {MIN_CARS} cars)')
    axes.set_xlabel('Year')
    axes.set_ylabel('Median price (€)')
    axes.tick_params(axis='x', rotation=60)
    _save(figure, filename)


def plot_price_by_brand(df, filename, top=15):
    """Median price of the brands that have enough cars (the top most expensive)."""
    data = price_by(df, 'Marque').head(top).iloc[::-1]  # reversed: the most expensive is at the top
    figure = Figure(figsize=(7, 0.35 * max(len(data), 4) + 1.5))
    axes = figure.subplots()
    axes.barh(data['Marque'], data['median_price'], color=MAIN_COLOR)
    axes.set_title(f'Median price by brand (brands with at least {MIN_CARS} cars)')
    axes.set_xlabel('Median price (€)')
    _save(figure, filename)


def plot_fuel_share(df, filename):
    """Number of cars by fuel."""
    counts = df['Carburant'].value_counts()
    figure = Figure(figsize=(7, 3.5))
    axes = figure.subplots()
    axes.bar(counts.index, counts.values, color=[FUEL_COLORS.get(f, OTHER_COLOR) for f in counts.index])
    axes.set_title('Cars by fuel')
    axes.set_ylabel('Number of cars')
    _save(figure, filename)


CHARTS = [
    ('price_distribution.png', 'Prices', plot_price_distribution),
    ('price_vs_km.png', 'Price and mileage', plot_price_vs_km),
    ('price_by_year.png', 'Price and year', plot_price_by_year),
    ('price_by_brand.png', 'Price and brand', plot_price_by_brand),
    ('fuel_share.png', 'Fuels', plot_fuel_share),
]


# ---------------------------------------------------------------- report


def _table(df, formats):
    """A small HTML table (the values are escaped)."""
    head = ''.join(f'<th>{html.escape(str(column))}</th>' for column in df.columns)
    rows = ''
    for _, row in df.iterrows():
        cells = ''.join(f'<td>{html.escape(formats.get(c, str)(row[c]))}</td>' for c in df.columns)
        rows += f'<tr>{cells}</tr>'
    return f'<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'


def build_report_html(df):
    """The HTML page of the report (the charts are the PNG files next to it)."""
    stats = summary(df)
    facts = [
        ('Cars', f'{stats["cars"]}'),
        ('Median price', euros(stats['price_median'])),
        ('Mean price', euros(stats['price_mean'])),
        ('Cheapest / most expensive', f'{euros(stats["price_min"])} / {euros(stats["price_max"])}'),
        ('Median mileage', f'{stats["km_median"]:,.0f} km'.replace(',', ' ')),
        ('Years', f'{stats["year_min"]} to {stats["year_max"]}'),
    ]
    facts_html = ''.join(f'<li><strong>{html.escape(k)}</strong> {html.escape(v)}</li>' for k, v in facts)
    corr = correlations(df)
    corr_html = ''.join(f'<li>{html.escape(str(name))}: {value}</li>' for name, value in corr.items())
    brands = price_by(df, 'Marque').head(10)
    brands_html = _table(brands, {'median_price': euros, 'median_km': lambda v: f'{v:,.0f} km'.replace(',', ' ')})
    charts_html = ''.join(
        f'<section><h2>{html.escape(title)}</h2><img src="{name}" alt="{html.escape(title)}"></section>'
        for name, title, _ in CHARTS
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Used cars: analysis</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  img {{ max-width: 100%; height: auto; }}
  ul.facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .5rem; padding: 0; list-style: none; }}
  ul.facts li {{ background: #f1f5f9; border-radius: 6px; padding: .6rem .8rem; }}
  ul.facts strong {{ display: block; font-size: .8rem; color: #555; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: .3rem .8rem; text-align: left; }}
</style>
</head>
<body>
<h1>Used cars: analysis</h1>
<ul class="facts">{facts_html}</ul>
<h2>What is linked to the price?</h2>
<p>Correlation with the price (-1 to 1; near 0 means no link):</p>
<ul>{corr_html}</ul>
<h2>Brands</h2>
{brands_html}
{charts_html}
</body>
</html>
"""


def make_report(df, folder):
    """
    Write the charts (PNG) and report.html in a folder (created if needed).

    Returns:
    - The path of report.html
    """
    if df.empty:
        raise UserError('There is no car to analyse')
    os.makedirs(folder, exist_ok=True)
    for name, _, plot in CHARTS:
        plot(df, os.path.join(folder, name))
    report = os.path.join(folder, 'report.html')
    with open(report, 'w', encoding='utf-8') as file:
        file.write(build_report_html(df))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Statistics and charts of the cleaned used-car data')
    parser.add_argument('data', help='cleaned CSV file (made by scripts/datacleaning.py)')
    parser.add_argument('--output', default='reports', help='folder for the charts and report.html (default: reports)')
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    if df.empty:
        raise UserError(f'There is no car in {args.data}')
    stats = summary(df)
    print(
        f'{stats["cars"]} cars, median price {euros(stats["price_median"])}, '
        f'years {stats["year_min"]} to {stats["year_max"]}'
    )
    print('Correlation with the price:', correlations(df).to_dict())
    report = make_report(df, args.output)
    print(f'Report: {report} (open it in a browser)')
    return report


if __name__ == '__main__':
    run(main)
