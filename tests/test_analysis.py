"""Tests of scripts/analysis.py, on made-up cars."""

import numpy as np
import pandas as pd
import pytest

from scripts import analysis
from scripts.analysis import (
    CHARTS,
    build_report_html,
    correlations,
    euros,
    main,
    make_report,
    price_by,
    price_by_year,
    summary,
)

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def make_cars(n=120, seed=1):
    rng = np.random.default_rng(seed)
    names = ['Peugeot 208', 'Peugeot 3008', 'Renault Clio', 'Tesla Model 3', 'Opel Astra']
    base = {
        'Peugeot 208': 9000,
        'Peugeot 3008': 18000,
        'Renault Clio': 8500,
        'Tesla Model 3': 30000,
        'Opel Astra': 11000,
    }
    rows = []
    for _ in range(n):
        name = str(rng.choice(names))
        year = int(rng.integers(2014, 2026))
        km = int(rng.integers(5_000, 150_000))
        rows.append(
            {
                'Nom': name,
                'Marque': name.split()[0],
                'Carburant': 'Électrique' if name.startswith('Tesla') else str(rng.choice(['Essence', 'Diesel'])),
                'Boite de vitesse': 'Manuelle',
                'Km': km,
                'année': year,
                'Prix': int(base[name] - 0.05 * km - 500 * (2026 - year) + 6000),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def cars():
    return make_cars()


def test_euros_formats_like_in_french():
    assert euros(12345.6) == '12 346 €'
    assert euros(999) == '999 €'
    assert euros(1234567) == '1 234 567 €'


def test_summary(cars):
    stats = summary(cars)
    assert stats['cars'] == 120
    assert stats['price_min'] == cars['Prix'].min()
    assert stats['price_max'] == cars['Prix'].max()
    assert stats['price_median'] == cars['Prix'].median()
    assert stats['year_min'] >= 2014 and stats['year_max'] <= 2025
    assert all(isinstance(stats[key], (int, float)) for key in stats)


def test_price_by_brand_leaves_out_small_groups_and_sorts(cars):
    extra = pd.concat([cars, make_cars(2, seed=9).assign(Marque='Rare')], ignore_index=True)  # only 2 cars
    result = price_by(extra, 'Marque')
    assert 'Rare' not in set(result['Marque'])
    assert list(result['median_price']) == sorted(result['median_price'], reverse=True)
    assert result['cars'].min() >= analysis.MIN_CARS
    assert result.iloc[0]['Marque'] == 'Tesla'  # the most expensive brand of the made-up data


def test_price_by_year_is_sorted_by_year(cars):
    result = price_by_year(cars)
    assert list(result['année']) == sorted(result['année'])
    assert result['cars'].min() >= analysis.MIN_CARS


def test_correlations_the_price_goes_down_with_age_and_mileage(cars):
    result = correlations(cars)
    assert result['Km'] < 0  # more kilometres, cheaper
    assert result['année'] > 0  # newer, more expensive
    assert 'Puissance' not in result.index


def test_correlations_include_the_power_when_present(cars):
    with_power = cars.assign(Puissance=(cars['Prix'] / 100).astype(int))
    assert correlations(with_power)['Puissance'] > 0.9


def test_make_report_writes_every_chart_and_the_page(cars, tmp_path):
    folder = tmp_path / 'reports' / 'nested'  # does not exist yet
    report = make_report(cars, str(folder))
    assert report == str(folder / 'report.html')
    for name, _, _ in CHARTS:
        content = (folder / name).read_bytes()
        assert content.startswith(PNG_SIGNATURE) and len(content) > 5_000, name
    page = (folder / 'report.html').read_text(encoding='utf-8')
    for name, _, _ in CHARTS:
        assert f'src="{name}"' in page
    assert '120' in page  # number of cars


def test_report_html_escapes_the_values():
    cars = make_cars(30).assign(Marque='<script>alert(1)</script>')
    page = build_report_html(cars)
    assert '<script>alert(1)</script>' not in page
    assert '&lt;script&gt;' in page


def test_make_report_refuses_an_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match='no car'):
        make_report(make_cars(0), str(tmp_path))


def test_main_reads_the_csv_and_prints_the_summary(cars, tmp_path, capsys):
    csv = tmp_path / 'cars_clean.csv'
    cars.to_csv(csv, index=False)
    report = main([str(csv), '--output', str(tmp_path / 'out')])
    assert (tmp_path / 'out' / 'report.html').exists()
    assert report.endswith('report.html')
    output = capsys.readouterr().out
    assert '120 cars' in output and 'median price' in output and 'Correlation' in output
