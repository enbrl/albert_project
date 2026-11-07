# Model card: used-car price estimate

What the price model is, what it was trained on, how wrong it is and where, and what it must not be used for.
The figures below were **measured on 2026-11-07** on the data scraped that day; the commands at the end reproduce them.

## Summary

The model estimates the **asking price of a used car** from its brand and model, fuel, gearbox, mileage, year and (when it
is written in the listing) power, trim and, for an electric car, battery capacity. It was trained on the **2693 cars of the whole aramisauto.com used-car listing**.
Measured on cars it had not seen (5-fold cross-validation), it is wrong by **1 256 € on average, 6.4 % of the
price**. Always answering the average price would be wrong by 5 193 € (30.8 %). It is a **learning project**, not a valuation tool.

## Intended use

- Showing the whole data-science chain on a real listing: scraping, cleaning, modelling, evaluation.
- Getting an order of magnitude for a **common** car of the listing (a Peugeot 208, a Renault Clio...), next to the real
  cars of the data that look like it (the web application shows the five closest).

**Not intended for**: buying or selling decisions, insurance, taxes or any valuation that somebody relies on; cars that
are not sold by this website; models that are rare in the data (see the limits); any use that treats the estimate as a
price somebody will pay.

## Data

| | |
|---|---|
| Source | the public used-car listing of aramisauto.com (`/voitures/occasion/`), a reconditioned-car dealer |
| Collected | 2026-11-07, all 113 pages, one request every 5 seconds (the `Crawl-delay` of its `robots.txt`) |
| Size | 2693 cars, 38 brands, 302 models; asking prices from 5 999 € to 69 999 € (median 18 499 €); years 2010 to 2026 |
| Fuel | petrol 1 069, hybrid 675, diesel 461, electric 271, plug-in hybrid 196, LPG 19, ethanol 2 |
| Columns | `Id`, `Nom`, `Carburant`, `Boite de vitesse`, `Km`, `année`, `Prix`, `Motorisation`, `Finition`, `Puissance`, `Batterie` |
| Power | found in the text of the engine for 97.8 % of the cars (for the others it is empty and the model uses the median) |
| Trim | written for 98.1 % of the cars (790 trims); the model uses its first word, the family (`Allure Pack` -> `allure`: 240 families) |
| Battery | the capacity in kWh, found in the text of the engine for 210 of the 271 electric cars (`50 kWh - 136`); two typos of plug-in hybrids (`180 kWh`, `220 kWh`) are emptied |

The CSV files are **not** in the repository (`.gitignore`). No personal data is collected: a listing has a number, a model and a price.
Read the terms of use of the website before publishing the data.

## Model

- **Features**: brand, model, fuel, gearbox, mileage, year, power, family of the trim, battery capacity (empty for the
  cars that are not electric: the model is told that it is missing). The condition of the car, its options and its colour
  are not in the listing page, so they are not used.
- **Target**: the asking price. The linear model and the boosting learn the **logarithm** of the price, so that a mistake of
  10 % counts the same on a cheap car and on an expensive one.
- **Choice**: three models are compared on the same 5 parts and the best is kept (`python -m scripts.models`):

| Model | Mean error | % of the price | R² |
|---|---:|---:|---:|
| **boosting** (gradient boosting, kept) | **1 256 €** | 6.4 % | 0.91 |
| ridge (linear) | 1 321 € | 6.6 % | 0.88 |
| random forest | 1 561 € | 8.0 % | 0.86 |
| always the average price | 5 193 € | 30.8 % | 0.00 |

The boosting is told which columns are labels: brand, fuel and gearbox are categories, and the model (287 names, more
than the 255 categories it accepts) is replaced by its mean price. Before, it read them as numbers in alphabetical order
(Audi=1 < BMW=2...), an order that means nothing for the price.

- **What matters most** (how much the error grows when a column is shuffled): power 1 385 €, year 1 351 €, mileage 1 079 €,
  model 1 063 €, brand 999 €, fuel 374 €, gearbox 307 €, family of the trim 227 €, battery 12 €. Only the electric cars
  have a battery, and on them it no longer changes the error on average over 4 cuts of the data (1 579 € with or without
  it): the power, now read for more cars, says the same thing.

## Where the model is wrong

Cross-validation, the cars cut in groups (`python -m scripts.evaluate`). **Bias** is the mean of (estimate - real price):
negative means the model estimates too low. Groups of fewer than 5 cars are left out.

**By how well the data knows the model**, the clearest weakness:

| Model | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| rare model (under 5 cars) | 339 | 2 697 € | 12.0 % | -750 € |
| common model (5+ cars) | 2 354 | 1 048 € | 5.6 % | -42 € |

A model with fewer than 5 cars in the data is estimated with an error **2.6 times larger**, and too low by 750 €
on average: the model has hardly seen it, so it can only lean on the brand, the year and the mileage. 339 cars of the
data are in this case.

**By price**: the model is pulled towards the middle.

| Price | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| cheapest quarter | 674 | 860 € | 7.5 % | +340 € |
| second quarter | 673 | 882 € | 5.3 % | +31 € |
| third quarter | 673 | 1 112 € | 5.4 % | +24 € |
| most expensive quarter | 673 | 2 170 € | 7.2 % | -921 € |

The most expensive quarter is estimated too low by 921 € on average, the cheapest quarter too high. The worst single
errors are premium cars (a Volvo EX90 at 69 999 € estimated near 41 100 €, a Mercedes Classe V at 40 999 € near 20 700 €,
a Volvo V90 at 34 499 € near 15 500 €). A Citroën AMI at 7 999 € is estimated near 18 500 €: a licence-free quadricycle
of 8 hp, the only one of the data, so the model has nothing like it to learn from.

**By fuel**:

| Fuel | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| Diesel | 461 | 1 620 € | 8.4 % | -264 € |
| Hybride rechargeable | 196 | 1 964 € | 7.0 % | -426 € |
| Électrique | 271 | 1 612 € | 6.8 % | -80 € |
| Essence | 1 069 | 921 € | 6.1 % | -53 € |
| GPL | 19 | 844 € | 5.4 % | -203 € |
| Hybride | 675 | 1 200 € | 5.1 % | -95 € |

**By brand** (the 10 brands with the most cars, and the 5 worst):

| Brand | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| Volvo | 23 | 5 490 € | 16.0 % | -3 172 € |
| Mercedes | 38 | 3 300 € | 11.3 % | -923 € |
| BMW | 50 | 2 720 € | 9.9 % | -529 € |
| Mazda | 9 | 1 719 € | 9.0 % | -334 € |
| Ford | 64 | 1 710 € | 8.7 % | -628 € |
| Volkswagen | 170 | 1 806 € | 7.9 % | -277 € |
| Audi | 91 | 2 072 € | 7.8 % | -167 € |
| DS | 65 | 1 413 € | 7.3 % | +91 € |
| Fiat | 70 | 835 € | 6.0 % | +178 € |
| Toyota | 82 | 1 347 € | 5.9 % | -44 € |
| Dacia | 81 | 906 € | 5.8 % | -36 € |
| Opel | 145 | 923 € | 5.8 % | +114 € |
| Renault | 365 | 980 € | 5.7 % | -105 € |
| Peugeot | 660 | 921 € | 5.5 % | -60 € |
| Citroën | 324 | 854 € | 5.4 % | +24 € |

**By age and by mileage** (no strong pattern, between 5 % and 10 %; the oldest cars and those with the most kilometres are
the least well estimated):

| Age | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| 2026 (newest) | 125 | 1 658 € | 5.6 % | -347 € |
| 2025 | 525 | 1 151 € | 4.4 % | -226 € |
| 2023 to 2024 | 833 | 1 276 € | 5.9 % | -89 € |
| 2020 to 2022 | 775 | 1 281 € | 7.1 % | -99 € |
| before 2020 | 435 | 1 183 € | 8.5 % | -93 € |

| Mileage | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| under 5 000 km | 233 | 1 223 € | 4.9 % | -163 € |
| 5 000 to 30 000 km | 876 | 1 156 € | 5.2 % | -141 € |
| 30 000 to 80 000 km | 1 033 | 1 346 € | 6.8 % | -100 € |
| 80 000 km or more | 551 | 1 258 € | 8.0 % | -161 € |

## Limits and risks

- **Asking price, not sale price.** The data is what the dealer asks, not what a buyer pays.
- **One seller, one day.** All the cars come from one reconditioned-car dealer on 2026-11-07. Other sellers or other months
  can have other prices; the model does not know that prices move with time.
- **No condition, options or colour.** Two cars of the same model, year and mileage can have different prices for reasons
  the model cannot see.
- **Rare models and premium cars** are the weak points (tables above). A tool that shows an estimate should also show how
  many similar cars the data holds (the web application does).
- **Powers that cannot be read**: 60 cars (2.2 %) have no power in the text of their engine (Tesla `75 kWh - RWD`, Aion
  `PREMIUM`...), and the Renault Zoé write an approximate one (`R110` for 108 hp) that is not read. The model uses the
  median for them. The Toyota hybrids (`116h`) had no power before the power parser read the 'h' (69 cars).
- **Small groups are noisy**: a group of 20 cars (LPG, Alfa Romeo...) gives an error that moves a lot from one scraping to
  the next. Do not read more into them than they say.

## Reproduce

```bash
python -m scripts.pipeline --all                  # scrape (about 10 minutes), clean, report, train
python -m scripts.evaluate data/cars_clean.csv    # the tables of "Where the model is wrong"
python -m scripts.models data/cars_clean.csv --compare
```

Another scraping gives other cars and other numbers: the figures of this card are those of **2026-11-07**. Measured with
Python 3.13, scikit-learn 1.9.1, pandas 3.1.1 and numpy 2.5.4 (`python -m scripts.models --info` shows the versions a
saved model was made with).
