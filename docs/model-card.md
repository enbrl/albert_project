# Model card: used-car price estimate

What the price model is, what it was trained on, how wrong it is and where, and what it must not be used for.
The figures below were **measured on 2026-11-07** on the data scraped that day; the commands at the end reproduce them.

## Summary

The model estimates the **asking price of a used car** from its brand and model, fuel, gearbox, mileage, year and (when it
is written in the listing) power, trim and, for an electric car, battery capacity. It was trained on the **2693 cars of the whole aramisauto.com used-car listing**.
Measured on cars it had not seen (5-fold cross-validation), it is wrong by **1 298 € on average, 6.6 % of the
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
| Power | found in the text of the engine for 95.1 % of the cars (for the others it is empty and the model uses the median) |
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
| **boosting** (gradient boosting, kept) | **1 298 €** | 6.6 % | 0.90 |
| ridge (linear) | 1 337 € | 6.7 % | 0.88 |
| random forest | 1 572 € | 8.1 % | 0.85 |
| always the average price | 5 193 € | 30.8 % | 0.00 |

The boosting is told which columns are labels: brand, fuel and gearbox are categories, and the model (287 names, more
than the 255 categories it accepts) is replaced by its mean price. Before, it read them as numbers in alphabetical order
(Audi=1 < BMW=2...) and was wrong by 1 436 €; without the family of the trim, it is wrong by 1 389 €.

- **What matters most** (how much the error grows when a column is shuffled): year 1 267 €, model 1 102 €, power 1 085 €,
  brand 1 025 €, mileage 1 025 €, fuel 370 €, gearbox 341 €, family of the trim 227 €, battery 11 € (only the electric cars
  have one; on them, it lowers the error from 1 692 € to 1 661 € on average over 4 cuts of the data, but not on every cut).

## Where the model is wrong

Cross-validation, the cars cut in groups (`python -m scripts.evaluate`). **Bias** is the mean of (estimate - real price):
negative means the model estimates too low. Groups of fewer than 5 cars are left out.

**By how well the data knows the model**, the clearest weakness:

| Model | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| rare model (under 5 cars) | 339 | 2 888 € | 13.0 % | -777 € |
| common model (5+ cars) | 2 354 | 1 070 € | 5.7 % | -38 € |

A model with fewer than 5 cars in the data is estimated with an error **2.7 times larger**, and too low by 777 €
on average: the model has hardly seen it, so it can only lean on the brand, the year and the mileage. 339 cars of the
data are in this case.

**By price**: the model is pulled towards the middle.

| Price | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| cheapest quarter | 674 | 898 € | 7.9 % | +369 € |
| second quarter | 673 | 922 € | 5.6 % | +48 € |
| third quarter | 673 | 1 127 € | 5.5 % | +16 € |
| most expensive quarter | 673 | 2 247 € | 7.5 % | -958 € |

The most expensive quarter is estimated too low by 958 € on average, the cheapest quarter too high. The worst single
errors are premium cars (a Volvo EX90 at 69 999 € estimated near 43 700 €, a Mercedes Classe V at 40 999 € near 21 300 €).
A Citroën AMI at 7 999 € is estimated near 24 100 €: a licence-free quadricycle of 8 hp, below the 30 hp the cleaning
accepts, so its power is unknown and the model treats it like an ordinary electric car.

**By fuel**:

| Fuel | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| Diesel | 461 | 1 665 € | 8.7 % | -278 € |
| Électrique | 271 | 1 720 € | 7.5 % | -70 € |
| Hybride rechargeable | 196 | 1 937 € | 6.9 % | -397 € |
| Essence | 1 069 | 942 € | 6.2 % | -47 € |
| GPL | 19 | 929 € | 5.9 % | -189 € |
| Hybride | 675 | 1 268 € | 5.4 % | -105 € |

**By brand** (the 10 brands with the most cars, and the 5 worst):

| Brand | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| Volvo | 23 | 5 286 € | 15.8 % | -2 850 € |
| Mercedes | 38 | 3 487 € | 12.2 % | -1 109 € |
| BMW | 50 | 3 087 € | 11.2 % | -409 € |
| Mini | 22 | 1 831 € | 9.3 % | -541 € |
| Mazda | 9 | 1 733 € | 9.0 % | -294 € |
| Volkswagen | 170 | 1 841 € | 8.2 % | -262 € |
| Audi | 91 | 2 183 € | 8.0 % | -216 € |
| Toyota | 82 | 1 786 € | 7.8 % | -141 € |
| DS | 65 | 1 384 € | 7.1 % | +133 € |
| Fiat | 70 | 978 € | 6.9 % | +87 € |
| Renault | 365 | 1 011 € | 5.9 % | -104 € |
| Dacia | 81 | 945 € | 5.9 % | -89 € |
| Citroën | 324 | 865 € | 5.6 % | +40 € |
| Peugeot | 660 | 937 € | 5.6 % | -71 € |
| Opel | 145 | 906 € | 5.6 % | +94 € |

**By age and by mileage** (no strong pattern, between 5 % and 10 %; the oldest cars and those with the most kilometres are
the least well estimated):

| Age | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| 2026 (newest) | 125 | 1 718 € | 5.9 % | -269 € |
| 2025 | 525 | 1 155 € | 4.5 % | -243 € |
| 2023 to 2024 | 833 | 1 309 € | 6.1 % | -88 € |
| 2020 to 2022 | 775 | 1 340 € | 7.4 % | -100 € |
| before 2020 | 435 | 1 255 € | 9.0 % | -96 € |

| Mileage | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| under 5 000 km | 233 | 1 280 € | 5.1 % | -141 € |
| 5 000 to 30 000 km | 876 | 1 178 € | 5.3 % | -155 € |
| 30 000 to 80 000 km | 1 033 | 1 391 € | 7.1 % | -105 € |
| 80 000 km or more | 551 | 1 324 € | 8.4 % | -137 € |

## Limits and risks

- **Asking price, not sale price.** The data is what the dealer asks, not what a buyer pays.
- **One seller, one day.** All the cars come from one reconditioned-car dealer on 2026-11-07. Other sellers or other months
  can have other prices; the model does not know that prices move with time.
- **No condition, options or colour.** Two cars of the same model, year and mileage can have different prices for reasons
  the model cannot see.
- **Rare models and premium cars** are the weak points (tables above). A tool that shows an estimate should also show how
  many similar cars the data holds (the web application does).
- **Two wrong powers remain**: a BMW written `220 170 DKG7` is read as 220 hp instead of 170, and an Audi e-tron written
  `55 quattro 408` as 55 hp instead of 408 (one car each; the e-tron, at 33 599 €, is estimated near 16 700 €). Audi and hybrid cars were
  wrong before the correction of the power parser (71 cars).
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
