# Model card: used-car price estimate

What the price model is, what it was trained on, how wrong it is and where, and what it must not be used for.
The figures below were **measured on 2026-11-07** on the data scraped that day; the commands at the end reproduce them.

## Summary

The model estimates the **asking price of a used car** from its brand and model, fuel, gearbox, mileage, year and (when it
is written in the listing) power and, for an electric car, battery capacity. It was trained on the **2693 cars of the whole aramisauto.com used-car listing**.
Measured on cars it had not seen (5-fold cross-validation), it is wrong by **1 390 € on average, 7.0 % of the
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
| Battery | the capacity in kWh, found in the text of the engine for 210 of the 271 electric cars (`50 kWh - 136`); two typos of plug-in hybrids (`180 kWh`, `220 kWh`) are emptied |

The CSV files are **not** in the repository (`.gitignore`). No personal data is collected: a listing has a number, a model and a price.
Read the terms of use of the website before publishing the data.

## Model

- **Features**: brand, model, fuel, gearbox, mileage, year, power, battery capacity (empty for the cars that are not
  electric: the model is told that it is missing). The condition of the car, its options and its colour
  are not in the listing page, so they are not used.
- **Target**: the asking price. The linear model and the boosting learn the **logarithm** of the price, so that a mistake of
  10 % counts the same on a cheap car and on an expensive one.
- **Choice**: three models are compared on the same 5 parts and the best is kept (`python -m scripts.models`):

| Model | Mean error | % of the price | R² |
|---|---:|---:|---:|
| **boosting** (gradient boosting, kept) | **1 389 €** | 7.0 % | 0.89 |
| ridge (linear) | 1 496 € | 7.5 % | 0.86 |
| random forest | 1 614 € | 8.3 % | 0.85 |
| always the average price | 5 193 € | 30.8 % | 0.00 |

The boosting is told which columns are labels: brand, fuel and gearbox are categories, and the model (287 names, more
than the 255 categories it accepts) is replaced by its mean price. Before, it read them as numbers in alphabetical order
(Audi=1 < BMW=2...) and was wrong by 1 436 €.

- **What matters most** (how much the error grows when a column is shuffled): year 1 264 €, power 1 246 €, brand 1 157 €,
  mileage 1 094 €, model 1 085 €, fuel 413 €, gearbox 370 €, battery 24 € (only the electric cars have one; on them, it
  lowers the error from 1 799 € to 1 720 € on average over 4 cuts of the data, but not on every cut).

## Where the model is wrong

Cross-validation, the cars cut in groups (`python -m scripts.evaluate`). **Bias** is the mean of (estimate - real price):
negative means the model estimates too low. Groups of fewer than 5 cars are left out.

**By how well the data knows the model**, the clearest weakness:

| Model | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| rare model (under 5 cars) | 339 | 2 857 € | 12.8 % | -638 € |
| common model (5+ cars) | 2 354 | 1 178 € | 6.2 % | -60 € |

A model with fewer than 5 cars in the data is estimated with an error **2.4 times larger**, and too low by 638 €
on average: the model has hardly seen it, so it can only lean on the brand, the year and the mileage. 339 cars of the
data are in this case.

**By price**: the model is pulled towards the middle.

| Price | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| cheapest quarter | 674 | 913 € | 8.0 % | +375 € |
| second quarter | 673 | 986 € | 5.9 % | +120 € |
| third quarter | 673 | 1 273 € | 6.2 % | -41 € |
| most expensive quarter | 673 | 2 386 € | 8.0 % | -983 € |

The most expensive quarter is estimated too low by 983 € on average, the cheapest quarter too high. The worst single
errors are premium cars (a Volvo EX90 at 69 999 € estimated near 43 300 €, a Mercedes Classe V at 40 999 € near 21 800 €).
A Citroën AMI at 7 999 € is estimated near 22 900 €: a licence-free quadricycle of 8 hp, below the 30 hp the cleaning
accepts, so its power is unknown and the model treats it like an ordinary electric car.

**By fuel**:

| Fuel | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| Diesel | 461 | 1 724 € | 9.0 % | -255 € |
| Électrique | 271 | 1 873 € | 8.1 % | +22 € |
| Hybride rechargeable | 196 | 1 874 € | 6.7 % | -469 € |
| GPL | 19 | 1 018 € | 6.4 % | -275 € |
| Essence | 1 069 | 980 € | 6.4 % | -60 € |
| Hybride | 675 | 1 484 € | 6.3 % | -119 € |

**By brand** (the 10 brands with the most cars, and the 5 worst):

| Brand | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| Volvo | 23 | 5 304 € | 15.7 % | -2 538 € |
| Mercedes | 38 | 3 816 € | 13.5 % | -905 € |
| BMW | 50 | 3 067 € | 11.4 % | -270 € |
| Mazda | 9 | 1 748 € | 9.2 % | -172 € |
| Honda | 6 | 1 858 € | 8.9 % | -1 215 € |
| Toyota | 82 | 2 073 € | 8.9 % | -187 € |
| Volkswagen | 170 | 1 890 € | 8.5 % | -205 € |
| Audi | 91 | 2 097 € | 7.8 % | -275 € |
| DS | 65 | 1 452 € | 7.5 % | +267 € |
| Renault | 365 | 1 219 € | 6.9 % | -156 € |
| Peugeot | 660 | 1 086 € | 6.4 % | -86 € |
| Fiat | 70 | 891 € | 6.2 % | +89 € |
| Dacia | 81 | 1 028 € | 6.2 % | -94 € |
| Citroën | 324 | 921 € | 5.9 % | +47 € |
| Opel | 145 | 893 € | 5.6 % | +93 € |

**By age and by mileage** (no strong pattern, between 5 % and 10 %; the oldest cars and those with the most kilometres are
the least well estimated):

| Age | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| 2026 (newest) | 125 | 1 840 € | 6.3 % | -330 € |
| 2025 | 525 | 1 329 € | 5.2 % | -224 € |
| 2023 to 2024 | 833 | 1 430 € | 6.6 % | -117 € |
| 2020 to 2022 | 775 | 1 391 € | 7.7 % | -73 € |
| before 2020 | 435 | 1 253 € | 8.9 % | -100 € |

| Mileage | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| under 5 000 km | 233 | 1 315 € | 5.2 % | -199 € |
| 5 000 to 30 000 km | 876 | 1 376 € | 6.2 % | -137 € |
| 30 000 to 80 000 km | 1 033 | 1 428 € | 7.2 % | -117 € |
| 80 000 km or more | 551 | 1 370 € | 8.7 % | -125 € |

## Limits and risks

- **Asking price, not sale price.** The data is what the dealer asks, not what a buyer pays.
- **One seller, one day.** All the cars come from one reconditioned-car dealer on 2026-11-07. Other sellers or other months
  can have other prices; the model does not know that prices move with time.
- **No condition, options or colour.** Two cars of the same model, year and mileage can have different prices for reasons
  the model cannot see.
- **Rare models and premium cars** are the weak points (tables above). A tool that shows an estimate should also show how
  many similar cars the data holds (the web application does).
- **Two wrong powers remain**: a BMW written `220 170 DKG7` is read as 220 hp instead of 170, and an Audi e-tron written
  `55 quattro 408` as 55 hp instead of 408 (one car each; the e-tron, at 33 599 €, is estimated near 16 800 €). Audi and hybrid cars were
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
