# Model card: used-car price estimate

What the price model is, what it was trained on, how wrong it is and where, and what it must not be used for.
The figures below were **measured on 2026-11-07** on the data scraped that day; the commands at the end reproduce them.

## Summary

The model estimates the **asking price of a used car** from its brand and model, fuel, gearbox, mileage, year and (when it
is written in the listing) power. It was trained on the **2693 cars of the whole aramisauto.com used-car listing**.
Measured on cars it had not seen (5-fold cross-validation), it is wrong by **1 461 € on average, 7.3 % of the
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
| Columns | `Id`, `Nom`, `Carburant`, `Boite de vitesse`, `Km`, `année`, `Prix`, `Motorisation`, `Finition`, `Puissance` |
| Power | found in the text of the engine for 95.1 % of the cars (for the others it is empty and the model uses the median) |

The CSV files are **not** in the repository (`.gitignore`). No personal data is collected: a listing has a number, a model and a price.
Read the terms of use of the website before publishing the data.

## Model

- **Features**: brand, model, fuel, gearbox, mileage, year, power. The condition of the car, its options and its colour
  are not in the listing page, so they are not used.
- **Target**: the asking price. The linear model and the boosting learn the **logarithm** of the price, so that a mistake of
  10 % counts the same on a cheap car and on an expensive one.
- **Choice**: three models are compared on the same 5 parts and the best is kept (`python -m scripts.models`):

| Model | Mean error | % of the price | R² |
|---|---:|---:|---:|
| **boosting** (gradient boosting, kept) | **1 461 €** | 7.3 % | 0.87 |
| ridge (linear) | 1 490 € | 7.5 % | 0.87 |
| random forest | 1 620 € | 8.3 % | 0.85 |
| always the average price | 5 193 € | 30.8 % | 0.00 |

The three real models are close: the choice of the algorithm matters less than the data.

- **What matters most** (how much the error grows when a column is shuffled): power 2 201 €, year 1 614 €, mileage 1 022 €,
  model 1 003 €, brand 937 €, fuel 460 €, gearbox 383 €.

## Where the model is wrong

Cross-validation, the cars cut in groups (`python -m scripts.evaluate`). **Bias** is the mean of (estimate - real price):
negative means the model estimates too low. Groups of fewer than 5 cars are left out.

**By how well the data knows the model**, the clearest weakness:

| Model | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| rare model (under 5 cars) | 339 | 3 315 € | 14.8 % | -1 312 € |
| common model (5+ cars) | 2 354 | 1 194 € | 6.2 % | -10 € |

A model with fewer than 5 cars in the data is estimated with an error **2.8 times larger**, and too low by 1 312 €
on average: the model has hardly seen it, so it can only lean on the brand, the year and the mileage. 339 cars of the
data are in this case.

**By price**: the model is pulled towards the middle.

| Price | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| cheapest quarter | 674 | 946 € | 8.3 % | +421 € |
| second quarter | 673 | 1 042 € | 6.2 % | +199 € |
| third quarter | 673 | 1 262 € | 6.2 % | +23 € |
| most expensive quarter | 673 | 2 593 € | 8.5 % | -1 339 € |

The most expensive quarter is estimated too low by 1 339 € on average, the cheapest quarter too high. The worst single
errors are premium cars (a Volvo EX90 at 69 999 € estimated near 31 600 €, an Audi e-tron at 33 599 € near 13 400 €).

**By fuel**:

| Fuel | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| Électrique | 271 | 2 352 € | 9.7 % | -504 € |
| Diesel | 461 | 1 735 € | 8.9 % | -322 € |
| Hybride rechargeable | 196 | 2 094 € | 7.7 % | -310 € |
| Essence | 1 069 | 1 002 € | 6.5 % | -15 € |
| GPL | 19 | 1 004 € | 6.5 % | -237 € |
| Hybride | 675 | 1 473 € | 6.4 % | -148 € |

**By brand** (the 10 brands with the most cars, and the 5 worst):

| Brand | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| Volvo | 23 | 6 040 € | 15.8 % | -3 808 € |
| Aion | 6 | 4 733 € | 14.4 % | -2 215 € |
| Mercedes | 38 | 3 881 € | 14.1 % | -1 625 € |
| Honda | 6 | 2 545 € | 12.5 % | -1 938 € |
| BMW | 50 | 3 178 € | 11.8 % | -1 004 € |
| Toyota | 82 | 2 196 € | 9.7 % | +194 € |
| Volkswagen | 170 | 2 008 € | 9.1 % | -189 € |
| Audi | 91 | 2 320 € | 8.3 % | -922 € |
| Fiat | 70 | 1 053 € | 7.8 % | +404 € |
| Renault | 365 | 1 285 € | 7.2 % | -21 € |
| Dacia | 81 | 1 179 € | 7.2 % | -47 € |
| DS | 65 | 1 506 € | 7.1 % | -343 € |
| Opel | 145 | 1 083 € | 6.7 % | +118 € |
| Citroën | 324 | 985 € | 6.3 % | +110 € |
| Peugeot | 660 | 917 € | 5.4 % | -35 € |

**By age and by mileage** (no strong pattern, between 5 % and 10 %; the oldest cars and those with the most kilometres are
the least well estimated):

| Age | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| 2026 (newest) | 125 | 2 153 € | 7.4 % | -426 € |
| 2025 | 525 | 1 406 € | 5.4 % | -244 € |
| 2023 to 2024 | 833 | 1 517 € | 7.1 % | -138 € |
| 2020 to 2022 | 775 | 1 402 € | 7.6 % | -181 € |
| before 2020 | 435 | 1 324 € | 9.4 % | -71 € |

| Mileage | Cars | Mean error | % of the price | Bias |
|---|---:|---:|---:|---:|
| under 5 000 km | 233 | 1 338 € | 5.3 % | -124 € |
| 5 000 to 30 000 km | 876 | 1 461 € | 6.5 % | -209 € |
| 30 000 to 80 000 km | 1 033 | 1 499 € | 7.5 % | -185 € |
| 80 000 km or more | 551 | 1 441 € | 9.1 % | -117 € |

## Limits and risks

- **Asking price, not sale price.** The data is what the dealer asks, not what a buyer pays.
- **One seller, one day.** All the cars come from one reconditioned-car dealer on 2026-11-07. Other sellers or other months
  can have other prices; the model does not know that prices move with time.
- **No condition, options or colour.** Two cars of the same model, year and mileage can have different prices for reasons
  the model cannot see.
- **Rare models and premium cars** are the weak points (tables above). A tool that shows an estimate should also show how
  many similar cars the data holds (the web application does).
- **Two wrong powers remain**: a BMW written `220 170 DKG7` is read as 220 hp instead of 170, and an Audi e-tron written
  `55 quattro 408` as 55 hp instead of 408 (one car each; the e-tron is the 33 599 € car above). Audi and hybrid cars were
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
