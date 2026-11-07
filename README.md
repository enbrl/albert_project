# Scraping Automobile

## Overview
This project is designed to automate the extraction of used car listings from [aramisauto.com](https://www.aramisauto.com/) and to utilize machine learning for estimating car prices based on the gathered data. The primary objectives are to streamline the process of data collection and to deploy a predictive model that offers accurate price estimations for users looking to buy or sell used cars.

## Features
- **Data Scraping**: Automated scraping tool to collect detailed information on used car listings, such as model, year, price, mileage, and features. This data serves as the foundation for the predictive model.
  
- **Machine Learning Model**: Implementation of a regression model to estimate car prices based on historical data. This model is trained with features extracted from the scraped data, adjusted for accuracy and reliability over time.
  
- **Data Analysis Interface**: Provides tools for data visualization and analysis, helping users understand the factors that influence car prices and how they vary over different segments.

## Project status
All the parts of the project work:

| Part | Where | Status |
|------|-------|--------|
| Scraping of car listings (aramisauto) | `scripts/aramis_scraper.py` | Working: `--all` scrapes the whole listing (about 2 700 cars in 10 minutes), saves after every page, retries on errors, 5 s between requests as asked by robots.txt |
| Scraping example (quotes) | `scripts/datasourcing.py`, `scripts/main.py` | Working, used as a template |
| Data cleaning | `scripts/datacleaning.py` | Working: harmonises the numbers, the fuels, the gearboxes and the brands (`Alfa Romeo`, `Citroën`...), removes the impossible values and the duplicates, and prints what it removed |
| Price model | `scripts/models.py` | Working: compares 3 models and keeps the best. Mean error of about 1 500 € (7 % of the price) on the 2 693 cars of the whole listing, scraped in November 2026. The versions of the libraries are saved next to the model (`--info`) |
| Data analysis interface | `scripts/analysis.py`, `scripts/predict.py`, `app.py` | Working: an HTML report with charts, a command and a web application (Streamlit) that estimate the price of a car |
| Everything in one command | `scripts/pipeline.py` | Working: scrape, clean, report and train with `python -m scripts.pipeline` |
| Check of the website | `tests/live/`, `.github/workflows/site-check.yml` | Every Monday, GitHub checks that the scraper still reads the website |

## Project structure
```
.
├── data/              # scraped data (CSV files are not versioned)
├── notebooks/         # demo_pipeline.ipynb (the whole project on a sample) + old prototypes
├── app.py             # web application (Streamlit) to estimate a price
├── scripts/
│   ├── pipeline.py       # the whole project in one command
│   ├── aramis_scraper.py # scraping of the used-car listings (aramisauto)
│   ├── datacleaning.py   # data cleaning
│   ├── models.py         # price model: comparison, training, evaluation, prediction
│   ├── analysis.py       # statistics, charts and HTML report
│   ├── predict.py        # estimate the price of one car (command line)
│   ├── estimator.py      # what the web application needs, without Streamlit
│   ├── cli.py            # clear error messages shared by the command lines
│   ├── datasourcing.py   # scraping example (quotes) + saving to CSV
│   └── main.py           # command line of the scraping example
├── tests/             # automatic tests (pytest); tests/live uses the real website
├── .github/workflows/ # tests on every pull request (old and new versions), web application, weekly check of the website
├── .pre-commit-config.yaml, ruff.toml  # automatic checks and formatting of the code, cleaning of the notebooks
├── pytest.ini
├── requirements.txt, requirements-app.txt, constraints-min.txt
├── LICENSE.md
└── README.md
```

## Installation
You need Python 3.9 or later. The tests run on GitHub with Python 3.9 (with the oldest packages allowed by `requirements.txt`), 3.10 and 3.13 (with the newest).

```bash
git clone https://github.com/enbrl/albert_project.git
cd albert_project

python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The web application needs Streamlit, which is not in `requirements.txt`: `pip install -r requirements-app.txt`.

## Usage
Run the scripts from the root of the project.

```bash
python -m scripts.main
```

The script scrapes every page of `https://quotes.toscrape.com/` and saves the result, sorted by author, in `data/quotes_script.csv`. It never asks a question, so it can be run in a script. The options are:

```bash
python -m scripts.main --url https://quotes.toscrape.com/page/5/ --output data/page5.csv
python -m scripts.main --help
```

To get the car data, a report and a price model, run everything with one command:

```bash
python -m scripts.pipeline                  # the first 5 pages (120 cars), a quick try
python -m scripts.pipeline --all            # the whole listing (about 10 minutes)
python -m scripts.pipeline --skip-scrape    # start again from the file already scraped
```

The pipeline prints the step it is on. If a step fails, it stops, says which one and what went wrong, and keeps the files of the steps before. Or run the steps one after the other:

```bash
python -m scripts.aramis_scraper --all --output data/cars_raw.csv        # scrape the whole listing (about 10 minutes, 5 s between pages)
python -m scripts.datacleaning data/cars_raw.csv -o data/cars_clean.csv  # clean
python -m scripts.analysis data/cars_clean.csv --output reports/         # charts and reports/report.html (open it in a browser)
python -m scripts.models data/cars_clean.csv                             # compare the models, train and save the best one
python -m scripts.predict --name "Peugeot 208" --fuel Essence --gearbox Manuelle --km 60000 --year 2019 --power 100
```

Useful options:
- `aramis_scraper --pages 5` scrapes only the first 5 pages (24 cars per page). The CSV is saved after every page; if the scraping stops, the message gives the command to continue (`--start-page N --append`).
- Several raw files can be given to `datacleaning` (for example the CSV files of the notebooks): they are merged and the duplicates are removed.
- `models --compare` only shows the comparison of the models, `models --kind ridge` trains a chosen one.
- `datacleaning` prints how many rows it removed and why (no name or price, impossible values, same listing seen twice, identical rows).
- `predict --power` is optional and used only if the model was trained with the power.
- A saved model (`models/price_model.joblib`) can only be read with the version of scikit-learn that made it. The versions are saved next to it in `models/price_model.json`: `python -m scripts.models --info models/price_model.joblib` shows them, and `predict` warns if they changed. After an update, train the model again.
- A wrong input gives one line `Error: ...` and the exit code 1 (for example a file that does not exist), not a page of technical error.

To estimate a price in a browser (after training the model):

```bash
pip install -r requirements-app.txt
streamlit run app.py
```

For a quick try on a small sample (about 20 seconds), open `notebooks/demo_pipeline.ipynb`.

To open the notebooks:

```bash
jupyter lab
```

## Dataset
Google Drive : https://drive.google.com/drive/folders/1hYUp5XBxwYOfoLZZuGwi5PK2o5-y4Zo2?usp=sharing

Download the files in the `data/` folder. They are not stored in the repository (see `.gitignore`).

## Tests
```bash
python -m pytest              # the tests, without the network
python -m pytest -m live      # check the real website (3 requests)
```
The first command also runs on GitHub (Actions) for every pull request and every push on `main`. The second one runs every Monday.

## Development
Install the automatic checks once:

```bash
pip install pre-commit
pre-commit install
```
Before each commit they check and format the code (`ruff`), remove the results saved in the notebooks (`nbstripout`) and check the files. GitHub runs the same checks on every pull request: code that is not formatted fails it. To format by hand: `pre-commit run --all-files`.

The tests run on GitHub with the oldest packages (`constraints-min.txt`) and the newest ones. If you need an older version of a package than the lower bounds of `requirements.txt`, test it first.

## Scraping rules
- The scraper follows the `robots.txt` of aramisauto.com: only the public listing pages, one request every 5 seconds (`Crawl-delay`), a timeout and a `User-Agent` that names the project. It never goes faster than 5 seconds.
- autoscout24.fr is **not** scraped: its `robots.txt` forbids the listing pages.
- The scraped CSV files are not stored in the repository. Read the terms of use of the website before publishing the data.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact
For any inquiries or help with the setup, please [email us](mailto:cpietri@albertschool.com).
