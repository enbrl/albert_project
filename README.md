# Scraping Automobile

## Overview
This project is designed to automate the extraction of used car listings from [aramisauto.com](https://www.aramisauto.com/) and to utilize machine learning for estimating car prices based on the gathered data. The primary objectives are to streamline the process of data collection and to deploy a predictive model that offers accurate price estimations for users looking to buy or sell used cars.

## Features
- **Data Scraping**: Automated scraping tool to collect detailed information on used car listings, such as model, year, price, mileage, and features. This data serves as the foundation for the predictive model.
  
- **Machine Learning Model**: Implementation of a regression model to estimate car prices based on historical data. This model is trained with features extracted from the scraped data, adjusted for accuracy and reliability over time.
  
- **Data Analysis Interface**: Provides tools for data visualization and analysis, helping users understand the factors that influence car prices and how they vary over different segments.

## Project status
Not everything described above is finished yet:

| Part | Where | Status |
|------|-------|--------|
| Scraping of car listings (aramisauto) | `scripts/aramis_scraper.py` | Working (`python -m scripts.aramis_scraper --pages 5`, 5 s between requests as asked by robots.txt) |
| Scraping example (quotes) | `scripts/datasourcing.py`, `scripts/main.py` | Working, used as a template |
| Data cleaning | `scripts/datacleaning.py` | Empty, to do |
| Price model | `scripts/models.py` | Empty, to do |
| Data analysis interface | - | To do |

## Project structure
```
.
├── data/              # scraped data (CSV files are not versioned)
├── notebooks/         # exploration and scraping prototypes
├── scripts/
│   ├── datasourcing.py   # scraping + saving to CSV
│   ├── datacleaning.py   # data cleaning (to do)
│   ├── models.py         # model training (to do)
│   └── main.py           # runs the whole pipeline
├── requirements.txt
└── README.md
```

## Installation
You need Python 3 (the scripts were tested with Python 3.13).

```bash
git clone https://github.com/enbrl/albert_project.git
cd albert_project

python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Run the scripts from the root of the project.

```bash
python -m scripts.main
```

The script asks for a URL (press Enter to use the default one, `https://quotes.toscrape.com/`), scrapes every page and saves the result, sorted by author, in `data/quotes_script.csv`.

To explore the notebooks:

```bash
jupyter lab
```

## Dataset
Google Drive : https://drive.google.com/drive/folders/1hYUp5XBxwYOfoLZZuGwi5PK2o5-y4Zo2?usp=sharing

Download the files in the `data/` folder. They are not stored in the repository (see `.gitignore`).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact
For any inquiries or help with the setup, please [email us](mailto:cpietri@albertschool.com).
