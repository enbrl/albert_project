#main script is used to run the whole project and get the final output
#It is used to call the functions from other scripts and run the project
#
#run from the root of the project :
#    python -m scripts.main
#    python -m scripts.main --url https://quotes.toscrape.com/page/5/ --output data/page5.csv
import argparse

from scripts.cli import run
from scripts.datasourcing import base_url, csv_saving, fetch_quotes

DEFAULT_OUTPUT = 'data/quotes_script.csv'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scrape the quotes of a website and save them in a CSV file (sorted by author)')
    parser.add_argument('--url', default=base_url, help=f'URL of the first page to scrape (default: {base_url})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help=f'CSV file to write (default: {DEFAULT_OUTPUT})')
    args = parser.parse_args(argv)

    quotes_data = fetch_quotes(args.url)
    df_sorted = csv_saving(quotes_data, args.output)
    return df_sorted


if __name__ == '__main__':
    run(main, print_result=True)
