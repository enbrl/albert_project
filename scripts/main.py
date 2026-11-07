#main script is used to run the whole project and get the final output
#It is used to call the functions from other scripts and run the project
from scripts.datasourcing import base_url, fetch_quotes, csv_saving

def main():
    url = input(f"Enter the URL to scrape (press Enter for {base_url}): ").strip() or base_url
    quotes_data = fetch_quotes(url)
    df_sorted = csv_saving(quotes_data, 'data/quotes_script.csv')
    return df_sorted




if __name__ == '__main__':
    #run from the root of the project : python -m scripts.main
    #url to paste : https://quotes.toscrape.com/
    print(main())
