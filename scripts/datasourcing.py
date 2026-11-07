#data sourcing script
#create a .csv file with the data scraped from the website

#libraries
import os  # for creating the output folder
from urllib.parse import urljoin  # for building absolute URLs

import requests  # for making HTTP requests to web pages
from bs4 import BeautifulSoup  # for parsing HTML content
import pandas as pd

# The base URL of the website we're scraping
base_url = 'https://quotes.toscrape.com/'

# Maximum time (in seconds) to wait for a server response
REQUEST_TIMEOUT = 10


def fetch_quotes(page_url=base_url):
    """
    Fetch quotes and authors from a given page URL, following the "next" links.

    Parameters:
    - page_url: URL of the first page to scrape

    Returns:
    - A list of dictionaries, each containing an 'Author' and their 'Citation'
    """
    quotes_data = []  # Initialize an empty list to store quotes and authors

    while page_url:  # Loop over the pages instead of using recursion
        response = requests.get(page_url, timeout=REQUEST_TIMEOUT)  # Make a GET request to fetch the page content
        response.raise_for_status()  # Stop with a clear error if the request failed (404, 500...)

        soup = BeautifulSoup(response.content, 'html.parser')  # Parse the HTML content of the page
        quotes = soup.find_all('div', class_='quote')  # Find all quote blocks on the page

        for quote in quotes:  # Iterate over each quote block
            text_element = quote.find('span', class_='text')  # Find the element containing the quote text
            author_element = quote.find('small', class_='author')  # Find the element containing the author's name
            if text_element and author_element:  # Ensure both elements were found
                text = text_element.get_text(strip=True)  # Extract the text of the quote, stripping whitespace
                author = author_element.get_text(strip=True)  # Extract the author's name, stripping whitespace
                quotes_data.append({'Author': author, 'Citation': text})  # Add the quote and author to our list

        # Check for a link to the next page
        next_page = soup.find('li', class_='next')
        if next_page and next_page.find('a'):  # Ensure the next page link exists
            # Build the next URL from the current page, so it works for any starting URL
            page_url = urljoin(page_url, next_page.find('a')['href'])
        else:
            page_url = None  # No more pages: stop the loop

    return quotes_data  # Return the list of quotes and authors


def csv_saving(data, filename):
    """
    Save the scraped data to a CSV file (sorted by author).

    Parameters:
    - data: List of dictionaries containing the scraped data
    - filename: Name of the CSV file to save

    Returns:
    - The sorted DataFrame that was saved
    """
    df = pd.DataFrame(data)  # Convert the list of dictionaries into a pandas DataFrame
    if not df.empty:  # Sorting needs the 'Author' column, which is missing when nothing was scraped
        df = df.sort_values(by='Author').reset_index(drop=True)

    folder = os.path.dirname(filename)
    if folder:  # Create the output folder (e.g. 'data/') if it does not exist yet
        os.makedirs(folder, exist_ok=True)

    df.to_csv(filename, index=False)  # Save the DataFrame to a CSV file without row numbers
    return df  # Return the sorted DataFrame

# Start the scraping process from the first page of the site

if __name__ == '__main__':
    quotes_data = fetch_quotes(base_url)
    df_sorted = csv_saving(quotes_data, 'data/quotes_script.csv')
    print(df_sorted) # Display the sorted DataFrame
