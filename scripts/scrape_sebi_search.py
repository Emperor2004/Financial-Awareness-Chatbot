# scripts/scrape_sebi_search.py

import os
import time
import re
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# --- Configuration ---
# New starting URL is the search page
STARTING_URL = "https://www.sebi.gov.in/search.html?searchval=circular"
# We still need the base URL for constructing full links
BASE_URL = "https://www.sebi.gov.in"
SAVE_DIR = "data/sebi_search"

# --- Helper Functions ---
def create_save_directory():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

def make_filename_safe(url):
    slug = url.split('/')[-1]
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', slug) + ".txt"

# --- Main Scraping Logic ---
def scrape_sebi_via_search():
    """Main function to scrape SEBI circulars via the internal search function."""
    print("--- Starting SEBI Scraper (via Search) ---")
    create_save_directory()
    driver = None

    try:
        # Step 1: Set up the undetected driver
        print("Setting up Undetected WebDriver...")
        options = uc.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = uc.Chrome(options=options)

        # Step 2: Load the search results page and wait for results
        print(f"Fetching search results page: {STARTING_URL}")
        driver.get(STARTING_URL)
        
        print("Waiting for search results to load...")
        # Wait up to 30 seconds for the search results list to be present
        # From inspection, the results are in a <ul> with class 'search-results'
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.search-results li'))
        )
        print("Search results loaded.")

        # Step 3: Parse the page and extract all the links
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        circular_links = []
        search_results_list = soup.find('ul', class_='search-results')
        
        if search_results_list:
            for item in search_results_list.find_all('li'):
                link_tag = item.find('a')
                if link_tag and 'href' in link_tag.attrs:
                    href = link_tag['href']
                    full_url = BASE_URL + href if not href.startswith('http') else href
                    circular_links.append(full_url)
        else:
            print("Could not find the search results list. Structure may have changed.")
            return

        circular_links = sorted(list(set(circular_links)))
        print(f"Found {len(circular_links)} links from the search results.")

        if not circular_links:
            return

        # Step 4: Loop through each found link and scrape its content
        for url in circular_links:
            filename = make_filename_safe(url)
            save_path = os.path.join(SAVE_DIR, filename)

            if os.path.exists(save_path):
                print(f"Skipping already scraped: {url.split('/')[-1]}")
                continue

            print(f"Scraping content from: {url}")
            try:
                driver.get(url)
                
                # We still need to wait for the content div on the target page
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'ck-content'))
                )
                
                page_soup = BeautifulSoup(driver.page_source, 'html.parser')
                content_div = page_soup.find('div', class_='ck-content')
                
                if content_div:
                    text = content_div.get_text(separator='\n', strip=True)
                    clean_text = re.sub(r'\n\s*\n', '\n\n', text)
                    
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(clean_text)
                    print(f"  -> Saved to {save_path}")
                else:
                    # Some links from the search might be PDFs directly. This handles that case.
                    if '.pdf' in url.lower():
                         print(f"  -> Link is a PDF, manual download would be needed.")
                    else:
                         print(f"  -> Content div 'ck-content' not found on this page.")

            except TimeoutException:
                 print(f"  -> Timed out waiting for content. This page may be blocked or structured differently.")
            except Exception as e:
                print(f"  -> An error occurred while processing {url}: {e}")

    except TimeoutException:
        print("Timed out waiting for the initial search results to load.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if driver:
            print("Closing WebDriver.")
            driver.quit()

    print("--- SEBI Search Scraper Finished ---")

if __name__ == "__main__":
    scrape_sebi_via_search()