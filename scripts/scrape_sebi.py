# scripts/scrape_sebi.py

import os
import time
import requests
from bs4 import BeautifulSoup
import re
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# --- Configuration ---
BASE_URL = "https://www.sebi.gov.in"
STARTING_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&smid=0&ssid=6"
SAVE_DIR = "data/sebi"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- Helper Functions ---
def create_save_directory():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

def make_filename_safe(url):
    slug = url.split('/')[-1]
    return re.sub(r'[^a-zA-Z0.9_-]', '_', slug) + ".txt"

# --- Main Scraping Logic ---
def scrape_sebi_circulars():
    """Main function to scrape SEBI master circulars using undetected_chromedriver."""
    print("--- Starting SEBI Scraper (Undetected Version) ---")
    create_save_directory()
    driver = None

    try:
        # Step 1: Get the index page using requests (it's faster and not protected)
        print(f"Fetching index page: {STARTING_URL}")
        response = requests.get(STARTING_URL, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("Finding links in the circulars table...")
        circular_links = []
        main_table = soup.find('table', id='sample_1')
        if main_table:
            for link_tag in main_table.select('tbody tr a'):
                if 'href' in link_tag.attrs:
                    href = link_tag['href']
                    full_url = BASE_URL + href if not href.startswith('http') else href
                    circular_links.append(full_url)
        else:
            print("Could not find the main table. Website structure may have changed.")
            return

        circular_links = sorted(list(set(circular_links)))
        print(f"Found {len(circular_links)} circulars to process.")
        
        if not circular_links:
            return
        
        # Step 2: Set up the undetected driver to visit the protected pages
        print("Setting up Undetected WebDriver...")
        options = uc.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = uc.Chrome(options=options)

        # Step 3: Loop through each link and scrape with the undetected browser
        for url in circular_links:
            filename = make_filename_safe(url)
            save_path = os.path.join(SAVE_DIR, filename)

            if os.path.exists(save_path):
                print(f"Skipping already scraped: {url.split('/')[-1]}")
                continue

            print(f"Scraping: {url}")
            try:
                driver.get(url)
                
                # Wait up to 20 seconds for the element to appear
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
                    print(f"  -> Content div 'ck-content' not found after waiting.")

            except TimeoutException:
                 print(f"  -> Timed out waiting for content. This page may be structured differently or blocked.")
            except Exception as e:
                print(f"  -> An error occurred while processing {url}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the main index page: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if driver:
            print("Closing WebDriver.")
            driver.quit()

    print("--- SEBI Scraper Finished ---")


if __name__ == "__main__":
    scrape_sebi_circulars()