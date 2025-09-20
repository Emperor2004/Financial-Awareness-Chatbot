# scripts/scrape_rbi.py

import os
import time
import requests
from bs4 import BeautifulSoup
import re

# --- Configuration ---
BASE_URL = "https://www.rbi.org.in"
STARTING_URL = "https://www.rbi.org.in/commonman/English/scripts/notification.aspx"
SAVE_DIR = "data/rbi"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- Helper Functions ---
def create_save_directory():
    """Creates the directory to save files if it doesn't exist."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

def make_filename_safe(url):
    """Creates a safe filename from a URL, focusing on the ID."""
    # RBI URLs use an 'Id' parameter which is great for unique filenames
    try:
        url_id = url.split('Id=')[1]
        return f"rbi_notification_{url_id}.txt"
    except IndexError:
        # Fallback for URLs without an Id
        slug = url.split('/')[-1]
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', slug) + ".txt"

# --- Main Scraping Logic ---
def scrape_rbi_notifications():
    """Main function to scrape RBI notifications."""
    print("--- Starting RBI Scraper ---")
    create_save_directory()

    try:
        # 1. Fetch the main listing page
        print(f"Fetching index page: {STARTING_URL}")
        response = requests.get(STARTING_URL, headers=HEADERS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 2. Find all links to individual notifications
        # From inspection, links are in a table with class 'tablebg', within the second 'td' of each row
        notification_links = []
        # Find all rows in the main content table
        table_rows = soup.select("table.tablebg tr") 
        for row in table_rows:
            # The link is usually in the second cell (index 1)
            cells = row.find_all('td')
            if len(cells) > 1:
                link_tag = cells[1].find('a')
                if link_tag and 'href' in link_tag.attrs:
                    href = link_tag['href']
                    # Construct the full URL. The hrefs are relative like '/Scripts/...'
                    full_url = BASE_URL + href
                    notification_links.append(full_url)

        if not notification_links:
            print("No notification links found. The website structure might have changed.")
            return

        print(f"Found {len(notification_links)} notifications to process.")

        # 3. Loop through each link and scrape its content
        for url in notification_links:
            filename = make_filename_safe(url)
            save_path = os.path.join(SAVE_DIR, filename)

            if os.path.exists(save_path):
                print(f"Skipping already scraped: {url.split('/')[-1]}")
                continue

            print(f"Scraping: {url}")
            try:
                page_response = requests.get(url, headers=HEADERS)
                page_response.raise_for_status()
                
                page_soup = BeautifulSoup(page_response.content, 'html.parser')
                
                # Extract the main content from the page
                # The primary content is typically within a <td> with align="justify"
                content_td = page_soup.find('td', align='justify')
                
                if content_td:
                    text = content_td.get_text(separator='\n', strip=True)
                    clean_text = re.sub(r'\n\s*\n', '\n\n', text)
                    
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(clean_text)
                    print(f"  -> Saved to {save_path}")
                else:
                    print(f"  -> Could not find content for: {url}")

                time.sleep(1) # Be respectful to the server

            except requests.exceptions.RequestException as e:
                print(f"  -> Failed to fetch {url}: {e}")
            except Exception as e:
                print(f"  -> An error occurred while processing {url}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the main index page: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("--- RBI Scraper Finished ---")


# --- Run the Scraper ---
if __name__ == "__main__":
    scrape_rbi_notifications()