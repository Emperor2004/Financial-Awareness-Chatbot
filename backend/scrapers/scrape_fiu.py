# scripts/scrape_fiu.py

import os
import time
import requests
from bs4 import BeautifulSoup
import re
import fitz  # PyMuPDF

# --- Configuration ---
# Note: The base URL needs to be constructed carefully from the relative links
BASE_URL = "https://fiuindia.gov.in"
# STARTING_URL = "https://fiuindia.gov.in/files/Compliance_Orders/orders.html"
STARTING_URL = "https://fiuindia.gov.in/files/FAQs/faqs.html"
SAVE_DIR = "data/fiu"
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
    """Creates a safe filename from a URL, targeting the PDF name."""
    try:
        pdf_name = url.split('/')[-1]
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', pdf_name.replace('.pdf', ''))
        return f"{safe_name}.txt"
    except Exception:
        return f"fiu_{int(time.time())}.txt"

# --- Main Scraping Logic ---
def scrape_fiu_orders():
    """Main function to scrape FIU-India compliance orders (PDFs)."""
    print("--- Starting FIU-India Scraper ---")
    create_save_directory()

    try:
        # 1. Fetch the main listing page
        print(f"Fetching index page: {STARTING_URL}")
        response = requests.get(STARTING_URL, headers=HEADERS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 2. Find all links that point to PDF files
        pdf_links = []
        # The links on this page are relative (e.g., ../../pdfs/judgements/...),
        # so we need to construct the absolute URL.
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            if href.lower().endswith('.pdf'):
                # urljoin handles relative paths like '../../' correctly
                from urllib.parse import urljoin
                full_url = urljoin(STARTING_URL, href)
                pdf_links.append(full_url)

        pdf_links = sorted(list(set(pdf_links))) # Remove duplicates

        if not pdf_links:
            print("No PDF links found. The website structure might have changed.")
            return

        print(f"Found {len(pdf_links)} PDF orders to process.")

        # 3. Loop through each link, download PDF, and extract text
        for url in pdf_links:
            filename = make_filename_safe(url)
            save_path = os.path.join(SAVE_DIR, filename)

            if os.path.exists(save_path):
                print(f"Skipping already scraped: {url.split('/')[-1]}")
                continue

            print(f"Processing PDF: {url}")
            try:
                pdf_response = requests.get(url, headers=HEADERS)
                pdf_response.raise_for_status()
                
                with fitz.open(stream=pdf_response.content, filetype="pdf") as doc:
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                
                clean_text = re.sub(r'\s+', ' ', full_text).strip()
                
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                print(f"  -> Saved text to {save_path}")

                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"  -> Failed to download {url}: {e}")
            except Exception as e:
                print(f"  -> An error occurred while processing PDF {url}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the main index page: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("--- FIU-India Scraper Finished ---")


# --- Run the Scraper ---
if __name__ == "__main__":
    scrape_fiu_orders()