# scripts/scrape_incometax.py

import os
import time
import requests
from bs4 import BeautifulSoup
import re
import fitz  # PyMuPDF

# --- Configuration ---
BASE_URL = "https://incometaxindia.gov.in"
STARTING_URL = "https://incometaxindia.gov.in/pages/communications/circulars.aspx"
SAVE_DIR = "data/incometax"
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
        # Replace .pdf with .txt and clean up
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', pdf_name.replace('.pdf', ''))
        return f"{safe_name}.txt"
    except Exception:
        # Fallback for unexpected URL formats
        return f"incometax_{int(time.time())}.txt"

# --- Main Scraping Logic ---
def scrape_incometax_circulars():
    """Main function to scrape Income Tax department circulars (PDFs)."""
    print("--- Starting Income Tax Department Scraper ---")
    create_save_directory()

    try:
        # 1. Fetch the main listing page
        print(f"Fetching index page: {STARTING_URL}")
        response = requests.get(STARTING_URL, headers=HEADERS)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 2. Find all links that point to PDF files
        pdf_links = []
        # From inspection, links are in a table, we specifically look for hrefs ending in .pdf
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            if href.lower().endswith('.pdf'):
                # Construct the full URL if the link is relative
                if not href.startswith('http'):
                    full_url = BASE_URL + href
                else:
                    full_url = href
                pdf_links.append(full_url)

        # Remove duplicate links that might be found on the page
        pdf_links = sorted(list(set(pdf_links)))

        if not pdf_links:
            print("No PDF links found. The website structure might have changed.")
            return

        print(f"Found {len(pdf_links)} PDF circulars to process.")

        # 3. Loop through each link, download the PDF, and extract its text
        for url in pdf_links:
            filename = make_filename_safe(url)
            save_path = os.path.join(SAVE_DIR, filename)

            if os.path.exists(save_path):
                print(f"Skipping already scraped: {url.split('/')[-1]}")
                continue

            print(f"Processing PDF: {url}")
            try:
                # Fetch the PDF content
                pdf_response = requests.get(url, headers=HEADERS)
                pdf_response.raise_for_status()
                
                # Open the PDF from the downloaded content in memory
                with fitz.open(stream=pdf_response.content, filetype="pdf") as doc:
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()
                
                # Clean the extracted text
                clean_text = re.sub(r'\s+', ' ', full_text).strip()
                
                # Save the text to a file
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                print(f"  -> Saved text to {save_path}")

                time.sleep(1) # Be respectful to the server

            except requests.exceptions.RequestException as e:
                print(f"  -> Failed to download {url}: {e}")
            except Exception as e:
                # This can catch errors from fitz if the PDF is corrupted/unreadable
                print(f"  -> An error occurred while processing PDF {url}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the main index page: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("--- Income Tax Department Scraper Finished ---")


# --- Run the Scraper ---
if __name__ == "__main__":
    scrape_incometax_circulars()