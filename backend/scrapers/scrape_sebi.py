import os
import re
import fitz  # PyMuPDF

# --- Configuration ---
# The folder where you manually saved the PDFs
SEBI_PDF_SOURCE_DIR = "data/manual_pdf_downloads/sebi"
# The folder where the extracted text files will be saved
TEXT_SAVE_DIR = "data/sebi"

# --- Main Processing Logic ---
def process_downloaded_pdfs():
    """Reads all PDFs from a source folder, extracts their text, and saves it."""
    print("--- Starting Local PDF Processor ---")

    if not os.path.exists(SEBI_PDF_SOURCE_DIR):
        print(f"Error: Source directory not found at '{SEBI_PDF_SOURCE_DIR}'")
        print("Please create it and download the SEBI PDFs into it.")
        return

    if not os.path.exists(TEXT_SAVE_DIR):
        os.makedirs(TEXT_SAVE_DIR)
        print(f"Created directory: {TEXT_SAVE_DIR}")

    pdf_files = [f for f in os.listdir(SEBI_PDF_SOURCE_DIR) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{SEBI_PDF_SOURCE_DIR}'.")
        return

    print(f"Found {len(pdf_files)} PDFs to process.")

    for pdf_filename in pdf_files:
        pdf_path = os.path.join(SEBI_PDF_SOURCE_DIR, pdf_filename)
        # Create a corresponding .txt filename
        txt_filename = os.path.splitext(pdf_filename)[0] + ".txt"
        save_path = os.path.join(TEXT_SAVE_DIR, txt_filename)

        if os.path.exists(save_path):
            print(f"Skipping already processed: {pdf_filename}")
            continue

        print(f"Processing: {pdf_filename}")
        try:
            with fitz.open(pdf_path) as doc:
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
            
            # Clean up the text
            clean_text = re.sub(r'\s+', ' ', full_text).strip()

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            print(f"  -> Extracted text to {save_path}")

        except Exception as e:
            print(f"  -> Failed to process {pdf_filename}: {e}")

    print("--- Local PDF Processing Finished ---")


if __name__ == "__main__":
    process_downloaded_pdfs()