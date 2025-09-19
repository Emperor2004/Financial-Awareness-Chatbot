# Financial Information Retrieval Assistant (Fin-Info-Bot)

**Fin-Info-Bot** is an intelligent, agentic chatbot designed to provide accurate and accessible financial information to users in India. It leverages a Retrieval-Augmented Generation (RAG) architecture to answer user queries based on a trusted knowledge base built from official government and regulatory sources.

This project addresses the challenge of navigating dense and complex financial documents on websites like the RBI, SEBI, and the Income Tax Department, providing users with clear, synthesized answers and direct source links.

**Disclaimer:** This tool is for informational and educational purposes only. It is **not** a financial advisor and does not provide financial, legal, or tax advice. Always consult a qualified human professional before making any financial decisions.

---

## 1. Problem Statement

Official financial information in India is spread across numerous government websites. This information is often presented in lengthy circulars, dense legal documents, and hard-to-navigate FAQs. For the average citizen seeking a clear answer to a specific question—such as "What are the tax benefits of PPF?" or "What is the process for reporting a fraudulent transaction?"—finding the right information is a time-consuming and often frustrating process. This information gap can lead to misinformation and poor financial decisions.

---

## 2. Solution: An Agentic RAG Chatbot

This project solves the problem by implementing a **Retrieval-Augmented Generation (RAG)** system. This architecture ensures that the chatbot's answers are grounded in facts from our curated knowledge base, significantly reducing the risk of AI "hallucinations" or providing incorrect information.



The workflow is as follows:
1.  **Data Ingestion**: Automated scripts (scrapers) and manual downloads are used to collect documents from authoritative sources (FIU, Income Tax Dept., SEBI, RBI).
2.  **Vectorization**: The collected text is cleaned, broken into smaller chunks, and converted into numerical representations (embeddings) that capture semantic meaning. These embeddings are stored in a specialized vector database.
3.  **User Query**: A user asks a question in plain English.
4.  **Retrieval**: The system converts the user's question into an embedding and searches the vector database to find the most relevant text chunks from the original documents.
5.  **Augmentation & Generation**: The retrieved text chunks are passed to a Large Language Model (LLM) like Google's Gemini along with a carefully crafted prompt. The LLM uses this context to generate a clear, human-readable answer.
6.  **Citation**: The final answer is presented to the user along with direct links to the source documents, ensuring transparency and trust.

---

## 3. Technology Stack

This project utilizes a modern, Python-based stack for AI and web development.

* **Backend**: **Flask** (a lightweight Python web framework for serving the API).
* **AI/NLP Framework**: **LangChain** (for orchestrating the RAG pipeline, including data loading, chunking, and interacting with the LLM).
* **LLM & Embeddings**: **Google Gemini & Google Embeddings API**.
* **Vector Database**: **ChromaDB** (for local development and efficient similarity search).
* **Data Collection**:
    * **Requests** & **BeautifulSoup4**: For scraping HTML content.
    * **PyMuPDF**: For extracting text from PDF documents.
* **Frontend**: Plain **HTML**, **CSS**, and **JavaScript** for a clean, responsive user interface.

---

## 4. Project Structure

The project is organized into a clean and scalable directory structure:

```bash
fin-info-bot/
│
├── scripts/              # Data collection and processing scripts
│   ├── scrape_fiu.py
│   ├── scrape_incometax.py
│   ├── process_local_pdfs.py  # For manually downloaded files
│   └── ingest.py            # Processes all data into the vector DB
│
├── app.py                # Main Flask application and API endpoint
│
├── templates/
│   └── index.html        # Frontend chat interface
│
├── static/
│   ├── css/style.css     # Styling for the chat
│   └── js/script.js      # Frontend logic for API calls
│
├── db/                   # Local Chroma vector database storage
│
├── data/                 # Raw text data scraped from sources
│   ├── fiu/
│   ├── incometax/
│   └── sebi_from_pdf/
│
├── .env                  # Stores secret API keys (e.g., GOOGLE_API_KEY)
└── README.md             # This file
```

## 5. Setup and Installation
Follow these steps to get the project running on your local machine.

1. Clone the Repository:

```Bash
git clone <your-repository-url>
cd fin-info-bot
```
2. Create and Activate a Virtual Environment:

```Bash
python -m venv fin_venv
# On Windows
.\fin_venv\Scripts\activate
# On macOS/Linux
source fin_venv/bin/activate
```
3. Install Dependencies:
Create a `requirements.txt` file with the following content:

```Plaintext
flask
langchain
langchain-google-genai
langchain-community
chromadb
beautifulsoup4
requests
PyMuPDF
python-dotenv
```
Then, install the packages:

```Bash
pip install -r requirements.txt
```
4. Set Up API Key:
- Create a file named `.env` in the root project directory.

- Get your API key from <a>Google AI Studio</a>.

- Add your key to the .env file:

`GOOGLE_API_KEY="YOUR_API_KEY_HERE"`

## 6. Usage Workflow
The project is run in three stages:

Stage 1: Data Collection
Run the scrapers to collect the latest information.

```Bash
python scripts/scrape_fiu.py
python scripts/scrape_incometax.py
```
For protected sites like SEBI and RBI, perform the manual PDF download as described in the development process and run the local processor:

```Bash
python scripts/process_local_pdfs.py
```
Stage 2: Data Ingestion

Process all the collected text files and build your vector database. This only needs to be done once after you've collected your data.

```Bash
python scripts/ingest.py
```
Stage 3: Run the Application

Start the Flask web server.

```Bash

python app.py
```
Open your web browser and navigate to http://127.0.0.1:5000 to start chatting with the bot.

## 7. Frontend UI/UX
The user interface is designed to be simple, intuitive, and mobile-friendly.

- **Chat Window:** A clean, scrollable window displays the conversation history. User messages are aligned to the right, and bot responses are aligned to the left.

- Input Area:** A text box at the bottom allows users to type their questions, with a clear "Send" button.

- **Loading Indicator:** While the bot is processing a query and generating a response, a subtle loading indicator appears to provide feedback to the user.

- **Source Citation:** Each response from the bot includes a list of clickable source URLs, allowing users to verify the information directly from the official documents.

- **Disclaimer:** A clear and persistent disclaimer is visible on the page to inform users of the tool's purpose and limitations.