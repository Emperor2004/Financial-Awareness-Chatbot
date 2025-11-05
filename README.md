# Financial Information Retrieval Assistant

**Financial Awareness Chatbot** is an intelligent, agentic chatbot designed to provide accurate and accessible financial information to users in India. It leverages a Retrieval-Augmented Generation (RAG) architecture to answer user queries based on a trusted knowledge base built from official government and regulatory sources.

This project addresses the challenge of navigating dense and complex financial documents on websites like the RBI, SEBI, and the Income Tax Department, providing users with clear, synthesized answers and direct source links.

**Disclaimer:** This tool is for informational and educational purposes only. It is **not** a financial advisor and does not provide financial, legal, or tax advice. Always consult a qualified human professional before making any financial decisions.

---

## 1. Problem Statement

Official financial information in India is spread across numerous government websites. This information is often presented in lengthy circulars, dense legal documents, and hard-to-navigate FAQs. For the average citizen seeking a clear answer to a specific question—such as "What are the tax benefits of PPF?" or "What is the process for reporting a fraudulent transaction?"—finding the right information is a time-consuming and often frustrating process. This information gap can lead to misinformation and poor financial decisions.

---

## 2. Solution

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

* **Backend**:
    * **Framework**: **FastAPI**
    * **AI/NLP**: **LangChain**, **Ollama**
    * **Vector Database**: **ChromaDB**
    * **Data Collection**: **BeautifulSoup4**, **PyMuPDF**, **Requests**, **Selenium**, **Undetected Chromedriver**
* **Frontend**:
    * **Framework**: **SvelteKit**
    * **Styling**: **Tailwind CSS**, **DaisyUI**
    * **Language**: **TypeScript**
---

## 4. Project Structure and File Descriptions

The project is organized into a monorepo with two primary directories: `backend` for the Python-based API and AI logic, and `frontend` for the SvelteKit-based user interface.
```
Financial Awareness Chatbot/
│
├── backend/
│   ├── scripts/
│   ├── data/
│   ├── db/
│   ├── .env
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── static/
│   ├── package.json
│   ├── svelte.config.js
│   └── tailwind.config.cjs
└── README.md
```

## 5. Setup and Installation
Follow these steps to get the project running on your local machine.

Clone the Repository:

```Bash
git clone <your-repository-url>
cd Financial Awareness Chatbot
```
### Backend Setup

1.  **Navigate to the backend directory**:
    ```bash
    cd backend
    ```
2.  **Create and activate a Python virtual environment**:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your Google API Key**:
    * Create a `.env` file in the `backend` directory.
    * Add your key: `GOOGLE_API_KEY="YOUR_API_KEY_HERE"`

---

### Frontend Setup

1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend
    ```
2.  **Install Node.js dependencies**:
    ```bash
    npm install
    ```

---

## 6. Usage Workflow
The project is run in three stages. You will need two separate terminals to run the backend and frontend servers simultaneously.

#### **Stage 1: Prepare the Knowledge Base** (in `backend` terminal)

This only needs to be done once to build your chatbot's brain.

1.  **Collect Data**: Run the scrapers to fetch documents. For sites that block scraping (like SEBI), you will need to manually download the relevant PDF or text files into the `data/` directory.
    ```bash
    python scripts/scrape_fiu.py
    python scripts/scrape_incometax.py
    ```
2.  **Build the Vector Database**: Process all text files into the ChromaDB store.
    ```bash
    python scripts/vector_embeddings.py
    ```

#### **Stage 2: Run the Servers**

1.  **Start the Backend API** (in `backend` terminal):
    ```bash
    uvicorn app:app --reload
    ```
    The backend API will be running on `http://127.0.0.1:8000`.

#### **Stage 3: Run the Frontends**
1.  **Start the Frontend Application** (in `frontend` terminal):
    ```bash
    npm run dev
    ```
    The frontend will be accessible at `http://localhost:5173`.

You can now open your browser and navigate to **http://localhost:5173** to use the Financial Awareness Chatbot.
