# Section-Aware Knowledge Base

This directory contains the ChromaDB vector database created with section-aware chunking strategy.

## Directory Structure

- `chroma.sqlite3` - ChromaDB SQLite database file
- `[collection-id]/` - Vector index files for the collection

## Collection Details

- **Collection Name**: `financial_regulations_section_aware`
- **Embedding Model**: `intfloat/e5-large-v2`
- **Chunking Strategy**: Section-aware hybrid chunking (major sections preserved)
- **Created By**: `ai_core/ingest_e5_section_aware.py`

## Notes

- Database files are excluded from git (see `.gitignore`)
- Regenerate this database by running `python ai_core/ingest_e5_section_aware.py`
- This knowledge base replaces the previous `db_e5/` database with improved section-aware chunking

