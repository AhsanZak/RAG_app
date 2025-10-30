# RAG Chat Application

An AI-powered chat application built with FastAPI (backend) and React + Ant Design (frontend) featuring Retrieval-Augmented Generation (RAG). You can upload PDFs, vectorize them with language-aware embeddings, and chat over their content using a local or remote LLM.

## Project Structure

```
RAG_App/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── .gitignore         # Python gitignore
├── frontend/               # React.js frontend
│   ├── public/            # Static files
│   ├── src/               # React source code
│   │   ├── App.js         # Main React component
│   │   ├── index.js       # React entry point
│   │   └── index.css      # Global styles
│   ├── package.json       # Node.js dependencies
│   └── .gitignore         # Node.js gitignore
└── README.md              # This file
```

## Key Features (Current)

- 🤖 Chat over documents with RAG (ChromaDB for vector storage)
- 📄 PDF ingestion with OCR fallback for scanned/image-based PDFs
- 🌐 Language detection and auto-embedding model selection
- 🧩 Embedding models from configuration (no DB required), on-demand download
- 🧠 LLM model management (e.g., Ollama/OpenAI) with model selector in UI
- 🖥️ Clean PDF chat UI (static upload area + scrollable sessions list)
- 📝 First message preview/summary after processing for quick verification

## Current Status

- ✅ FastAPI backend with PDF upload, processing (OCR), chat endpoints
- ✅ Embedding models config with downloadable models (HuggingFace)
- ✅ React UI for PDF chat, LLM model selection, embedding selection
- ✅ ChromaDB integration with robust metadata handling
- ✅ Sessions, messages, and collection existence checks

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- System OCR dependencies (for scanned PDFs):
  - Tesseract OCR
    - Windows: install Tesseract and add `tesseract.exe` to PATH
    - Linux: `sudo apt install tesseract-ocr`
    - macOS: `brew install tesseract`
  - Poppler (for `pdf2image`)
    - Windows: install Poppler and add its `bin` folder to PATH
    - Linux: `sudo apt install poppler-utils`
    - macOS: `brew install poppler`

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

Backend dependencies to note (pip-installed):

- `pdf2image`, `pytesseract`, `Pillow`, `langdetect`, `pdfplumber`, `PyPDF2`, `chromadb`, `sentence-transformers`, `huggingface-hub`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000`

## API Endpoints

Core

- `GET /` – root health
- `GET /health` – health check
- `GET /api/endpoints` – list endpoints

Embedding models (from configuration file `backend/config/embedding_models.py`)

- `GET /api/embedding-models` – list available embedding models
- `GET /api/embedding-models/by-name/{model_name}` – details for one model
- `POST /api/embedding-models/{model_name}/download` – download model (on-demand)
- `GET /api/embedding-models/status|test|direct` – diagnostics

PDF workflow

- `POST /api/pdf/upload` – upload a PDF (returns extracted chunks, `pdf_type`, and sample text)
- `POST /api/pdf/process` – vectorize chunks into ChromaDB (OCR fallback, language detect, preview summary)
- `POST /api/pdf/chat` – RAG query against the session’s collection
- `GET /api/sessions` – list sessions (includes `hasVectorizedData` and `collection_name`)
- `GET /api/sessions/{session_id}/messages` – list messages for a session

## How it Works (Highlights)

1. Upload: `/api/pdf/upload` extracts text with `pdfplumber`/`PyPDF2`. If no text is found, OCR fallback (`pdf2image` + `pytesseract`) attempts extraction. Returns `pdf_type` for UI info.
2. Process: `/api/pdf/process` chunks data, auto-detects language (`langdetect`), auto-selects embedding model (Arabic → `BAAI/bge-m3`, English → `all-mpnet-base-v2`, otherwise → `intfloat/multilingual-e5-base`), generates embeddings, and stores them in ChromaDB. It also saves a first assistant message with a preview/summary.
3. Chat: `/api/pdf/chat` retrieves relevant chunks from ChromaDB and generates answers using the selected LLM.

Guards

- If no text is found after OCR, processing returns `400` with a clear message instead of failing inside ChromaDB.
- Before chat, the server checks that the session’s collection exists.

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, ChromaDB
- **Frontend**: React.js, Ant Design
- **Embeddings**: sentence-transformers/HF models (config-driven)
- **OCR**: Tesseract via `pytesseract`, `pdf2image`, `Pillow`
- **LLM**: pluggable (e.g., Ollama/OpenAI) via model selector

## Troubleshooting

- Scanned PDF still fails
  - Ensure Tesseract and Poppler are installed and on PATH.
  - Check backend logs for lines starting with `[PDF]` / `[PDF][OCR]` / `[UPLOAD]` / `[PROCESS]`.
- Embedding model list is empty
  - Check `GET /api/embedding-models/status` and `GET /api/embedding-models`.
  - Confirm `backend/config/embedding_models.py` exists and is importable.
- ChromaDB telemetry warnings
  - Telemetry is disabled in the service; warnings can be ignored if present.

## License

This project is for educational and development purposes.
