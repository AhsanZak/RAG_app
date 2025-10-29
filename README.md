# RAG Chat Application

A modern AI-powered chat application built with FastAPI backend and React.js frontend, featuring Retrieval-Augmented Generation (RAG) capabilities for analyzing and chatting with news articles and other data from a vector database.

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

## Features (Planned)

- 🤖 **AI-Powered Chat**: Intelligent conversations using RAG technology
- 📊 **Vector Database**: Semantic search and retrieval capabilities
- 🚀 **FastAPI Backend**: High-performance API with automatic documentation
- ⚛️ **React Frontend**: Modern UI with Ant Design components
- 📰 **News Analysis**: Chat with news articles and other content
- 🔍 **Semantic Search**: Find relevant information using vector embeddings

## Current Status

This is the initial setup with:
- ✅ Basic FastAPI backend structure
- ✅ React frontend with welcome screen
- ✅ Ant Design UI components
- ✅ Project structure and configuration files
- ✅ CORS configuration for frontend-backend communication

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

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

### Current Endpoints

- `GET /` - Root endpoint (health check)
- `GET /health` - Health check endpoint
- `POST /chat` - Main chat endpoint (placeholder)
- `GET /api/endpoints` - List all available endpoints

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint (placeholder)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

## Next Steps

This is just the beginning! Future development will include:

1. **Vector Database Integration** (ChromaDB/Pinecone)
2. **Document Processing Pipeline**
3. **Embedding Generation**
4. **RAG Implementation**
5. **Chat Interface**
6. **News Data Integration**
7. **Advanced Search Features**

## Technology Stack

- **Backend**: FastAPI, Python
- **Frontend**: React.js, Ant Design
- **Database**: Vector Database (TBD)
- **AI/ML**: Embeddings, RAG (TBD)

## Contributing

This project is in active development. More features and documentation will be added step by step.

## License

This project is for educational and development purposes.
