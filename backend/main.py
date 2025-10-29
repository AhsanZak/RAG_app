from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import requests
import json
import aiofiles
import os
from datetime import datetime
from sqlalchemy.orm import Session
from database import get_db
from models import User, LLMModel, ChatSession, ChatMessage as ChatMessageModel, LLMProvider
from services.pdf_processor import PDFProcessor
from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService
from services.rag_service import RAGService
from services.llm_service import LLMService

app = FastAPI(
    title="RAG Chat API",
    description="A FastAPI backend for RAG-based chat application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor = PDFProcessor()
embedding_service = EmbeddingService()
chromadb_service = ChromaDBService()
rag_service = RAGService()
llm_service = LLMService()

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    message: str

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="success",
        message="RAG Chat API is running!"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is operational"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """
    Main chat endpoint for RAG-based conversations
    This will be implemented in future steps
    """
    # Placeholder response
    return ChatResponse(
        response=f"Echo: {message.message}",
        sources=["placeholder_source"],
        timestamp="2024-01-01T00:00:00Z"
    )

@app.get("/api/endpoints")
async def list_endpoints():
    """List all available API endpoints"""
    return {
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Root endpoint - health check"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint"
            },
            {
                "path": "/chat",
                "method": "POST",
                "description": "Main chat endpoint for RAG conversations"
            },
            {
                "path": "/api/endpoints",
                "method": "GET",
                "description": "List all available endpoints"
            },
            {
                "path": "/api/llm-models",
                "method": "GET",
                "description": "List all LLM models"
            },
            {
                "path": "/api/llm-models/{id}",
                "method": "GET",
                "description": "Get single LLM model by ID"
            },
            {
                "path": "/api/llm-models",
                "method": "POST",
                "description": "Create new LLM model"
            },
            {
                "path": "/api/llm-models/{id}",
                "method": "PUT",
                "description": "Update LLM model"
            },
            {
                "path": "/api/llm-models/{id}",
                "method": "DELETE",
                "description": "Delete LLM model"
            },
            {
                "path": "/api/users",
                "method": "GET",
                "description": "List all users"
            }
        ]
    }

# Database endpoints
@app.get("/api/llm-models")
async def get_llm_models(db: Session = Depends(get_db)):
    """Get all LLM models"""
    models = db.query(LLMModel).all()
    return models

@app.get("/api/llm-models/{model_id}")
async def get_llm_model(model_id: int, db: Session = Depends(get_db)):
    """Get single LLM model by ID"""
    model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@app.post("/api/llm-models")
async def create_llm_model(model_data: dict, db: Session = Depends(get_db)):
    """Create new LLM model"""
    model = LLMModel(**model_data)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model

@app.put("/api/llm-models/{model_id}")
async def update_llm_model(model_id: int, model_data: dict, db: Session = Depends(get_db)):
    """Update LLM model"""
    model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    for key, value in model_data.items():
        setattr(model, key, value)
    
    db.commit()
    db.refresh(model)
    return model

@app.delete("/api/llm-models/{model_id}")
async def delete_llm_model(model_id: int, db: Session = Depends(get_db)):
    """Delete LLM model"""
    model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db.delete(model)
    db.commit()
    return {"message": "Model deleted successfully"}

@app.get("/api/users")
async def get_users(db: Session = Depends(get_db)):
    """Get all users"""
    users = db.query(User).all()
    return users

@app.post("/api/llm-models/test")
async def test_llm_connection(model_data: dict):
    """Test LLM model connection"""
    provider = model_data.get('provider')
    base_url = model_data.get('base_url')
    model_name = model_data.get('model_name')
    
    if provider == 'ollama':
        return await test_ollama_connection(base_url, model_name)
    elif provider == 'openai':
        return await test_openai_connection(base_url, model_data.get('api_key'), model_name)
    else:
        return {
            "success": False,
            "message": f"Connection testing not implemented for {provider} provider",
            "details": "This provider is not yet supported for connection testing"
        }

async def test_ollama_connection(base_url: str, model_name: str):
    """Test Ollama connection and model availability"""
    try:
        # Test if Ollama server is running
        health_url = f"{base_url.rstrip('/')}/api/tags"
        response = requests.get(health_url, timeout=10)
        
        if response.status_code != 200:
            return {
                "success": False,
                "message": "Ollama server is not responding",
                "details": f"Server returned status code: {response.status_code}"
            }
        
        # Get available models
        available_models = response.json().get('models', [])
        model_names = [model['name'] for model in available_models]
        
        # Check if the specified model is available
        model_found = any(model_name in model for model in model_names)
        
        if model_found:
            return {
                "success": True,
                "message": f"Ollama connection successful! Model '{model_name}' is available.",
                "details": {
                    "server_status": "running",
                    "available_models": model_names,
                    "requested_model": model_name,
                    "model_found": True
                }
            }
        else:
            return {
                "success": False,
                "message": f"Model '{model_name}' not found on Ollama server",
                "details": {
                    "server_status": "running",
                    "available_models": model_names,
                    "requested_model": model_name,
                    "model_found": False,
                    "suggestion": f"Available models: {', '.join(model_names[:5])}{'...' if len(model_names) > 5 else ''}"
                }
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": "Cannot connect to Ollama server",
            "details": "Make sure Ollama is running and the URL is correct"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "Connection timeout",
            "details": "Ollama server took too long to respond"
        }
    except Exception as e:
        return {
            "success": False,
            "message": "Unexpected error occurred",
            "details": str(e)
        }

async def test_openai_connection(base_url: str, api_key: str, model_name: str):
    """Test OpenAI connection"""
    try:
        if not api_key:
            return {
                "success": False,
                "message": "API key is required for OpenAI",
                "details": "Please provide a valid OpenAI API key"
            }
        
        # Test with a simple completion request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        test_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "message": f"OpenAI connection successful! Model '{model_name}' is working.",
                "details": {
                    "api_status": "valid",
                    "model": model_name,
                    "response_time": f"{response.elapsed.total_seconds():.2f}s"
                }
            }
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            return {
                "success": False,
                "message": f"OpenAI API error: {response.status_code}",
                "details": error_data.get('error', {}).get('message', 'Unknown error')
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": "Cannot connect to OpenAI API",
            "details": "Check your internet connection and API endpoint"
        }
    except Exception as e:
        return {
            "success": False,
            "message": "Unexpected error occurred",
            "details": str(e)
        }

# PDF Chat Endpoints
@app.post("/api/pdf/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: int = Form(1),  # Default user, should be from auth
    db: Session = Depends(get_db)
):
    """Upload and process PDF file"""
    try:
        # Read file content
        contents = await file.read()
        
        # Extract text from PDF
        extraction_result = pdf_processor.extract_text_from_pdf(contents, file.filename)
        
        # Store file info temporarily (in production, save to storage)
        return {
            "success": True,
            "filename": file.filename,
            "chunks": extraction_result["chunks"],
            "metadata": extraction_result["metadata"],
            "total_chunks": extraction_result["total_chunks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/api/pdf/process")
async def process_pdfs(
    session_data: dict,
    db: Session = Depends(get_db)
):
    """Process PDFs and create ChromaDB collection"""
    try:
        files = session_data.get("files", [])
        user_id = session_data.get("user_id", 1)
        session_name = session_data.get("session_name", "New Session")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create or get session
        session = ChatSession(
            user_id=user_id,
            session_name=session_name,
            llm_model_id=session_data.get("llm_model_id", 1),
            created_at=datetime.utcnow()
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Collection name based on session ID
        collection_name = f"session_{session.id}"
        
        all_texts = []
        all_metadatas = []
        all_ids = []
        
        # Process each file
        for file_data in files:
            chunks = file_data.get("chunks", [])
            filename = file_data.get("filename")
            
            for idx, chunk in enumerate(chunks):
                all_texts.append(chunk["text"])
                # Ensure metadata is always a non-empty dict (ChromaDB requirement)
                metadata = chunk.get("metadata", {}) or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                
                # Always add required fields to ensure non-empty metadata
                metadata["session_id"] = str(session.id)
                metadata["filename"] = filename or "unknown.pdf"
                metadata["page_number"] = chunk.get("page", idx + 1)
                metadata["chunk_index"] = idx
                
                all_metadatas.append(metadata)
                all_ids.append(f"{filename}_{chunk.get('page', idx)}_{idx}")
        
        # Generate embeddings
        embeddings = embedding_service.generate_embeddings(all_texts)
        
        # Add to ChromaDB
        chromadb_service.add_documents(
            collection_name=collection_name,
            texts=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        return {
            "success": True,
            "session_id": session.id,
            "collection_name": collection_name,
            "total_documents": len(all_texts),
            "message": "PDFs processed and vectorized successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDFs: {str(e)}")

@app.post("/api/pdf/chat")
async def pdf_chat(
    chat_data: dict,
    db: Session = Depends(get_db)
):
    """Chat with PDF documents using RAG"""
    try:
        session_id = chat_data.get("session_id")
        message = chat_data.get("message")
        model_id = chat_data.get("model_id")
        user_id = chat_data.get("user_id", 1)
        
        if not session_id or not message:
            raise HTTPException(status_code=400, detail="session_id and message are required")
        
        # Get session
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get model config
        model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_config = {
            "provider": model.provider.value,
            "model_name": model.model_name,
            "base_url": model.base_url
        }
        
        # Collection name
        collection_name = f"session_{session_id}"
        
        # Query using RAG
        rag_result = rag_service.query(
            query_text=message,
            collection_name=collection_name,
            model_config=model_config,
            n_results=5
        )
        
        # Save user message
        user_message = ChatMessageModel(
            session_id=session_id,
            role="user",
            message=message,
            created_at=datetime.utcnow()
        )
        db.add(user_message)
        
        # Save assistant message
        assistant_message = ChatMessageModel(
            session_id=session_id,
            role="assistant",
            message=rag_result["response"],
            meta_data={"sources": rag_result["sources"]},
            created_at=datetime.utcnow()
        )
        db.add(assistant_message)
        db.commit()
        
        return {
            "response": rag_result["response"],
            "sources": rag_result["sources"],
            "message_id": assistant_message.id,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/sessions")
async def get_sessions(
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Get all sessions for a user"""
    sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
    return [
        {
            "id": s.id,
            "name": s.session_name,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "message_count": len(s.chat_messages)
        }
        for s in sessions
    ]

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get all messages for a session"""
    messages = db.query(ChatMessageModel).filter(
        ChatMessageModel.session_id == session_id
    ).order_by(ChatMessageModel.created_at).all()
    
    return [
        {
            "id": m.id,
            "role": m.role,
            "message": m.message,
            "metadata": m.meta_data,
            "created_at": m.created_at.isoformat()
        }
        for m in messages
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
