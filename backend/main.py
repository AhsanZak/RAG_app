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
from models import User, LLMModel, ChatSession, ChatMessage as ChatMessageModel, LLMProvider, EmbeddingModel
from config.embedding_models import AVAILABLE_EMBEDDINGS, get_default_embedding_model, get_embedding_model_info, list_embedding_models
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

# Verify embedding models configuration on startup
try:
    from config.embedding_models import AVAILABLE_EMBEDDINGS
    print(f"✅ Embedding models configuration loaded: {len(AVAILABLE_EMBEDDINGS)} models available")
    print(f"   Models: {', '.join(list(AVAILABLE_EMBEDDINGS.keys())[:5])}...")
except Exception as e:
    print(f"❌ Failed to load embedding models configuration: {str(e)}")
    import traceback
    traceback.print_exc()

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

# Embedding Model Endpoints
@app.get("/api/embedding-models")
async def get_embedding_models(language: Optional[str] = None):
    """Get all available embedding models from configuration"""
    try:
        # Import AVAILABLE_EMBEDDINGS with error handling
        try:
            # Try using the module-level import first
            models_to_return = AVAILABLE_EMBEDDINGS
        except NameError:
            # If not available, import it
            try:
                from config.embedding_models import AVAILABLE_EMBEDDINGS
                models_to_return = AVAILABLE_EMBEDDINGS
            except ImportError as import_err:
                # Try alternative import paths
                import sys
                import os
                backend_dir = os.path.dirname(os.path.abspath(__file__))
                if backend_dir not in sys.path:
                    sys.path.insert(0, backend_dir)
                from config.embedding_models import AVAILABLE_EMBEDDINGS
                models_to_return = AVAILABLE_EMBEDDINGS
        
        if not models_to_return or len(models_to_return) == 0:
            print("[ERROR] AVAILABLE_EMBEDDINGS is empty!")
            return []
        
        print(f"[DEBUG] Found {len(models_to_return)} models in configuration")
        
        # Apply language filter if provided
        if language:
            if language == "multi" or language == "multilingual":
                models_to_return = {k: v for k, v in models_to_return.items() if "multi" in v.get("languages", [])}
            else:
                models_to_return = {k: v for k, v in models_to_return.items() if language in v.get("languages", [])}
        
        # Convert to list format for frontend
        result = []
        for idx, (model_name, config) in enumerate(models_to_return.items(), 1):
            try:
                languages = config.get("languages", [])
                language_str = languages[0] if languages else "unknown"
                if "multi" in languages:
                    language_str = "multilingual"
                
                result.append({
                    "id": idx,
                    "model_name": model_name,
                    "display_name": config.get("display_name", model_name),
                    "description": config.get("description", ""),
                    "language": language_str,
                    "model_path": model_name,
                    "dimension": config.get("dimension", 384),
                    "is_active": 1,
                    "is_downloaded": 0,
                    "download_progress": 0,
                    "provider": config.get("provider", "sentence-transformers"),
                    "languages": languages,
                    "is_default": config.get("is_default", False)
                })
            except Exception as e:
                print(f"[ERROR] Failed to process model {model_name}: {str(e)}")
                continue
        
        print(f"[DEBUG] Returning {len(result)} models")
        return result
    except Exception as e:
        print(f"❌ Error getting embedding models: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get embedding models: {str(e)}")

@app.get("/api/embedding-models/status")
async def embedding_models_status():
    """Check status of embedding models configuration"""
    try:
        # Check if AVAILABLE_EMBEDDINGS is available
        try:
            models_count = len(AVAILABLE_EMBEDDINGS)
            models_list = list(AVAILABLE_EMBEDDINGS.keys())[:5]
            return {
                "status": "loaded",
                "count": models_count,
                "sample_models": models_list,
                "message": f"Successfully loaded {models_count} embedding models"
            }
        except NameError:
            # Try to import it
            from config.embedding_models import AVAILABLE_EMBEDDINGS
            models_count = len(AVAILABLE_EMBEDDINGS)
            models_list = list(AVAILABLE_EMBEDDINGS.keys())[:5]
            return {
                "status": "loaded_via_import",
                "count": models_count,
                "sample_models": models_list,
                "message": f"Successfully loaded {models_count} embedding models via import"
            }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "message": "Failed to load embedding models configuration"
        }

@app.get("/api/embedding-models/test")
async def test_embedding_models():
    """Test endpoint to verify embedding models configuration"""
    try:
        from config.embedding_models import AVAILABLE_EMBEDDINGS
        models_count = len(AVAILABLE_EMBEDDINGS)
        first_model = list(AVAILABLE_EMBEDDINGS.keys())[0] if AVAILABLE_EMBEDDINGS else None
        
        # Test list_embedding_models function
        test_result = list_embedding_models()
        print(f"AVAILABLE_EMBEDDINGS count: {models_count}")
        print(f"list_embedding_models() count: {len(test_result)}")
        
        return {
            "success": True,
            "total_models": models_count,
            "first_model": first_model,
            "list_function_works": len(test_result) > 0,
            "models": list(AVAILABLE_EMBEDDINGS.keys()),
            "message": "Embedding models configuration loaded successfully"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to load embedding models configuration"
        }

@app.get("/api/embedding-models/direct")
async def get_embedding_models_direct():
    """Direct endpoint to return raw models - for debugging"""
    try:
        from config.embedding_models import AVAILABLE_EMBEDDINGS
        return {
            "total": len(AVAILABLE_EMBEDDINGS),
            "models": [
                {
                    "model_name": k,
                    "display_name": v.get("display_name", k),
                    "description": v.get("description", ""),
                    "language": v.get("languages", [])[0] if v.get("languages") else "unknown",
                    "provider": v.get("provider", ""),
                    "dimension": v.get("dimension", 384)
                }
                for k, v in AVAILABLE_EMBEDDINGS.items()
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/embedding-models/by-name/{model_name}")
async def get_embedding_model_by_name(model_name: str):
    """Get single embedding model by name"""
    config = get_embedding_model_info(model_name)
    if not config:
        raise HTTPException(status_code=404, detail="Embedding model not found")
    
    languages = config.get("languages", [])
    language_str = languages[0] if languages else "unknown"
    if "multi" in languages:
        language_str = "multilingual"
    
    return {
        "model_name": model_name,
        "display_name": config.get("display_name", model_name),
        "description": config.get("description", ""),
        "language": language_str,
        "model_path": model_name,
        "dimension": config.get("dimension", 384),
        "provider": config.get("provider", "sentence-transformers"),
        "languages": languages,
        "is_default": config.get("is_default", False)
    }

@app.post("/api/embedding-models/{model_name}/download")
async def download_embedding_model(model_name: str):
    """Download embedding model from HuggingFace"""
    config = get_embedding_model_info(model_name)
    if not config:
        raise HTTPException(status_code=404, detail="Embedding model not found")
    
    try:
        # Download model
        result = embedding_service.download_model(model_name)
        
        if result.get("success"):
            return {
                "success": True,
                "message": result.get("message", "Model downloaded successfully"),
                "model_path": result.get("model_path"),
                "model_name": model_name
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Download failed"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")

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
        embedding_model_id = session_data.get("embedding_model_id")  # May be null, using config now
        embedding_model_name = session_data.get("embedding_model_name")
        
        # Determine embedding model name from config
        if not embedding_model_name:
            if embedding_model_id:
                # If ID provided, try to find model name (for backwards compatibility)
                # But now we use config, so we'll use the default
                embedding_model_name = get_default_embedding_model()
            else:
                # Use default model name from config
                embedding_model_name = get_default_embedding_model()
        else:
            # Validate model name exists in config
            if not get_embedding_model_info(embedding_model_name):
                # If not found, use default
                embedding_model_name = get_default_embedding_model()
        
        session = ChatSession(
            user_id=user_id,
            session_name=session_name,
            llm_model_id=session_data.get("llm_model_id", 1),
            embedding_model_id=None,  # No longer using DB for embedding models
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
        
        # Generate embeddings using selected embedding model
        embeddings = embedding_service.generate_embeddings(all_texts, model_name=embedding_model_name)
        
        # Add to ChromaDB
        chromadb_service.add_documents(
            collection_name=collection_name,
            texts=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        # Build preview and summary for user verification
        preview_samples = []
        max_preview_chunks = min(3, len(all_texts))
        for i in range(max_preview_chunks):
            preview_samples.append(all_texts[i].strip())
        # Simple summary: concatenate first few chunks and trim
        combined_text = "\n\n".join(preview_samples) if preview_samples else ""
        summary_text = combined_text[:2000]

        # Save an initial assistant message with the summary/preview
        try:
            assistant_intro = (
                "Your documents have been processed and vectorized. "
                "Here's a quick preview of the extracted content so you can verify everything looks right.\n\n"
                f"{summary_text}"
            )
            assistant_message = ChatMessageModel(
                session_id=session.id,
                role="assistant",
                message=assistant_intro,
                created_at=datetime.utcnow()
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)
            assistant_message_id = assistant_message.id
        except Exception:
            assistant_message_id = None

        return {
            "success": True,
            "session_id": session.id,
            "collection_name": collection_name,
            "total_documents": len(all_texts),
            "message": "PDFs processed and vectorized successfully",
            "preview": {
                "samples": preview_samples,
                "summary": summary_text
            },
            "assistant_message_id": assistant_message_id
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
        
        # Get embedding model from session
        embedding_model = None
        embedding_model_name = None
        if session.embedding_model_id:
            embedding_model = db.query(EmbeddingModel).filter(
                EmbeddingModel.id == session.embedding_model_id
            ).first()
            if embedding_model:
                embedding_model_name = embedding_model.model_name
        
        # Collection name
        collection_name = f"session_{session_id}"
        
        # Check if collection exists before querying
        try:
            collections = chromadb_service.list_collections()
            if collection_name not in collections:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Collection '{collection_name}' does not exist. Please process PDF files first for this session."
                )
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            # If list_collections fails, try to query anyway (might be a different error)
            pass
        
        # Query using RAG with embedding model
        rag_result = rag_service.query(
            query_text=message,
            collection_name=collection_name,
            model_config=model_config,
            embedding_model_name=embedding_model_name,
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
    
    # Get list of existing collections
    try:
        existing_collections = chromadb_service.list_collections()
    except Exception:
        existing_collections = []
    
    result = []
    for s in sessions:
        collection_name = f"session_{s.id}"
        has_vectorized_data = collection_name in existing_collections
        
        result.append({
            "id": s.id,
            "name": s.session_name,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "message_count": len(s.chat_messages),
            "embedding_model_id": s.embedding_model_id,
            "hasVectorizedData": has_vectorized_data,
            "collection_name": collection_name,
            "embedding_model": {
                "id": s.embedding_model.id,
                "model_name": s.embedding_model.model_name,
                "display_name": s.embedding_model.display_name,
                "language": s.embedding_model.language
            } if s.embedding_model else None
        })
    
    return result

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
