from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import requests
import json
from sqlalchemy.orm import Session
from database import get_db
from models import User, LLMModel, ChatSession, ChatMessage as ChatMessageModel, LLMProvider

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
