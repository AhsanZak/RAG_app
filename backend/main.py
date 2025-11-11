from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import requests
import json
import aiofiles
import os
import uuid
import copy
from datetime import datetime
from sqlalchemy.orm import Session
from database import get_db
from models import User, LLMModel, ChatSession, ChatMessage as ChatMessageModel, LLMProvider, EmbeddingModel, DatabaseConnection, DatabaseSchema, DatabaseKnowledge
from config.embedding_models import AVAILABLE_EMBEDDINGS, get_default_embedding_model, get_embedding_model_info, list_embedding_models

KNOWLEDGE_UPLOAD_DIR = os.path.join(os.getcwd(), "uploads", "database_knowledge")
os.makedirs(KNOWLEDGE_UPLOAD_DIR, exist_ok=True)


def _serialize_knowledge_item(item: DatabaseKnowledge) -> Dict[str, Any]:
    return {
        "id": item.id,
        "connection_id": item.connection_id,
        "session_id": item.session_id,
        "user_id": item.user_id,
        "title": item.title,
        "description": item.description,
        "original_filename": item.original_filename,
        "file_type": item.file_type,
        "content_type": item.content_type,
        "file_url": f"/api/database/knowledge/{item.id}/download",
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "created_at_human": item.created_at.strftime("%Y-%m-%d %H:%M:%S") if item.created_at else None,
        "tags": item.tags or [],
    }

def _create_enriched_schema_chunks(schema_data: dict, schema_text: str) -> List[dict]:
    """
    Create enriched schema chunks based purely on actual schema structure
    No hardcoded mappings - relies on semantic understanding from embeddings
    """
    chunks = []
    
    if schema_data and 'tables' in schema_data:
        tables = schema_data['tables']
        
        # Create comprehensive table-level chunks with full context
        for table in tables:
            table_name = table.get('name', '')
            
            # Build comprehensive description using actual schema structure
            description_parts = []
            
            # Table header with full context
            description_parts.append(f"Database Table: {table_name}")
            
            # Schema context if available
            schema_name = table.get('schema') or schema_data.get('schema_name')
            if schema_name:
                description_parts.append(f"Schema: {schema_name}")
            
            # Detailed column information
            if table.get('columns'):
                description_parts.append("Table Columns:")
                for col in table['columns']:
                    col_name = col.get('name', '')
                    col_type = str(col.get('type', '')).lower()
                    nullable = col.get('nullable', True)
                    default = col.get('default')
                    
                    col_desc = f"  {col_name}: {col_type}"
                    if not nullable:
                        col_desc += " (NOT NULL)"
                    if default:
                        col_desc += f" DEFAULT {default}"
                    description_parts.append(col_desc)
            
            # Primary key constraints
            if table.get('primary_keys'):
                pk_columns = ', '.join(table['primary_keys'])
                description_parts.append(f"Primary Key: {pk_columns}")
            
            # Foreign key relationships - very important for understanding table connections
            if table.get('foreign_keys'):
                description_parts.append("Foreign Key Relationships:")
                for fk in table['foreign_keys']:
                    fk_cols = ', '.join(fk.get('constrained_columns', []))
                    ref_table = fk.get('referred_table', '')
                    ref_cols = ', '.join(fk.get('referred_columns', []))
                    description_parts.append(f"  {fk_cols} references {ref_table}({ref_cols})")
            
            # Indexes for query optimization hints
            if table.get('indexes'):
                description_parts.append("Indexes:")
                for idx in table['indexes']:
                    idx_cols = ', '.join(idx.get('columns', []))
                    unique = "UNIQUE " if idx.get('unique') else ""
                    description_parts.append(f"  {unique}Index: {idx_cols}")
            
            # Create comprehensive table chunk with rich context for embeddings
            table_chunk_text = '\n'.join(description_parts)
            
            # Add additional context that helps embedding models understand table purpose
            # by analyzing column names and relationships semantically
            column_names = [col.get('name', '') for col in table.get('columns', [])]
            if column_names:
                table_chunk_text += f"\nColumn Names: {', '.join(column_names)}"
            
            # Add relationship context
            if table.get('foreign_keys'):
                related_tables = set([fk.get('referred_table', '') for fk in table.get('foreign_keys', [])])
                if related_tables:
                    table_chunk_text += f"\nRelated Tables: {', '.join(sorted(related_tables))}"
            
            chunks.append({
                'text': table_chunk_text,
                'metadata': {
                    'table_name': table_name,
                    'chunk_type': 'table',
                    'column_count': len(table.get('columns', [])),
                    'has_foreign_keys': len(table.get('foreign_keys', [])) > 0
                }
            })
            
            # Create individual column chunks with full context for better granularity
            if table.get('columns'):
                for col in table['columns']:
                    col_name = col.get('name', '')
                    col_type = str(col.get('type', '')).lower()
                    
                    # Build column chunk with full table context
                    col_chunk_text = f"""Column Definition:
Table: {table_name}
Column Name: {col_name}
Data Type: {col_type}
"""
                    if not col.get('nullable', True):
                        col_chunk_text += "Constraint: NOT NULL\n"
                    if col.get('default'):
                        col_chunk_text += f"Default Value: {col.get('default')}\n"
                    
                    # Include relationship context if this column is part of a foreign key
                    if table.get('foreign_keys'):
                        for fk in table['foreign_keys']:
                            if col_name in fk.get('constrained_columns', []):
                                ref_table = fk.get('referred_table', '')
                                col_chunk_text += f"Foreign Key: References {ref_table}\n"
                    
                    # Include primary key context
                    if table.get('primary_keys') and col_name in table['primary_keys']:
                        col_chunk_text += "Primary Key: Yes\n"
                    
                    chunks.append({
                        'text': col_chunk_text,
                        'metadata': {
                            'table_name': table_name,
                            'column_name': col_name,
                            'chunk_type': 'column',
                            'data_type': col_type
                        }
                    })
            
            # Create relationship chunks to help understand table connections
            if table.get('foreign_keys'):
                for fk in table['foreign_keys']:
                    fk_cols = ', '.join(fk.get('constrained_columns', []))
                    ref_table = fk.get('referred_table', '')
                    ref_cols = ', '.join(fk.get('referred_columns', []))
                    
                    relationship_chunk = f"""Table Relationship:
Source Table: {table_name}
Source Columns: {fk_cols}
References Table: {ref_table}
Referenced Columns: {ref_cols}
Relationship Type: Foreign Key Constraint
"""
                    chunks.append({
                        'text': relationship_chunk,
                        'metadata': {
                            'table_name': table_name,
                            'referenced_table': ref_table,
                            'chunk_type': 'relationship'
                        }
                    })
    else:
        # Fallback to text-based chunking if schema_data is not available
        chunk_size = 1000
        overlap = 200
        lines = schema_text.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line)
            if current_length + line_length > chunk_size and current_chunk:
                chunks.append({
                    'text': '\n'.join(current_chunk),
                    'metadata': {
                        'chunk_type': 'text',
                        'chunk_index': len(chunks)
                    }
                })
                overlap_lines = current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                current_chunk = overlap_lines + [line]
                current_length = sum(len(l) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_length += line_length
        
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'metadata': {
                    'chunk_type': 'text',
                    'chunk_index': len(chunks)
                }
            })
    
    return chunks


def _build_condensed_chat_summary(
    chroma_service: "ChromaDBService",
    embed_service: "EmbeddingService",
    collection_name: str,
    user_query: str,
    embedding_model_name: str,
    fallback_messages: Optional[List["ChatMessageModel"]] = None,
    max_items: int = 5,
    max_chars: int = 1200
) -> Optional[str]:
    """
    Build a condensed conversational summary using vectorized chat history.
    Falls back to latest raw messages if vector retrieval fails.
    """
    summary_lines: List[str] = []

    try:
        query_embedding = embed_service.generate_embeddings(
            [user_query],
            model_name=embedding_model_name
        )
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        results = chroma_service.query_collection(
            collection_name=collection_name,
            query_embeddings=query_embedding,
            n_results=max_items,
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        if documents and len(documents) > 0:
            doc_list = documents[0]
            meta_list = metadatas[0] if metadatas else [{}] * len(doc_list)

            for doc, meta in zip(doc_list[:max_items], meta_list[:max_items]):
                if not isinstance(doc, str):
                    continue

                role = "Message"
                if isinstance(meta, dict):
                    role = meta.get("role", role)

                text = doc.strip()
                if ":" in text:
                    prefix, remainder = text.split(":", 1)
                    if not isinstance(meta, dict) or not meta.get("role"):
                        role = prefix.strip()
                    text = remainder.strip()

                text = " ".join(text.split())
                if len(text) > 200:
                    text = text[:197] + "..."

                summary_lines.append(f"{role.title()}: {text}")

    except Exception as e:
        print(f"[Chat] Unable to build vectorized conversation summary: {str(e)}")

    if not summary_lines and fallback_messages:
        recent = fallback_messages[-max_items:]
        for msg in recent:
            role = getattr(msg, "role", "message").title()
            text = getattr(msg, "message", "")
            text = " ".join((text or "").split())
            if len(text) > 200:
                text = text[:197] + "..."
            summary_lines.append(f"{role}: {text}")

    if not summary_lines:
        return None

    summary_text = "Relevant conversation context:\n" + "\n".join(f"- {line}" for line in summary_lines)
    if len(summary_text) > max_chars:
        summary_text = summary_text[: max_chars - 3] + "..."
    return summary_text

def _vectorize_schema_for_session(
    db_session: Session,
    session: ChatSession,
    schema_record: DatabaseSchema,
    connection: DatabaseConnection,
    embedding_model_name: Optional[str] = None
) -> bool:
    """Ensure a session has an up-to-date vector store for its schema."""
    if not schema_record or not schema_record.schema_data:
        return False

    schema_data = copy.deepcopy(schema_record.schema_data or {})
    schema_text = schema_record.schema_text or ""

    if embedding_model_name and not get_embedding_model_info(embedding_model_name):
        embedding_model_name = None
    if not embedding_model_name:
        # Try to read embedding model from schema metadata, fall back to default
        meta_block = None
        if isinstance(schema_data, dict):
            meta_block = schema_data.get("_meta") or schema_data.get("meta")
        if isinstance(meta_block, dict):
            embedding_model_name = meta_block.get("embedding_model")
    if embedding_model_name and not get_embedding_model_info(embedding_model_name):
        embedding_model_name = None
    if not embedding_model_name:
        embedding_model_name = get_default_embedding_model()

    collection_name = f"db_session_{session.id}"
    try:
        existing_collections = chromadb_service.list_collections()
    except Exception:
        existing_collections = []

    try:
        if collection_name in existing_collections:
            chromadb_service.delete_collection(collection_name)
    except Exception as cleanup_error:
        print(f"[SchemaVectors] Unable to clear existing collection '{collection_name}': {cleanup_error}")

    chunks = _create_enriched_schema_chunks(schema_data, schema_text)
    if not chunks:
        return False

    all_texts = [chunk.get("text") for chunk in chunks if chunk.get("text")]
    if not all_texts:
        return False

    embeddings = embedding_service.generate_embeddings(all_texts, model_name=embedding_model_name)

    all_metadatas = []
    all_ids = []
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text")
        if not text:
            continue
        metadata = dict(chunk.get("metadata") or {})
        metadata["session_id"] = str(session.id)
        metadata["connection_id"] = str(connection.id)
        metadata["chunk_index"] = idx
        all_metadatas.append(metadata)
        all_ids.append(f"schema_{connection.id}_{session.id}_{idx}")

    chromadb_service.add_documents(
        collection_name=collection_name,
        texts=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )

    # Update schema metadata and flags
    if isinstance(schema_data, dict):
        meta_block = schema_data.setdefault("_meta", {})
        if isinstance(meta_block, dict):
            meta_block["embedding_model"] = embedding_model_name
            meta_block["vectorized_at"] = datetime.utcnow().isoformat()
        schema_record.schema_data = schema_data

    schema_record.is_processed = 1
    schema_record.session_id = session.id
    schema_record.updated_at = datetime.utcnow()
    db_session.commit()

    return True

def _generate_result_analysis(
    llm_service: "LLMService",
    user_query: str,
    executed_sql: str,
    result_rows: List[Dict[str, Any]],
    model_config: Dict[str, Any],
    max_rows: int = 8
) -> Optional[str]:
    """Ask the LLM to analyze query results and produce a narrative summary."""
    if not result_rows:
        return None

    try:
        sample_rows = result_rows[:max_rows]
        rows_text = json.dumps(sample_rows, indent=2, default=str)

        prompt = (
            "You are a senior data analyst. Review the SQL query, the user's request, and the returned data.\n"
            "Provide a concise natural-language explanation that links the findings back to the user's intent.\n"
            "Highlight notable patterns, relationships (e.g., rental counts, actor popularity), or factors that explain the results.\n"
            "If the data alone is insufficient to fully answer the user's question, mention what evidence is available and what is missing.\n"
            "Do not invent facts beyond the dataset.\n\n"
            f"User Query:\n{user_query}\n\n"
            f"Executed SQL:\n{executed_sql}\n\n"
            f"Result Sample (JSON):\n{rows_text}\n\n"
            "Respond with one or two paragraphs summarizing the key insights, using field names from the data when relevant."
        )

        analysis = llm_service.generate_response(
            messages=[{"role": "user", "content": prompt}],
            model_config=model_config,
            temperature=0.5
        )
        return analysis.strip()
    except Exception as analysis_error:
        print(f"[ResultAnalysis] Unable to generate analysis: {analysis_error}")
        return None

from services.pdf_processor import PDFProcessor
from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService
from services.rag_service import RAGService
from services.llm_service import LLMService
from services.database_schema_service import DatabaseSchemaService
from services.db_agents import DatabaseAgentSystem
from services.sql_execution_service import SQLExecutionService
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

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
database_schema_service = DatabaseSchemaService()
sql_execution_service = SQLExecutionService()

# Verify embedding models configuration on startup
try:
    from config.embedding_models import AVAILABLE_EMBEDDINGS
    print(f"[OK] Embedding models configuration loaded: {len(AVAILABLE_EMBEDDINGS)} models available")
    print(f"   Models: {', '.join(list(AVAILABLE_EMBEDDINGS.keys())[:5])}...")
except Exception as e:
    print(f"[ERROR] Failed to load embedding models configuration: {str(e)}")
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
        print(f"[ERROR] Error getting embedding models: {str(e)}")
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
        print(f"[UPLOAD] filename={file.filename}, bytes_len={len(contents) if contents else 0}")
        
        # Extract text from PDF
        extraction_result = pdf_processor.extract_text_from_pdf(contents, file.filename)
        print(f"[UPLOAD] extraction_method={extraction_result.get('metadata',{}).get('extraction_method')}, total_chunks={extraction_result.get('total_chunks')}")
        # Determine PDF type
        total_chunks = extraction_result.get("total_chunks", 0)
        extraction_method = extraction_result.get("metadata", {}).get("extraction_method")
        pdf_type = "text_pdf" if total_chunks > 0 and extraction_method in ["pdfplumber", "pypdf2"] else (
            "image_pdf" if extraction_method == "ocr" or total_chunks == 0 else "unknown"
        )
        # Sample text for preview
        sample_text = "\n\n".join([c.get("text", "") for c in extraction_result.get("chunks", [])[:2]]).strip()
        
        # Store file info temporarily (in production, save to storage)
        response = {
            "success": True,
            "filename": file.filename,
            "chunks": extraction_result["chunks"],
            "metadata": extraction_result["metadata"],
            "total_chunks": extraction_result["total_chunks"],
            "pdf_type": pdf_type,
            "sample_text": sample_text
        }
        print(f"[UPLOAD] pdf_type={pdf_type}, sample_len={len(sample_text)}")
        return response
    except Exception as e:
        print(f"[UPLOAD][ERROR] {e}")
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
        print(f"[PROCESS] session_name={session_name}, files_count={len(files)}")
        
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
        print(f"[PROCESS] collection={collection_name}")
        
        all_texts = []
        all_metadatas = []
        all_ids = []
        
        # Process each file
        for file_data in files:
            chunks = file_data.get("chunks", [])
            filename = file_data.get("filename")
            print(f"[PROCESS] file={filename}, chunks={len(chunks)}")
            
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
        
        # Guard: ensure there is text to vectorize; if not, inform user clearly
        if len(all_texts) == 0:
            print("[PROCESS] No texts to vectorize even after OCR")
            raise HTTPException(status_code=400, detail="No text found to vectorize. The PDF may be image-only or protected. Try the OCR option or upload a different file.")

        # Detect language from combined text (first few chunks)
        try:
            sample_for_lang = " \n ".join(all_texts[:3])[:2000]
            detected_lang = detect(sample_for_lang) if sample_for_lang and sample_for_lang.strip() else "unknown"
        except Exception:
            detected_lang = "unknown"

        # Auto-select embedding model if not provided (based on detected language)
        if not embedding_model_name:
            if detected_lang in ["ar", "fa", "ur"]:
                embedding_model_name = "BAAI/bge-m3"  # strong multilingual including Arabic
            elif detected_lang in ["en"]:
                embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
            else:
                embedding_model_name = "intfloat/multilingual-e5-base"

        # Generate embeddings using selected embedding model
        print(f"[PROCESS] detected_lang={detected_lang}, model={embedding_model_name}, texts={len(all_texts)}")
        embeddings = embedding_service.generate_embeddings(all_texts, model_name=embedding_model_name)
        print(f"[PROCESS] embeddings_shape={(len(embeddings) if embeddings is not None else 0)}")
        
        # Add to ChromaDB
        print(f"[PROCESS] adding to chroma: ids={len(all_ids)}, metadatas={len(all_metadatas)}, texts={len(all_texts)}")
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
            "detected_language": detected_lang,
            "used_embedding_model": embedding_model_name,
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
    connection_id: int = None,  # Optional filter for database sessions
    db: Session = Depends(get_db)
):
    """Get all sessions for a user (PDF and Database sessions)"""
    query = db.query(ChatSession).filter(ChatSession.user_id == user_id)
    
    # If connection_id is provided, filter for database sessions only
    if connection_id:
        # Get sessions that have DatabaseSchema linked to this connection
        schemas = db.query(DatabaseSchema).filter(
            DatabaseSchema.connection_id == connection_id
        ).all()
        session_ids = [s.session_id for s in schemas if s.session_id]
        if session_ids:
            query = query.filter(ChatSession.id.in_(session_ids))
        else:
            # No sessions for this connection yet
            return []
    
    sessions = query.all()
    
    # Get list of existing collections
    try:
        existing_collections = chromadb_service.list_collections()
    except Exception:
        existing_collections = []
    
    result = []
    for s in sessions:
        # Check if this is a database session (has DatabaseSchema) or PDF session
        schema = db.query(DatabaseSchema).filter(DatabaseSchema.session_id == s.id).first()
        
        if schema:
            collection_name = f"db_session_{s.id}"
            session_type = "database"
            has_vectorized_data = collection_name in existing_collections

            if schema.is_processed and not has_vectorized_data:
                connection = db.query(DatabaseConnection).filter(DatabaseConnection.id == schema.connection_id).first()
                if connection:
                    try:
                        rebuilt = _vectorize_schema_for_session(
                            db,
                            s,
                            schema,
                            connection,
                            embedding_model_name=None
                        )
                        if rebuilt:
                            existing_collections = chromadb_service.list_collections()
                            has_vectorized_data = collection_name in existing_collections
                    except Exception as ensure_err:
                        print(f"[SESSIONS] Failed to rebuild vectors for session {s.id}: {ensure_err}")
        else:
            # PDF/Excel sessions use session_ prefix
            collection_name = f"session_{s.id}"
            session_type = "pdf"

            # Try to determine if this session belongs to Excel chat based on metadata
            if collection_name in existing_collections:
                try:
                    sample = chromadb_service.peek_collection(collection_name, n_results=3)
                    metadata_entries = sample.get("metadatas") or []

                    # Flatten metadata lists and keep dicts only
                    flattened_metadata = []
                    for item in metadata_entries:
                        if isinstance(item, dict):
                            flattened_metadata.append(item)
                        elif isinstance(item, list):
                            flattened_metadata.extend([meta for meta in item if isinstance(meta, dict)])

                    def is_excel_metadata(meta: Dict) -> bool:
                        filename = meta.get("filename", "")
                        sheet_name = meta.get("sheet_name")
                        file_type = meta.get("file_type")
                        return (
                            file_type == "excel"
                            or bool(sheet_name)
                            or (isinstance(filename, str) and filename.lower().endswith((".xlsx", ".xls", ".xlsm")))
                        )

                    if any(is_excel_metadata(meta) for meta in flattened_metadata):
                        session_type = "excel"
                except Exception as e:
                    print(f"[SESSIONS] Could not inspect collection '{collection_name}' metadata: {e}")

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
            "session_type": session_type,
            "connection_id": schema.connection_id if schema else None,
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

    response_payload = [
        {
            "id": m.id,
            "role": m.role,
            "message": m.message,
            "metadata": m.meta_data,
            "created_at": m.created_at.isoformat()
        }
        for m in messages
    ]

    print(
        "[SessionMessages]",
        f"session_id={session_id}",
        f"message_count={len(response_payload)}",
        f"message_ids={[msg['id'] for msg in response_payload]}"
    )
    print("[SessionMessages] payload=", response_payload)

    return response_payload

# Database Schema Chat Endpoints
@app.post("/api/database/connections")
async def create_database_connection(
    connection_data: dict,
    db: Session = Depends(get_db)
):
    """Create a new database connection"""
    try:
        connection = DatabaseConnection(
            user_id=connection_data.get("user_id", 1),
            name=connection_data.get("name"),
            database_type=connection_data.get("database_type"),
            host=connection_data.get("host"),
            port=connection_data.get("port"),
            database_name=connection_data.get("database_name"),
            username=connection_data.get("username"),
            password=connection_data.get("password"),
            connection_string=connection_data.get("connection_string"),
            schema_name=connection_data.get("schema_name"),
            is_active=connection_data.get("is_active", 1)
        )
        db.add(connection)
        db.commit()
        db.refresh(connection)
        return {
            "success": True,
            "connection": {
                "id": connection.id,
                "name": connection.name,
                "database_type": connection.database_type,
                "database_name": connection.database_name,
                "created_at": connection.created_at.isoformat()
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create connection: {str(e)}")

@app.get("/api/database/connections")
async def get_database_connections(
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Get all database connections for a user"""
    connections = db.query(DatabaseConnection).filter(
        DatabaseConnection.user_id == user_id,
        DatabaseConnection.is_active == 1
    ).all()
    
    return [
        {
            "id": c.id,
            "name": c.name,
            "database_type": c.database_type,
            "host": c.host,
            "port": c.port,
            "database_name": c.database_name,
            "username": c.username,
            "schema_name": c.schema_name,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat() if c.updated_at else None
        }
        for c in connections
    ]

@app.get("/api/database/connections/{connection_id}")
async def get_database_connection(
    connection_id: int,
    db: Session = Depends(get_db)
):
    """Get a single database connection"""
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id
    ).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    return {
        "id": connection.id,
        "name": connection.name,
        "database_type": connection.database_type,
        "host": connection.host,
        "port": connection.port,
        "database_name": connection.database_name,
        "username": connection.username,
        "schema_name": connection.schema_name,
        "created_at": connection.created_at.isoformat(),
        "updated_at": connection.updated_at.isoformat() if connection.updated_at else None
    }

@app.put("/api/database/connections/{connection_id}")
async def update_database_connection(
    connection_id: int,
    connection_data: dict,
    db: Session = Depends(get_db)
):
    """Update a database connection"""
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id
    ).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    try:
        for key, value in connection_data.items():
            if hasattr(connection, key):
                setattr(connection, key, value)
        
        db.commit()
        db.refresh(connection)
        
        return {
            "success": True,
            "connection": {
                "id": connection.id,
                "name": connection.name,
                "database_type": connection.database_type,
                "updated_at": connection.updated_at.isoformat() if connection.updated_at else None
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update connection: {str(e)}")

@app.delete("/api/database/connections/{connection_id}")
async def delete_database_connection(
    connection_id: int,
    db: Session = Depends(get_db)
):
    """Delete a database connection"""
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id
    ).first()
    
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    try:
        # Soft delete by setting is_active to 0
        connection.is_active = 0
        db.commit()
        return {"success": True, "message": "Connection deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete connection: {str(e)}")

@app.post("/api/database/test-connection")
async def test_database_connection(connection_data: dict):
    """Test database connection"""
    try:
        result = database_schema_service.test_connection(
            database_type=connection_data.get("database_type"),
            connection_string=connection_data.get("connection_string"),
            host=connection_data.get("host"),
            port=connection_data.get("port"),
            database_name=connection_data.get("database_name"),
            username=connection_data.get("username"),
            password=connection_data.get("password")
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "message": f"Connection test failed: {str(e)}"
        }

@app.post("/api/database/extract-schema")
async def extract_database_schema(
    connection_data: dict,
    db: Session = Depends(get_db)
):
    """Extract schema from database"""
    try:
        connection_id = connection_data.get("connection_id")
        
        # If connection_id is provided, get connection from database
        if connection_id:
            connection = db.query(DatabaseConnection).filter(
                DatabaseConnection.id == connection_id
            ).first()
            if not connection:
                raise HTTPException(status_code=404, detail="Connection not found")
            
            # Use connection data from database
            extract_result = database_schema_service.extract_schema(
                database_type=connection.database_type,
                connection_string=connection.connection_string,
                host=connection.host,
                port=connection.port,
                database_name=connection.database_name,
                username=connection.username,
                password=connection.password,
                schema_name=connection.schema_name
            )
        else:
            # Use provided connection data directly
            extract_result = database_schema_service.extract_schema(
                database_type=connection_data.get("database_type"),
                connection_string=connection_data.get("connection_string"),
                host=connection_data.get("host"),
                port=connection_data.get("port"),
                database_name=connection_data.get("database_name"),
                username=connection_data.get("username"),
                password=connection_data.get("password"),
                schema_name=connection_data.get("schema_name")
            )
        
        if not extract_result.get("success"):
            raise HTTPException(status_code=500, detail=extract_result.get("error", "Failed to extract schema"))
        
        return {
            "success": True,
            "schema_data": extract_result.get("schema_data"),
            "schema_text": extract_result.get("schema_text"),
            "metadata": extract_result.get("metadata")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract schema: {str(e)}")

@app.post("/api/database/save-schema")
async def save_database_schema(
    schema_data: dict,
    db: Session = Depends(get_db)
):
    """Save extracted schema to database"""
    try:
        connection_id = schema_data.get("connection_id")
        schema_info = schema_data.get("schema_data")
        schema_text = schema_data.get("schema_text")
        
        if not connection_id or not schema_info:
            raise HTTPException(status_code=400, detail="connection_id and schema_data are required")
        
        # Check if schema already exists for this connection
        existing_schema = db.query(DatabaseSchema).filter(
            DatabaseSchema.connection_id == connection_id
        ).first()
        
        if existing_schema:
            # Update existing schema
            existing_schema.schema_data = schema_info
            existing_schema.schema_text = schema_text
            existing_schema.is_processed = 0  # Reset processed flag if schema changed
            db.commit()
            db.refresh(existing_schema)
            schema_id = existing_schema.id
        else:
            # Create new schema
            new_schema = DatabaseSchema(
                connection_id=connection_id,
                schema_data=schema_info,
                schema_text=schema_text,
                is_processed=0
            )
            db.add(new_schema)
            db.commit()
            db.refresh(new_schema)
            schema_id = new_schema.id
        
        return {
            "success": True,
            "schema_id": schema_id,
            "message": "Schema saved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save schema: {str(e)}")

@app.post("/api/database/process-schema")
async def process_database_schema(
    session_data: dict,
    db: Session = Depends(get_db)
):
    """Process a database schema for a specific session and build its vector store."""
    try:
        connection_id = session_data.get("connection_id")
        user_id = session_data.get("user_id", 1)
        target_session_id = session_data.get("session_id")
        requested_llm_model_id = session_data.get("llm_model_id")

        if not connection_id:
            raise HTTPException(status_code=400, detail="connection_id is required")

        # Determine the base schema (session_id is NULL)
        base_schema = (
            db.query(DatabaseSchema)
            .filter(
                DatabaseSchema.connection_id == connection_id,
                DatabaseSchema.session_id.is_(None)
            )
            .order_by(DatabaseSchema.updated_at.desc(), DatabaseSchema.created_at.desc())
            .first()
        )

        if not base_schema:
            # Fallback for legacy data – use the most recent schema regardless of session binding
            base_schema = (
                db.query(DatabaseSchema)
                .filter(DatabaseSchema.connection_id == connection_id)
                .order_by(DatabaseSchema.updated_at.desc(), DatabaseSchema.created_at.desc())
                .first()
            )

        if not base_schema or not base_schema.schema_data:
            raise HTTPException(
                status_code=404,
                detail="Schema not found. Please extract and save the schema before processing."
            )

        schema_data = copy.deepcopy(base_schema.schema_data or {})
        schema_text = base_schema.schema_text or ""

        # Resolve embedding model to use
        embedding_model_name = session_data.get("embedding_model_name")
        if embedding_model_name and not get_embedding_model_info(embedding_model_name):
            embedding_model_name = None
        if not embedding_model_name:
            session_schema_meta = None
            if isinstance(base_schema.schema_data, dict):
                session_schema_meta = base_schema.schema_data.get("_meta")
            if isinstance(session_schema_meta, dict):
                stored_model = session_schema_meta.get("embedding_model")
                if stored_model and get_embedding_model_info(stored_model):
                    embedding_model_name = stored_model
            if not embedding_model_name:
                embedding_model_name = get_default_embedding_model()

        # Resolve LLM model ID
        llm_model_id = requested_llm_model_id
        session: Optional[ChatSession] = None
        session_created = False

        if target_session_id:
            session = db.query(ChatSession).filter(ChatSession.id == target_session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            if not llm_model_id:
                llm_model_id = session.llm_model_id
        else:
            session_name = session_data.get(
                "session_name",
                f"Database Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            session = ChatSession(
                user_id=user_id,
                session_name=session_name,
                llm_model_id=session_data.get("llm_model_id", 1),
                embedding_model_id=None,
                created_at=datetime.utcnow()
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_created = True
            if not llm_model_id:
                llm_model_id = session.llm_model_id

        if not llm_model_id:
            default_llm = (
                db.query(LLMModel)
                .filter(LLMModel.is_active == 1)
                .order_by(LLMModel.id)
                .first()
            ) or db.query(LLMModel).order_by(LLMModel.id).first()
            if not default_llm:
                raise HTTPException(status_code=400, detail="No LLM models are configured. Add an LLM model first.")
            llm_model_id = default_llm.id
            if session_created:
                session.llm_model_id = llm_model_id
                db.commit()

        if session.llm_model_id != llm_model_id:
            session.llm_model_id = llm_model_id
            db.commit()

        # Ensure there is a schema record bound to this session
        session_schema = (
            db.query(DatabaseSchema)
            .filter(
                DatabaseSchema.connection_id == connection_id,
                DatabaseSchema.session_id == session.id
            )
            .first()
        )

        if not session_schema:
            session_schema = DatabaseSchema(
                connection_id=connection_id,
                session_id=session.id,
                schema_data=schema_data,
                schema_text=schema_text,
                is_processed=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(session_schema)
        else:
            session_schema.schema_data = schema_data
            session_schema.schema_text = schema_text
            session_schema.is_processed = 0
            session_schema.updated_at = datetime.utcnow()
        db.commit()

        collection_name = f"db_session_{session.id}"

        # Clear existing vectors for this session (if any)
        try:
            existing_collections = chromadb_service.list_collections()
            if collection_name in existing_collections:
                chromadb_service.delete_collection(collection_name)
        except Exception as cleanup_error:
            print(f"[SchemaProcess] Unable to clear existing collection '{collection_name}': {cleanup_error}")

        chunks = _create_enriched_schema_chunks(schema_data, schema_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No schema content to vectorize")

        all_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_service.generate_embeddings(all_texts, model_name=embedding_model_name)

        all_metadatas = []
        all_ids = []
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.get("metadata") or {})
            metadata["session_id"] = str(session.id)
            metadata["connection_id"] = str(connection_id)
            metadata["chunk_index"] = idx
            all_metadatas.append(metadata)
            all_ids.append(f"schema_{connection_id}_{session.id}_{idx}")

        chromadb_service.add_documents(
            collection_name=collection_name,
            texts=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )

        session_schema.is_processed = 1
        session_schema.updated_at = datetime.utcnow()
        db.commit()

        preview_text = schema_text[:2000] if schema_text else "Schema processed successfully"

        if preview_text and preview_text.strip():
            summary_message = ChatMessageModel(
                session_id=session.id,
                role="assistant",
                message=preview_text,
                meta_data={
                    "summary": True,
                    "schema_processed": True,
                    "timestamp": datetime.utcnow().isoformat()
                },
                created_at=datetime.utcnow()
            )
            db.add(summary_message)
            db.commit()

        return {
            "success": True,
            "session_id": session.id,
            "collection_name": collection_name,
            "total_chunks": len(chunks),
            "message": "Schema processed and vectorized successfully",
            "used_embedding_model": embedding_model_name,
            "session_created": session_created,
            "session": {
                "id": session.id,
                "name": session.session_name,
                "created_at": session.created_at.isoformat(),
                "connection_id": connection_id,
                "hasVectorizedData": True,
                "session_type": "database"
            },
            "preview": {
                "summary": preview_text
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process schema: {str(e)}")


@app.post("/api/database/sessions")
async def create_database_session(
    session_data: dict,
    db: Session = Depends(get_db)
):
    """Create a brand new chat session for a database connection (no vectors yet)."""
    try:
        connection_id = session_data.get("connection_id")
        if not connection_id:
            raise HTTPException(status_code=400, detail="connection_id is required")

        connection = db.query(DatabaseConnection).filter(DatabaseConnection.id == connection_id).first()
        if not connection:
            raise HTTPException(status_code=404, detail="Database connection not found")

        # Locate base schema (session_id is NULL)
        base_schema = (
            db.query(DatabaseSchema)
            .filter(
                DatabaseSchema.connection_id == connection_id,
                DatabaseSchema.session_id.is_(None)
            )
            .order_by(DatabaseSchema.updated_at.desc(), DatabaseSchema.created_at.desc())
            .first()
        )

        if not base_schema:
            base_schema = (
                db.query(DatabaseSchema)
                .filter(DatabaseSchema.connection_id == connection_id)
                .order_by(DatabaseSchema.updated_at.desc(), DatabaseSchema.created_at.desc())
                .first()
            )

        if not base_schema or not base_schema.schema_data:
            raise HTTPException(
                status_code=400,
                detail="Extract the schema for this connection before creating a chat session."
            )

        user_id = session_data.get("user_id", connection.user_id if hasattr(connection, "user_id") else 1)
        llm_model_id = session_data.get("llm_model_id")

        if not llm_model_id:
            default_llm = (
                db.query(LLMModel)
                .filter(LLMModel.is_active == 1)
                .order_by(LLMModel.id)
                .first()
            ) or db.query(LLMModel).order_by(LLMModel.id).first()
            if not default_llm:
                raise HTTPException(status_code=400, detail="No LLM models are configured. Add an LLM model first.")
            llm_model_id = default_llm.id

        session_name = session_data.get(
            "session_name",
            f"{connection.name} Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        new_session = ChatSession(
            user_id=user_id,
            session_name=session_name,
            llm_model_id=llm_model_id,
            embedding_model_id=None,
            created_at=datetime.utcnow()
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)

        # Create a schema record bound to this session (marked as not processed)
        session_schema = DatabaseSchema(
            connection_id=connection_id,
            session_id=new_session.id,
            schema_data=copy.deepcopy(base_schema.schema_data or {}),
            schema_text=base_schema.schema_text or "",
            is_processed=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(session_schema)
        db.commit()

        return {
            "success": True,
            "message": "New database chat session created. Process the schema before chatting.",
            "session": {
                "id": new_session.id,
                "name": new_session.session_name,
                "created_at": new_session.created_at.isoformat(),
                "connection_id": connection_id,
                "hasVectorizedData": False,
                "session_type": "database"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create database session: {str(e)}")

@app.post("/api/database/chat")
async def database_chat(
    chat_data: dict,
    db: Session = Depends(get_db)
):
    """Chat with database using agent system"""
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
            "base_url": model.base_url,
            "api_key": model.api_key if hasattr(model, 'api_key') else None
        }
        
        # Locate the schema for this session
        schema = (
            db.query(DatabaseSchema)
            .filter(DatabaseSchema.session_id == session_id)
            .first()
        )
        if not schema:
            raise HTTPException(status_code=404, detail="Schema not found for this session")

        connection = db.query(DatabaseConnection).filter(
            DatabaseConnection.id == schema.connection_id
        ).first()
        if not connection:
            raise HTTPException(status_code=404, detail="Database connection not found")
        connection_id = connection.id

        if not schema.is_processed:
            raise HTTPException(
                status_code=400,
                detail="This session has not been processed yet. Process & vectorize the schema before chatting."
            )
        
        # Ensure the session has a vector store; rebuild if needed
        collection_name = f"db_session_{session_id}"
        try:
            existing_collections = chromadb_service.list_collections()
        except Exception:
            existing_collections = []

        if schema.is_processed and collection_name not in existing_collections:
            try:
                rebuilt = _vectorize_schema_for_session(
                    db,
                    session,
                    schema,
                    connection,
                    embedding_model_name=None
                )
                if rebuilt:
                    existing_collections = chromadb_service.list_collections()
            except Exception as rebuild_error:
                print(f"[Chat] Failed to rebuild vectors for session {session_id}: {rebuild_error}")

        if collection_name not in existing_collections:
            raise HTTPException(
                status_code=400,
                detail="This session has not been processed yet. Process the schema before chatting."
            )

        # Create agent system with model config and services for semantic search
        agent_llm_service = LLMService()
        # Create agent system with model config and embedding/chromadb services
        db_agent_system = DatabaseAgentSystem(
            agent_llm_service, 
            database_schema_service,
            embedding_service=embedding_service,
            chromadb_service=chromadb_service,
            model_config=model_config
        )
        
        # Get chat history for context
        previous_messages = db.query(ChatMessageModel).filter(
            ChatMessageModel.session_id == session_id
        ).order_by(ChatMessageModel.created_at).limit(10).all()
        
        # Get embedding model name (use default if not specified)
        embedding_model_name = get_default_embedding_model()
        
        # Build condensed conversation summary (prefer vectorized history)
        conversation_summary = None
        try:
            collections = chromadb_service.list_collections()
            if collection_name in collections:
                conversation_summary = _build_condensed_chat_summary(
                    chromadb_service,
                    embedding_service,
                    collection_name,
                    message,
                    embedding_model_name,
                    fallback_messages=previous_messages
                )
        except Exception as e:
            print(f"[Chat] Unable to build conversation summary from vectors: {str(e)}")
        
        if not conversation_summary and previous_messages:
            fallback_lines = []
            for msg in previous_messages[-5:]:
                role = msg.role.title()
                text = (msg.message or "").strip().replace("\n", " ")
                if len(text) > 200:
                    text = text[:197] + "..."
                fallback_lines.append(f"{role}: {text}")
            if fallback_lines:
                conversation_summary = "Recent conversation highlights:\n" + "\n".join(f"- {line}" for line in fallback_lines)
        
        # Process query through agent system with semantic search
        agent_result = db_agent_system.process_query(
            user_query=message,
            schema_details=schema.schema_data,
            schema_text=schema.schema_text or "",
            database_type=connection.database_type,
            collection_name=collection_name,  # For semantic search
            embedding_model_name=embedding_model_name,  # For semantic search
            connection_string=connection.connection_string,
            host=connection.host,
            port=connection.port,
            database_name=connection.database_name,
            username=connection.username,
            password=connection.password,
            conversation_summary=conversation_summary,
            previous_messages=[
                {"role": m.role, "content": m.message}
                for m in previous_messages
            ] if previous_messages else None
        )
        
        # Prepare response
        response_text = ""
        sql_executed = False
        query_result = None
        relevant_schema_parts = agent_result.get('relevant_schema_parts', [])
        metadata = {
            "intent": agent_result.get('intent', {}),
            "requires_sql": agent_result.get('requires_sql', False),
            "sql": agent_result.get('sql'),
            "sql_validation": agent_result.get('sql_validation', {}),
            "relevant_schema_parts_count": len(relevant_schema_parts),
            "used_semantic_search": len(relevant_schema_parts) > 0
        }
        table_selection_info = agent_result.get('table_selection') or {}
        metadata["table_selection"] = {
            "enabled": bool(table_selection_info),
            "selected_tables": table_selection_info.get('selected_tables', []),
            "confidence": table_selection_info.get('confidence'),
            "method": table_selection_info.get('selection_method')
        }
        
        # If SQL should be executed
        if agent_result.get('should_execute') and agent_result.get('sql'):
            # Execute SQL
            print(f"[DEBUG] Executing SQL: {agent_result['sql']}")
            sql_result = sql_execution_service.execute_query(
                sql=agent_result['sql'],
                database_type=connection.database_type,
                connection_string=connection.connection_string,
                host=connection.host,
                port=connection.port,
                database_name=connection.database_name,
                username=connection.username,
                password=connection.password
            )
            
            print(f"[DEBUG] SQL execution result - success: {sql_result.get('success')}, row_count: {sql_result.get('row_count', 0)}, has_data: {bool(sql_result.get('data'))}")
            
            sql_executed = True
            query_result = sql_result
            
            if sql_result.get('success'):
                # Format SQL results directly - show actual query results
                try:
                    query_data = sql_result.get('data', [])
                    row_count = sql_result.get('row_count', 0)
                    columns = sql_result.get('columns', [])
                    
                    # Build response with actual query results
                    if row_count > 0:
                        # Keep response concise - frontend will display the table
                        response_text = f"I found **{row_count} result(s)** for your query. The query results are displayed in the table below."
                    else:
                        # No results found
                        response_text = f"I executed your query, but it returned **no results**.\n\n"
                        if agent_result.get('sql'):
                            response_text += f"**Executed Query:**\n```sql\n{agent_result['sql']}\n```\n\n"
                        response_text += "The query completed successfully, but there are no records matching your criteria."
                    
                    # Store query results in metadata for frontend display
                    if row_count > 0:
                        metadata['query_result'] = {
                            'row_count': row_count,
                            'columns': columns,
                            'data': query_data[:100] if len(query_data) > 100 else query_data  # Store up to 100 rows
                        }
                    else:
                        metadata['query_result'] = {
                            'row_count': 0,
                            'columns': columns,
                            'data': []
                        }
                    analysis_text = _generate_result_analysis(
                        agent_llm_service,
                        message,
                        agent_result['sql'],
                        metadata['query_result']['data'],
                        model_config
                    )
                    if analysis_text:
                        metadata['analysis'] = analysis_text
                        response_text += "\n\n" + analysis_text
                except Exception as e:
                    # If formatting fails, provide a simple response with data
                    import logging
                    logging.error(f"Error formatting SQL results: {str(e)}")
                    row_count = sql_result.get('row_count', 0)
                    query_data = sql_result.get('data', [])
                    columns = sql_result.get('columns', [])
                    
                    if row_count > 0:
                        response_text = f"I successfully executed your query and found **{row_count} result(s)**.\n\n"
                        if agent_result.get('sql'):
                            response_text += f"**Executed Query:**\n```sql\n{agent_result['sql']}\n```\n\n"
                        # Include raw data even if formatting failed
                        response_text += "**Query Results:**\n```json\n" + json.dumps(query_data[:10], indent=2, default=str) + "\n```"
                    else:
                        response_text = "I executed your query successfully, but it returned **no results**.\n\n"
                        if agent_result.get('sql'):
                            response_text += f"**Executed Query:**\n```sql\n{agent_result['sql']}\n```"
                    
                    # Still store results in metadata
                    metadata['query_result'] = {
                        'row_count': row_count,
                        'columns': columns,
                        'data': query_data[:100] if len(query_data) > 100 else query_data
                    }
                    analysis_text = _generate_result_analysis(
                        agent_llm_service,
                        message,
                        agent_result['sql'],
                        metadata['query_result']['data'],
                        model_config
                    )
                    if analysis_text:
                        metadata['analysis'] = analysis_text
                        response_text += "\n\n" + analysis_text
            else:
                # SQL execution failed - provide user-friendly error message
                error_msg = sql_result.get('error', 'Unknown error occurred')
                error_type = sql_result.get('error_type', 'execution_error')
                
                # Don't expose raw SQL errors to frontend
                # Generate a helpful response instead
                if 'table' in error_msg.lower() or 'column' in error_msg.lower():
                    response_text = "I couldn't execute the query because some tables or columns referenced don't exist in the database. Could you rephrase your question or be more specific about what you're looking for?"
                elif 'syntax' in error_msg.lower() or 'invalid' in error_msg.lower():
                    response_text = "There was an issue with the generated SQL query. Let me try to understand your question better - could you rephrase it?"
                elif 'permission' in error_msg.lower() or 'access' in error_msg.lower():
                    response_text = "I don't have permission to execute this type of query on the database. Please check your database connection permissions."
                elif 'connection' in error_msg.lower():
                    response_text = "I couldn't connect to the database to execute the query. Please check your database connection settings."
                else:
                    # Generic user-friendly error message
                    response_text = "I encountered an issue executing the query. Could you try rephrasing your question or provide more specific details about what you need?"
                
                # Store error info in metadata (for debugging, but not exposed to user)
                metadata['query_error'] = {
                    'type': error_type,
                    'message': error_msg  # For backend logging only
                }
                metadata['sql'] = None  # Don't expose SQL if it failed
        
        elif agent_result.get('chat_response'):
            # Use chat response from agent
            response_text = agent_result['chat_response']
        else:
            # Fallback: Use RAG with schema context
            try:
                collections = chromadb_service.list_collections()
                if collection_name in collections:
                    rag_result = rag_service.query(
                        query_text=message,
                        collection_name=collection_name,
                        model_config=model_config,
                        embedding_model_name=embedding_model_name,
                        n_results=5
                    )
                    response_text = rag_result["response"]
                    metadata['sources'] = rag_result.get("sources", [])
                else:
                    response_text = "I couldn't process your query. Please try rephrasing it or make sure the schema has been processed first."
            except Exception as e:
                import logging
                logging.error(f"RAG query error: {str(e)}")
                response_text = "I'm having trouble processing your query right now. Could you try rephrasing it or check if the database schema has been processed?"
        
        # Save user message
        user_message = ChatMessageModel(
            session_id=session_id,
            role="user",
            message=message,
            created_at=datetime.utcnow()
        )
        db.add(user_message)
        
        # Save assistant message with metadata
        assistant_message = ChatMessageModel(
            session_id=session_id,
            role="assistant",
            message=response_text,
            meta_data=metadata,
            created_at=datetime.utcnow()
        )
        db.add(assistant_message)
        db.commit()
        
        # Vectorize chat history (add new messages to collection)
        collection_name = f"db_session_{session_id}"
        try:
            # Get all messages for this session
            all_messages = db.query(ChatMessageModel).filter(
                ChatMessageModel.session_id == session_id
            ).order_by(ChatMessageModel.created_at).all()
            
            # Prepare texts for vectorization (last N messages)
            recent_messages = all_messages[-10:]  # Last 10 messages
            chat_texts = []
            chat_metadatas = []
            chat_ids = []
            
            for msg in recent_messages:
                text = f"{msg.role}: {msg.message}"
                chat_texts.append(text)
                chat_metadatas.append({
                    "session_id": str(session_id),
                    "message_id": str(msg.id),
                    "role": msg.role,
                    "timestamp": msg.created_at.isoformat()
                })
                chat_ids.append(f"msg_{msg.id}")
            
            if chat_texts:
                # Generate embeddings
                embeddings = embedding_service.generate_embeddings(
                    chat_texts, 
                    model_name=get_default_embedding_model()
                )
                
                # Add to ChromaDB (update collection)
                try:
                    # Get existing collection or create
                    collections = chromadb_service.list_collections()
                    if collection_name not in collections:
                        # Create collection if doesn't exist
                        chromadb_service.create_collection(collection_name)
                    
                    # Add messages
                    chromadb_service.add_documents(
                        collection_name=collection_name,
                        texts=chat_texts,
                        embeddings=embeddings,
                        metadatas=chat_metadatas,
                        ids=chat_ids
                    )
                except Exception as e:
                    print(f"[Chat] Error vectorizing chat history: {str(e)}")
        except Exception as e:
            print(f"[Chat] Error in vectorization process: {str(e)}")
        
        # Prepare response - only include SQL if execution was successful
        response_data = {
            "response": response_text,
            "message_id": assistant_message.id,
            "session_id": session_id,
            "requires_sql": agent_result.get('requires_sql', False),
            "sql_executed": sql_executed,
            "metadata": {
                "intent": metadata.get('intent', {}),
                "used_semantic_search": metadata.get('used_semantic_search', False),
                "relevant_schema_parts_count": metadata.get('relevant_schema_parts_count', 0),
                "table_selection": metadata.get('table_selection')
            }
        }
        if table_selection_info:
            response_data["table_selection"] = {
                "selected_tables": table_selection_info.get('selected_tables', []),
                "confidence": table_selection_info.get('confidence'),
                "method": table_selection_info.get('selection_method')
            }
        
        # Only include SQL in response if execution was successful
        if sql_executed and query_result and query_result.get('success'):
            response_data["sql"] = agent_result.get('sql')
            # Get query data - check if it exists and is not None
            query_data = query_result.get('data')
            row_count = query_result.get('row_count', 0)
            columns = query_result.get('columns', [])
            
            # Debug logging
            print(f"[DEBUG] SQL executed successfully. Row count: {row_count}, Data type: {type(query_data)}, Data length: {len(query_data) if query_data else 0}")
            if query_data and len(query_data) > 0:
                print(f"[DEBUG] First row sample: {query_data[0] if query_data else 'None'}")
            
            # Always include query_result in response when SQL executes successfully
            if query_data is not None:
                # Return all data (up to 100 rows for display, frontend can paginate)
                response_data["query_result"] = query_data[:100] if len(query_data) > 100 else query_data
                response_data["query_result_count"] = row_count
                # Extract columns from data if not provided
                if columns and len(columns) > 0:
                    response_data["query_result_columns"] = columns
                elif query_data and len(query_data) > 0 and isinstance(query_data[0], dict):
                    response_data["query_result_columns"] = list(query_data[0].keys())
                else:
                    response_data["query_result_columns"] = []
                if metadata.get('analysis'):
                    response_data["analysis"] = metadata['analysis']
            else:
                # No data returned, but query was successful (e.g., INSERT/UPDATE/DELETE)
                response_data["query_result"] = []
                response_data["query_result_count"] = row_count
                response_data["query_result_columns"] = columns if columns else []
            
            print(f"[DEBUG] Response includes query_result: {len(response_data.get('query_result', []))} rows")
        elif agent_result.get('sql') and not sql_executed:
            # SQL was generated but not executed (validation failed)
            # Don't expose the SQL to frontend
            response_data["sql"] = None
            response_data["query_result"] = []
            response_data["query_result_count"] = 0
            response_data["query_result_columns"] = []
        else:
            # No SQL generated or errors occurred
            response_data["sql"] = None
            response_data["query_result"] = []
            response_data["query_result_count"] = 0
            response_data["query_result_columns"] = []
        
        return response_data
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/database/schemas/{connection_id}")
async def get_database_schema(
    connection_id: int,
    db: Session = Depends(get_db)
):
    """Get saved schema for a connection"""
    schema = db.query(DatabaseSchema).filter(
        DatabaseSchema.connection_id == connection_id
    ).first()
    
    if not schema:
        raise HTTPException(status_code=404, detail="Schema not found")
    
    return {
        "id": schema.id,
        "connection_id": schema.connection_id,
        "session_id": schema.session_id,
        "schema_data": schema.schema_data,
        "schema_text": schema.schema_text,
        "is_processed": bool(schema.is_processed),
        "created_at": schema.created_at.isoformat() if schema.created_at else None,
        "updated_at": schema.updated_at.isoformat() if schema.updated_at else None
    }


@app.post("/api/database/knowledge")
async def upload_database_knowledge(
    connection_id: Optional[int] = Form(None),
    session_id: Optional[int] = Form(None),
    user_id: int = Form(1),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload knowledge document for database chat"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")
    
    filename = file.filename
    ext = os.path.splitext(filename)[1] or ""
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(KNOWLEDGE_UPLOAD_DIR, unique_name)
    
    try:
        async with aiofiles.open(save_path, "wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await out_file.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store file: {str(e)}")
    
    knowledge = DatabaseKnowledge(
        connection_id=connection_id,
        session_id=session_id,
        user_id=user_id,
        title=title or os.path.splitext(filename)[0],
        description=description,
        original_filename=filename,
        file_path=save_path,
        file_type=ext.lstrip(".").lower(),
        content_type=file.content_type,
        tags=None,
    )
    
    db.add(knowledge)
    db.commit()
    db.refresh(knowledge)
    
    return _serialize_knowledge_item(knowledge)


@app.get("/api/database/knowledge")
async def list_database_knowledge(
    connection_id: Optional[int] = None,
    session_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """List uploaded knowledge documents"""
    query = db.query(DatabaseKnowledge)
    if connection_id is not None:
        query = query.filter(DatabaseKnowledge.connection_id == connection_id)
    if session_id is not None:
        query = query.filter(DatabaseKnowledge.session_id == session_id)
    items = query.order_by(DatabaseKnowledge.created_at.desc()).all()
    return [_serialize_knowledge_item(item) for item in items]


@app.get("/api/database/knowledge/{knowledge_id}/download")
async def download_database_knowledge(
    knowledge_id: int,
    db: Session = Depends(get_db)
):
    """Download a knowledge document"""
    knowledge = db.query(DatabaseKnowledge).filter(DatabaseKnowledge.id == knowledge_id).first()
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
    
    if not os.path.exists(knowledge.file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    filename = knowledge.original_filename or os.path.basename(knowledge.file_path)
    media_type = knowledge.content_type or "application/octet-stream"
    return FileResponse(knowledge.file_path, media_type=media_type, filename=filename)


@app.delete("/api/database/knowledge/{knowledge_id}")
async def delete_database_knowledge(
    knowledge_id: int,
    db: Session = Depends(get_db)
):
    """Delete knowledge document"""
    knowledge = db.query(DatabaseKnowledge).filter(DatabaseKnowledge.id == knowledge_id).first()
    if not knowledge:
        raise HTTPException(status_code=404, detail="Knowledge document not found")
    
    file_path = knowledge.file_path
    
    db.delete(knowledge)
    db.commit()
    
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass
    
    return {"success": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
