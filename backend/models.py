from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class LLMProvider(str, enum.Enum):
    """Enum for LLM provider types"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMModel(Base):
    """Model for storing LLM configurations from various sources"""
    __tablename__ = "llm_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, unique=True, index=True)
    provider = Column(SQLEnum(LLMProvider), nullable=False)
    base_url = Column(String(500))  # API endpoint URL
    api_key = Column(String(500))  # Encrypted API key
    description = Column(Text)
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with chat sessions
    chat_sessions = relationship("ChatSession", back_populates="llm_model")
    
    def __repr__(self):
        return f"<LLMModel(name='{self.model_name}', provider='{self.provider}')>"


class EmbeddingModel(Base):
    """Model for storing embedding model configurations"""
    __tablename__ = "embedding_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(200), nullable=False, unique=True, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    language = Column(String(50))  # 'english', 'arabic', 'multilingual', etc.
    model_path = Column(String(500))  # HuggingFace model path or local path
    dimension = Column(Integer)  # Embedding dimension
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    is_downloaded = Column(Integer, default=0)  # 1 if downloaded, 0 if not
    download_progress = Column(Integer, default=0)  # Download progress percentage
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with chat sessions
    chat_sessions = relationship("ChatSession", back_populates="embedding_model")
    
    def __repr__(self):
        return f"<EmbeddingModel(name='{self.model_name}', language='{self.language}')>"


class User(Base):
    """Model for storing users"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    email = Column(String(100), nullable=False, unique=True, index=True)
    password_hash = Column(String(255))  # Hashed password
    full_name = Column(String(100))
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with chat sessions
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class ChatSession(Base):
    """Model for storing chat sessions"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_name = Column(String(200))  # Optional session name
    llm_model_id = Column(Integer, ForeignKey("llm_models.id"), nullable=False)
    embedding_model_id = Column(Integer, ForeignKey("embedding_models.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with user
    user = relationship("User", back_populates="chat_sessions")
    
    # Relationship with LLM model
    llm_model = relationship("LLMModel", back_populates="chat_sessions")
    
    # Relationship with embedding model
    embedding_model = relationship("EmbeddingModel", back_populates="chat_sessions")
    
    # Relationship with chat messages
    chat_messages = relationship("ChatMessage", back_populates="chat_session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, session_name='{self.session_name}')>"


class ChatMessage(Base):
    """Model for storing individual chat messages"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    message = Column(Text, nullable=False)
    meta_data = Column(JSON)  # Store additional metadata like sources, tokens, etc.
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship with chat session
    chat_session = relationship("ChatSession", back_populates="chat_messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, role='{self.role}')>"


class Document(Base):
    """Model for storing documents/news articles for RAG"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    document_type = Column(String(50))  # 'news', 'article', 'blog', etc.
    source = Column(String(500))  # Source URL or publication
    meta_data = Column(JSON)  # Additional metadata
    is_indexed = Column(Integer, default=0)  # 1 if indexed in vector DB
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}')>"


class DatabaseConnection(Base):
    """Model for storing database connection configurations"""
    __tablename__ = "database_connections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(200), nullable=False)  # Connection name/alias
    database_type = Column(String(50), nullable=False)  # 'postgresql', 'mysql', 'sqlite', 'oracle', 'mssql', etc.
    host = Column(String(200))
    port = Column(Integer)
    database_name = Column(String(200))
    username = Column(String(200))
    password = Column(String(500))  # Should be encrypted in production
    connection_string = Column(Text)  # Full connection string if preferred
    schema_name = Column(String(200))  # Specific schema to extract (if applicable)
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with user
    user = relationship("User")
    
    def __repr__(self):
        return f"<DatabaseConnection(id={self.id}, name='{self.name}', type='{self.database_type}')>"


class DatabaseSchema(Base):
    """Model for storing extracted database schema information"""
    __tablename__ = "database_schemas"
    
    id = Column(Integer, primary_key=True, index=True)
    connection_id = Column(Integer, ForeignKey("database_connections.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=True)  # Associated chat session
    schema_data = Column(JSON, nullable=False)  # Full schema information (tables, columns, relationships, etc.)
    schema_text = Column(Text)  # Human-readable text representation
    is_processed = Column(Integer, default=0)  # 1 if processed and vectorized, 0 if not
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    connection = relationship("DatabaseConnection")
    session = relationship("ChatSession")
    
    def __repr__(self):
        return f"<DatabaseSchema(id={self.id}, connection_id={self.connection_id})>"
