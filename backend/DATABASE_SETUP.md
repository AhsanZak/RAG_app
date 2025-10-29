# Database Setup Guide

This guide will help you set up the database for the RAG Chat Application.

## 📊 Database Schema

The application uses SQLite database with the following tables:

### 1. **users**
- Stores user information
- Fields: id, username, email, password_hash, full_name, is_active, created_at, updated_at

### 2. **llm_models**
- Stores LLM configurations from various sources (OpenAI, Ollama, etc.)
- Fields: id, model_name, provider, base_url, api_key, description, is_active, created_at, updated_at
- Provider types: OPENAI, OLLAMA, AZURE, ANTHROPIC, LOCAL

### 3. **chat_sessions**
- Stores chat session information
- Fields: id, user_id, session_name, llm_model_id, created_at, updated_at
- Links users to their chat sessions and LLM models

### 4. **chat_messages**
- Stores individual chat messages
- Fields: id, session_id, role, message, metadata, created_at
- role can be 'user' or 'assistant'

### 5. **documents**
- Stores documents/news articles for RAG
- Fields: id, title, content, document_type, source, metadata, is_indexed, created_at, updated_at

## 🚀 Setup Instructions

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Initialize Database

```bash
python init_db.py
```

This will:
- Create the database file (`rag_chat.db`)
- Create all necessary tables
- Add default LLM models (GPT-4, GPT-3.5, LLaMA 2, Mistral, Mixtral)
- Create a default admin user

### Step 3: Verify Database

Check if the database was created successfully:

```bash
# On Windows
dir rag_chat.db

# On macOS/Linux
ls -lh rag_chat.db
```

## 📋 Default Data

### Default LLM Models

The initialization script creates the following models:

1. **OpenAI Models**
   - gpt-4-turbo-preview
   - gpt-3.5-turbo

2. **Ollama Models (Local)**
   - llama2
   - mistral
   - mixtral

### Default User

- **Username**: admin
- **Email**: admin@ragchat.com
- **Note**: Update the password_hash with proper hashing in production

## 🔧 Configuration

### Database URL

The database location can be changed using an environment variable:

```bash
export DATABASE_URL="sqlite:///./path/to/your/database.db"
```

Or update the `DATABASE_URL` in `database.py`:

```python
DATABASE_URL = "sqlite:///./your_database.db"
```

### Adding API Keys

After initialization, you should add API keys for LLM models:

```python
from database import SessionLocal
from models import LLMModel

db = SessionLocal()
model = db.query(LLMModel).filter(LLMModel.model_name == "gpt-4-turbo-preview").first()
model.api_key = "your-api-key-here"
db.commit()
```

## 🗄️ Using the Database

### In FastAPI Endpoints

```python
from fastapi import Depends
from sqlalchemy.orm import Session
from database import get_db

@app.get("/items")
async def get_items(db: Session = Depends(get_db)):
    # Use db here
    items = db.query(MyModel).all()
    return items
```

### Direct Database Access

```python
from database import SessionLocal
from models import LLMModel

db = SessionLocal()
models = db.query(LLMModel).all()
db.close()
```

## 🔍 Viewing the Database

You can view the database using SQLite tools:

```bash
# Using SQLite CLI
sqlite3 rag_chat.db

# Then run SQL commands
.tables
SELECT * FROM llm_models;
SELECT * FROM users;
```

Or use a GUI tool like:
- [DB Browser for SQLite](https://sqlitebrowser.org/)
- [DBeaver](https://dbeaver.io/)

## 🔄 Re-initializing the Database

To start fresh:

```bash
# Remove the existing database
rm rag_chat.db  # On Windows: del rag_chat.db

# Run the initialization script again
python init_db.py
```

## ⚠️ Production Notes

1. **Use PostgreSQL**: Replace SQLite with PostgreSQL for production
2. **Environment Variables**: Store sensitive data in environment variables
3. **Password Hashing**: Use proper password hashing (bcrypt, Argon2)
4. **Encryption**: Encrypt API keys and sensitive data
5. **Backups**: Set up regular database backups
6. **Connection Pooling**: Configure appropriate connection pooling

## 📝 Next Steps

1. Update API keys for your LLM providers
2. Create proper user accounts with hashed passwords
3. Implement authentication
4. Add document upload functionality
5. Connect vector database for RAG

