"""
Database initialization script for RAG Chat Application
Run this script to create the database and all required tables
"""

from sqlalchemy import create_engine
from models import Base, User, LLMModel, LLMProvider, ChatSession, ChatMessage, Document, EmbeddingModel, DatabaseConnection, DatabaseSchema
import os
from datetime import datetime


# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_chat.db")
engine = create_engine(DATABASE_URL, echo=True)


def init_database():
    """Create all database tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    # Create default LLM models
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if models already exist
        existing_models = session.query(LLMModel).count()
        if existing_models > 0:
            print(f"⚠️  Found {existing_models} existing LLM models. Skipping default models.")
        else:
            # Add default LLM models
            default_models = [
                {
                    "model_name": "gpt-4-turbo-preview",
                    "provider": LLMProvider.OPENAI,
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "",  # User will need to add this
                    "description": "OpenAI GPT-4 Turbo - Powerful language model from OpenAI",
                    "is_active": 1
                },
                {
                    "model_name": "gpt-3.5-turbo",
                    "provider": LLMProvider.OPENAI,
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "",  # User will need to add this
                    "description": "OpenAI GPT-3.5 Turbo - Fast and efficient language model",
                    "is_active": 1
                },
                {
                    "model_name": "llama2",
                    "provider": LLMProvider.OLLAMA,
                    "base_url": "http://localhost:11434",  # Default Ollama endpoint
                    "api_key": "",  # Ollama doesn't require API key
                    "description": "Meta LLaMA 2 - Locally hosted via Ollama",
                    "is_active": 1
                },
                {
                    "model_name": "mistral",
                    "provider": LLMProvider.OLLAMA,
                    "base_url": "http://localhost:11434",
                    "api_key": "",
                    "description": "Mistral AI model - Locally hosted via Ollama",
                    "is_active": 1
                },
                {
                    "model_name": "mixtral",
                    "provider": LLMProvider.OLLAMA,
                    "base_url": "http://localhost:11434",
                    "api_key": "",
                    "description": "Mixtral MoE model - Locally hosted via Ollama",
                    "is_active": 1
                }
            ]
            
            for model_data in default_models:
                llm_model = LLMModel(**model_data)
                session.add(llm_model)
            
            session.commit()
            print(f"✅ Added {len(default_models)} default LLM models")
            
            # Create default embedding models
            print("\n📝 Creating default embedding models...")
            existing_embedding_models = session.query(EmbeddingModel).count()
            if existing_embedding_models > 0:
                print(f"⚠️  Found {existing_embedding_models} existing embedding models. Skipping default models.")
            else:
                default_embedding_models = [
                    {
                        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                        "display_name": "MiniLM-L6-v2 (English)",
                        "description": "Fast and efficient English embedding model. Good for general-purpose English text.",
                        "language": "english",
                        "model_path": "sentence-transformers/all-MiniLM-L6-v2",
                        "dimension": 384,
                        "is_active": 1,
                        "is_downloaded": 0
                    },
                    {
                        "model_name": "sentence-transformers/all-mpnet-base-v2",
                        "display_name": "MPNet Base v2 (English)",
                        "description": "Higher quality English embedding model. Better accuracy but slower than MiniLM.",
                        "language": "english",
                        "model_path": "sentence-transformers/all-mpnet-base-v2",
                        "dimension": 768,
                        "is_active": 1,
                        "is_downloaded": 0
                    },
                    {
                        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        "display_name": "Multilingual MiniLM (50+ languages)",
                        "description": "Multilingual embedding model supporting 50+ languages including Arabic, Spanish, French, etc.",
                        "language": "multilingual",
                        "model_path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        "dimension": 384,
                        "is_active": 1,
                        "is_downloaded": 0
                    },
                    {
                        "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        "display_name": "Multilingual MPNet (50+ languages)",
                        "description": "High-quality multilingual embedding model. Best for multilingual documents.",
                        "language": "multilingual",
                        "model_path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        "dimension": 768,
                        "is_active": 1,
                        "is_downloaded": 0
                    },
                    {
                        "model_name": "aubmindlab/bert-base-arabertv2",
                        "display_name": "AraBERT v2 (Arabic)",
                        "description": "Arabic-specific BERT model optimized for Arabic text embeddings.",
                        "language": "arabic",
                        "model_path": "aubmindlab/bert-base-arabertv2",
                        "dimension": 768,
                        "is_active": 1,
                        "is_downloaded": 0
                    },
                    {
                        "model_name": "CAMeL-Lab/bert-base-arabic-camelbert-da",
                        "display_name": "CAMeLBERT (Arabic)",
                        "description": "Arabic CAMeLBERT model for Arabic text embeddings and understanding.",
                        "language": "arabic",
                        "model_path": "CAMeL-Lab/bert-base-arabic-camelbert-da",
                        "dimension": 768,
                        "is_active": 1,
                        "is_downloaded": 0
                    }
                ]
                
                for model_data in default_embedding_models:
                    embedding_model = EmbeddingModel(**model_data)
                    session.add(embedding_model)
                
                session.commit()
                print(f"✅ Added {len(default_embedding_models)} default embedding models")
            
            # Create a default user (for testing)
            print("\n📝 Creating a default test user...")
            default_user = User(
                username="admin",
                email="admin@ragchat.com",
                password_hash="placeholder_hash",  # Should be properly hashed in production
                full_name="Administrator",
                is_active=1
            )
            session.add(default_user)
            session.commit()
            print("✅ Created default user (username: admin, email: admin@ragchat.com)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        session.rollback()
    finally:
        session.close()
    
    print("\n" + "="*50)
    print("🎉 Database initialization complete!")
    print("="*50)
    print("\n📊 Database Summary:")
    print(f"   - Database URL: {DATABASE_URL}")
    print("   - Tables created: users, llm_models, embedding_models, chat_sessions, chat_messages, documents, database_connections, database_schemas")
    print("\n💡 Next steps:")
    print("   1. Update API keys in llm_models table")
    print("   2. Download embedding models as needed")
    print("   3. Create user accounts")
    print("   4. Start adding documents for RAG")


if __name__ == "__main__":
    print("="*50)
    print("🚀 Initializing RAG Chat Database")
    print("="*50)
    print()
    init_database()
    print("\n✅ Database setup complete!")

