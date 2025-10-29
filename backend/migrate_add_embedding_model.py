"""
Migration script to add embedding_model_id column to chat_sessions table
Run this script to update your existing database schema
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_chat.db")
engine = create_engine(DATABASE_URL, echo=True)

def migrate_database():
    """Add embedding_model_id column to chat_sessions table if it doesn't exist"""
    print("="*50)
    print("Running Database Migration")
    print("="*50)
    print()
    
    try:
        # Check if column exists
        with engine.begin() as conn:
            # For SQLite, check if column exists
            result = conn.execute(text("PRAGMA table_info(chat_sessions)"))
            columns = [row[1] for row in result]
            
            if 'embedding_model_id' in columns:
                print("[OK] Column 'embedding_model_id' already exists in chat_sessions table")
            else:
                # Add the column if it doesn't exist
                print("[INFO] Adding 'embedding_model_id' column to chat_sessions table...")
                conn.execute(text("""
                    ALTER TABLE chat_sessions 
                    ADD COLUMN embedding_model_id INTEGER
                """))
                print("[OK] Successfully added 'embedding_model_id' column")
            
            # Also create embedding_models table if it doesn't exist
            print("\n[INFO] Checking embedding_models table...")
            result = conn.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='embedding_models'
            """))
            table_exists = result.fetchone()
            
            if not table_exists:
                print("[INFO] Creating embedding_models table...")
                conn.execute(text("""
                    CREATE TABLE embedding_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name VARCHAR(200) NOT NULL UNIQUE,
                        display_name VARCHAR(200) NOT NULL,
                        description TEXT,
                        language VARCHAR(50),
                        model_path VARCHAR(500),
                        dimension INTEGER,
                        is_active INTEGER DEFAULT 1,
                        is_downloaded INTEGER DEFAULT 0,
                        download_progress INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                print("[OK] Successfully created embedding_models table")
            else:
                print("[OK] embedding_models table already exists")
        
        print("\n" + "="*50)
        print("Migration completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    migrate_database()

