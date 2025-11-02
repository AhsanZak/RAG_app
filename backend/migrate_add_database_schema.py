"""
Migration script to add database_connections and database_schemas tables
Run this if you already have an existing database
"""

from sqlalchemy import create_engine
from models import Base, DatabaseConnection, DatabaseSchema
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_chat.db")
engine = create_engine(DATABASE_URL, echo=True)


def migrate_database():
    """Add new tables for database schema chat feature"""
    print("="*50)
    print("🔧 Migrating Database - Adding Database Schema Tables")
    print("="*50)
    print()
    
    try:
        print("Creating database_connections and database_schemas tables...")
        Base.metadata.create_all(
            bind=engine,
            tables=[
                DatabaseConnection.__table__,
                DatabaseSchema.__table__
            ]
        )
        print("✅ Migration complete!")
        print()
        print("📊 New tables added:")
        print("   - database_connections")
        print("   - database_schemas")
        print()
        print("💡 You can now use the database chat feature!")
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    migrate_database()

