"""
Database Schema Extraction Service
Extracts schema information from various database types
"""

from typing import Dict, List, Optional, Any
import json
from sqlalchemy import create_engine, inspect, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)


class DatabaseSchemaService:
    """Service for extracting database schema information"""
    
    SUPPORTED_DATABASES = ['postgresql', 'mysql', 'sqlite', 'oracle', 'mssql', 'mariadb']
    
    def __init__(self):
        self.supported_formats = self.SUPPORTED_DATABASES
    
    def extract_schema(
        self,
        database_type: str,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database_name: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        schema_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract schema from database
        
        Args:
            database_type: Type of database (postgresql, mysql, sqlite, etc.)
            connection_string: Full connection string (optional, overrides other params)
            host: Database host
            port: Database port
            database_name: Database name
            username: Database username
            password: Database password
            schema_name: Specific schema to extract (if applicable)
            
        Returns:
            Dictionary with schema information
        """
        try:
            # Build connection string if not provided
            if not connection_string:
                connection_string = self._build_connection_string(
                    database_type, host, port, database_name, username, password
                )
            
            # Create engine
            engine = create_engine(connection_string, echo=False)
            
            # Extract schema
            inspector = inspect(engine)
            metadata = MetaData()
            metadata.reflect(bind=engine, schema=schema_name)
            
            schema_info = {
                'database_type': database_type,
                'database_name': database_name,
                'schema_name': schema_name,
                'tables': [],
                'metadata': {
                    'total_tables': 0,
                    'total_columns': 0,
                    'extraction_method': 'sqlalchemy_inspect'
                }
            }
            
            # Get list of tables
            try:
                if schema_name:
                    tables = inspector.get_table_names(schema=schema_name)
                else:
                    tables = inspector.get_table_names()
            except TypeError:
                # Some databases don't support schema parameter
                tables = inspector.get_table_names()
            
            for table_name in tables:
                try:
                    table_info = self._extract_table_info(inspector, table_name, schema_name)
                    schema_info['tables'].append(table_info)
                    schema_info['metadata']['total_columns'] += len(table_info.get('columns', []))
                except Exception as e:
                    logger.warning(f"Failed to extract info for table {table_name}: {str(e)}")
                    continue
            
            schema_info['metadata']['total_tables'] = len(schema_info['tables'])
            
            # Generate human-readable text representation
            schema_text = self._generate_schema_text(schema_info)
            schema_info['schema_text'] = schema_text
            
            # Close connection
            engine.dispose()
            
            return {
                'success': True,
                'schema_data': schema_info,
                'schema_text': schema_text,
                'metadata': schema_info['metadata']
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Database error extracting schema: {str(e)}")
            return {
                'success': False,
                'error': f"Database error: {str(e)}",
                'schema_data': None,
                'schema_text': None
            }
        except Exception as e:
            logger.error(f"Error extracting schema: {str(e)}")
            return {
                'success': False,
                'error': f"Error extracting schema: {str(e)}",
                'schema_data': None,
                'schema_text': None
            }
    
    def _build_connection_string(
        self,
        database_type: str,
        host: Optional[str],
        port: Optional[int],
        database_name: Optional[str],
        username: Optional[str],
        password: Optional[str]
    ) -> str:
        """Build SQLAlchemy connection string"""
        if database_type == 'postgresql':
            driver = 'postgresql+psycopg2'
            conn_str = f"{driver}://"
            if username:
                conn_str += username
                if password:
                    conn_str += f":{password}"
                conn_str += "@"
            if host:
                conn_str += host
                if port:
                    conn_str += f":{port}"
            if database_name:
                conn_str += f"/{database_name}"
        elif database_type == 'mysql' or database_type == 'mariadb':
            driver = 'mysql+pymysql'
            conn_str = f"{driver}://"
            if username:
                conn_str += username
                if password:
                    conn_str += f":{password}"
                conn_str += "@"
            if host:
                conn_str += host
                if port:
                    conn_str += f":{port}"
            if database_name:
                conn_str += f"/{database_name}"
        elif database_type == 'sqlite':
            # SQLite uses file path
            conn_str = f"sqlite:///{database_name or ':memory:'}"
        elif database_type == 'oracle':
            driver = 'oracle+cx_oracle'
            conn_str = f"{driver}://"
            if username:
                conn_str += username
                if password:
                    conn_str += f":{password}"
                conn_str += "@"
            if host:
                conn_str += host
                if port:
                    conn_str += f":{port}"
            if database_name:
                conn_str += f"/{database_name}"
        elif database_type == 'mssql':
            driver = 'mssql+pyodbc'
            conn_str = f"{driver}://"
            if username:
                conn_str += username
                if password:
                    conn_str += f":{password}"
                conn_str += "@"
            if host:
                conn_str += host
                if port:
                    conn_str += f":{port}"
            if database_name:
                conn_str += f"?driver=ODBC+Driver+17+for+SQL+Server&database={database_name}"
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
        
        return conn_str
    
    def _extract_table_info(self, inspector, table_name: str, schema_name: Optional[str] = None) -> Dict[str, Any]:
        """Extract detailed information about a table"""
        table_info = {
            'name': table_name,
            'schema': schema_name,
            'columns': [],
            'primary_keys': [],
            'foreign_keys': [],
            'indexes': []
        }
        
        try:
            # Get columns - handle schema parameter gracefully
            try:
                columns = inspector.get_columns(table_name, schema=schema_name) if schema_name else inspector.get_columns(table_name)
            except TypeError:
                columns = inspector.get_columns(table_name)
            
            for col in columns:
                col_info = {
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True),
                    'default': str(col.get('default', '')) if col.get('default') else None
                }
                table_info['columns'].append(col_info)
        except Exception as e:
            logger.warning(f"Failed to get columns for {table_name}: {str(e)}")
        
        try:
            # Get primary keys
            try:
                pk_constraint = inspector.get_pk_constraint(table_name, schema=schema_name) if schema_name else inspector.get_pk_constraint(table_name)
            except TypeError:
                pk_constraint = inspector.get_pk_constraint(table_name)
            
            if pk_constraint and pk_constraint.get('constrained_columns'):
                table_info['primary_keys'] = pk_constraint['constrained_columns']
        except Exception as e:
            logger.warning(f"Failed to get primary keys for {table_name}: {str(e)}")
        
        try:
            # Get foreign keys
            try:
                fks = inspector.get_foreign_keys(table_name, schema=schema_name) if schema_name else inspector.get_foreign_keys(table_name)
            except TypeError:
                fks = inspector.get_foreign_keys(table_name)
            
            for fk in fks:
                fk_info = {
                    'name': fk.get('name', ''),
                    'constrained_columns': fk.get('constrained_columns', []),
                    'referred_table': fk.get('referred_table', ''),
                    'referred_columns': fk.get('referred_columns', [])
                }
                table_info['foreign_keys'].append(fk_info)
        except Exception as e:
            logger.warning(f"Failed to get foreign keys for {table_name}: {str(e)}")
        
        try:
            # Get indexes
            try:
                indexes = inspector.get_indexes(table_name, schema=schema_name) if schema_name else inspector.get_indexes(table_name)
            except TypeError:
                indexes = inspector.get_indexes(table_name)
            
            for idx in indexes:
                idx_info = {
                    'name': idx.get('name', ''),
                    'columns': idx.get('column_names', []),
                    'unique': idx.get('unique', False)
                }
                table_info['indexes'].append(idx_info)
        except Exception as e:
            logger.warning(f"Failed to get indexes for {table_name}: {str(e)}")
        
        return table_info
    
    def _generate_schema_text(self, schema_info: Dict[str, Any]) -> str:
        """Generate human-readable text representation of schema"""
        lines = []
        
        lines.append(f"Database: {schema_info.get('database_name', 'Unknown')}")
        lines.append(f"Type: {schema_info.get('database_type', 'Unknown')}")
        if schema_info.get('schema_name'):
            lines.append(f"Schema: {schema_info.get('schema_name')}")
        lines.append("")
        
        for table in schema_info.get('tables', []):
            lines.append(f"Table: {table['name']}")
            lines.append("-" * 50)
            
            # Columns
            if table.get('columns'):
                lines.append("Columns:")
                for col in table['columns']:
                    col_line = f"  - {col['name']}: {col['type']}"
                    if not col.get('nullable', True):
                        col_line += " (NOT NULL)"
                    if col.get('default'):
                        col_line += f" DEFAULT {col['default']}"
                    lines.append(col_line)
            
            # Primary Keys
            if table.get('primary_keys'):
                lines.append(f"Primary Keys: {', '.join(table['primary_keys'])}")
            
            # Foreign Keys
            if table.get('foreign_keys'):
                lines.append("Foreign Keys:")
                for fk in table['foreign_keys']:
                    fk_line = f"  - {', '.join(fk['constrained_columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                    lines.append(fk_line)
            
            # Indexes
            if table.get('indexes'):
                lines.append("Indexes:")
                for idx in table['indexes']:
                    idx_line = f"  - {idx['name']}: {', '.join(idx['columns'])}"
                    if idx.get('unique'):
                        idx_line += " (UNIQUE)"
                    lines.append(idx_line)
            
            lines.append("")
        
        return "\n".join(lines)
    
    def test_connection(
        self,
        database_type: str,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database_name: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test database connection"""
        try:
            if not connection_string:
                connection_string = self._build_connection_string(
                    database_type, host, port, database_name, username, password
                )
            
            engine = create_engine(connection_string, echo=False)
            
            # Test connection with a simple query
            with engine.connect() as conn:
                if database_type == 'postgresql':
                    conn.execute(text("SELECT 1"))
                elif database_type in ['mysql', 'mariadb']:
                    conn.execute(text("SELECT 1"))
                elif database_type == 'sqlite':
                    conn.execute(text("SELECT 1"))
                elif database_type == 'oracle':
                    conn.execute(text("SELECT 1 FROM DUAL"))
                elif database_type == 'mssql':
                    conn.execute(text("SELECT 1"))
            
            engine.dispose()
            
            return {
                'success': True,
                'message': 'Connection successful'
            }
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return {
                'success': False,
                'message': f"Connection failed: {str(e)}"
            }

