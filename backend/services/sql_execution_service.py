"""
SQL Execution Service
Executes SQL queries on database connections
"""

from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)


class SQLExecutionService:
    """Service for executing SQL queries on databases"""
    
    def __init__(self):
        pass
    
    def execute_query(
        self,
        sql: str,
        database_type: str,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database_name: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute SQL query on database
        
        Returns:
            {
                'success': bool,
                'data': List[Dict] or None,
                'columns': List[str] or None,
                'row_count': int,
                'error': str or None,
                'execution_time': float
            }
        """
        import time
        start_time = time.time()
        
        try:
            # Build connection string if not provided
            if not connection_string:
                connection_string = self._build_connection_string(
                    database_type, host, port, database_name, username, password
                )
            
            # Create engine
            engine = create_engine(connection_string, echo=False)
            
            # Execute query
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                
                # Fetch results
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = result.fetchall()
                    
                    # Convert rows to dicts
                    data = [dict(zip(columns, row)) for row in rows]
                    
                    return {
                        'success': True,
                        'data': data,
                        'columns': columns,
                        'row_count': len(data),
                        'error': None,
                        'execution_time': time.time() - start_time
                    }
                else:
                    # For INSERT/UPDATE/DELETE
                    rowcount = result.rowcount
                    return {
                        'success': True,
                        'data': None,
                        'columns': None,
                        'row_count': rowcount,
                        'error': None,
                        'execution_time': time.time() - start_time
                    }
        
        except SQLAlchemyError as e:
            error_msg = str(e)
            logger.error(f"SQL execution error: {error_msg}")
            
            # Provide user-friendly error messages
            user_friendly_error = self._format_error_message(error_msg)
            
            return {
                'success': False,
                'data': None,
                'columns': None,
                'row_count': 0,
                'error': user_friendly_error,
                'error_type': 'sql_error',
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error executing SQL: {error_msg}")
            
            # Provide user-friendly error messages
            user_friendly_error = self._format_error_message(error_msg)
            
            return {
                'success': False,
                'data': None,
                'columns': None,
                'row_count': 0,
                'error': user_friendly_error,
                'error_type': 'execution_error',
                'execution_time': time.time() - start_time
            }
        finally:
            try:
                engine.dispose()
            except:
                pass
    
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
    
    def _format_error_message(self, error_msg: str) -> str:
        """Format technical error messages into user-friendly messages"""
        error_lower = error_msg.lower()
        
        # Common error patterns and user-friendly messages
        if 'relation' in error_lower or 'does not exist' in error_lower or 'table' in error_lower:
            if 'table' in error_lower:
                return "The table you're trying to query doesn't exist in the database. Please check the table name."
            else:
                return "The table or column you're referencing doesn't exist in the database."
        
        if 'column' in error_lower and ('does not exist' in error_lower or 'unknown' in error_lower):
            return "One or more columns in your query don't exist in the database. Please check the column names."
        
        if 'syntax error' in error_lower or 'invalid syntax' in error_lower:
            return "There's a syntax error in the generated SQL query. The AI will try to correct it."
        
        if 'permission denied' in error_lower or 'access denied' in error_lower:
            return "You don't have permission to execute this query on the database."
        
        if 'connection' in error_lower or 'timeout' in error_lower:
            return "Could not connect to the database. Please check your database connection settings."
        
        if 'foreign key' in error_lower:
            return "There's a foreign key constraint issue. The query references relationships that don't exist."
        
        if 'unique constraint' in error_lower or 'duplicate' in error_lower:
            return "The query violates a unique constraint in the database."
        
        if 'cannot be null' in error_lower or 'null constraint' in error_lower:
            return "The query tries to insert or update a NULL value in a column that doesn't allow NULL values."
        
        # Generic user-friendly message
        return "The database query encountered an error. Please try rephrasing your question or contact support if the issue persists."
    
