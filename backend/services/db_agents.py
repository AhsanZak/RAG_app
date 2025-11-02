"""
Database Chat Agents
Multi-agent system for processing database queries:
1. Intent Detection Agent - Determines if query needs SQL
2. NL-to-SQL Agent - Converts natural language to SQL
3. SQL Validation Agent - Validates and corrects SQL against schema
"""

from typing import Dict, List, Optional, Any
import re
import json
import numpy as np
from services.llm_service import LLMService
from services.database_schema_service import DatabaseSchemaService
from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService


class IntentDetectionAgent:
    """Agent to detect user intent - whether they need SQL query or general chat"""
    
    INTENT_SQL_KEYWORDS = [
        'query', 'select', 'find', 'get', 'show', 'list', 'retrieve', 'fetch',
        'count', 'sum', 'average', 'avg', 'max', 'min', 'group', 'join',
        'where', 'filter', 'search', 'database', 'table', 'column', 'rows',
        'records', 'data', 'information', 'extract', 'display'
    ]
    
    def __init__(self, llm_service: LLMService, embedding_service: Optional[EmbeddingService] = None,
                 chromadb_service: Optional[ChromaDBService] = None, model_config: Optional[Dict] = None):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.chromadb_service = chromadb_service
        self.model_config = model_config or {}
    
    def detect_intent(self, user_query: str, schema_context: Optional[str] = None,
                      collection_name: Optional[str] = None, embedding_model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect if user intent requires SQL query
        
        Returns:
            {
                'requires_sql': bool,
                'confidence': float,
                'intent_type': str,  # 'sql_query', 'schema_question', 'general_chat'
                'reasoning': str
            }
        """
        user_query_lower = user_query.lower()
        
        # Quick keyword check
        has_sql_keywords = any(keyword in user_query_lower for keyword in self.INTENT_SQL_KEYWORDS)
        
        # Use semantic search to get relevant schema context if available
        relevant_context = schema_context
        if collection_name and self.embedding_service and self.chromadb_service:
            try:
                # Expand query with synonyms for better semantic matching
                expanded_queries = self._expand_query_with_synonyms(user_query)
                
                # Convert all expanded queries to embeddings
                all_queries = [user_query] + expanded_queries
                query_embeddings = self.embedding_service.generate_embeddings(
                    all_queries,
                    model_name=embedding_model_name
                )
                
                # Use the first (original) query embedding for retrieval
                if query_embeddings.ndim == 1:
                    query_embeddings = query_embeddings.reshape(1, -1)
                else:
                    query_embeddings = query_embeddings[0:1]  # Use first query
                
                # Retrieve relevant schema chunks
                results = self.chromadb_service.query_collection(
                    collection_name=collection_name,
                    query_embeddings=query_embeddings,
                    n_results=8  # Get more results for intent detection
                )
                
                documents = results.get('documents', [[]])[0] if results.get('documents') else []
                if documents:
                    relevant_context = "\n\n".join(documents)
                    print(f"[IntentDetection] Using {len(documents)} relevant schema chunks for intent detection")
            except Exception as e:
                print(f"[IntentDetection] Semantic search failed: {str(e)}")
                # Fallback to provided schema_context
        
        # Use LLM for better intent detection with semantically relevant context
        prompt = f"""Analyze the following user query and determine if it requires a SQL database query to answer.

User Query: "{user_query}"

Relevant Schema Context (semantically matched): {relevant_context[:800] if relevant_context else 'Not available'}

Determine:
1. Does this query require fetching data from the database using SQL? (requires_sql: true/false)
2. What type of intent is this? (sql_query, schema_question, general_chat)
3. Confidence level (0.0 to 1.0)

Respond in JSON format:
{{
    "requires_sql": true/false,
    "confidence": 0.0-1.0,
    "intent_type": "sql_query|schema_question|general_chat",
    "reasoning": "brief explanation"
}}
"""
        
        try:
            response = self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model_config=self.model_config,
                temperature=0.1
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'requires_sql': result.get('requires_sql', has_sql_keywords),
                    'confidence': result.get('confidence', 0.8 if has_sql_keywords else 0.3),
                    'intent_type': result.get('intent_type', 'sql_query' if has_sql_keywords else 'general_chat'),
                    'reasoning': result.get('reasoning', '')
                }
        except Exception as e:
            print(f"[IntentDetection] Error: {str(e)}")
        
        # Fallback to keyword-based detection
        return {
            'requires_sql': has_sql_keywords,
            'confidence': 0.7 if has_sql_keywords else 0.3,
            'intent_type': 'sql_query' if has_sql_keywords else 'general_chat',
            'reasoning': 'Keyword-based detection'
        }
    
    def _expand_query_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms for better semantic matching"""
        query_lower = query.lower()
        expanded = []
        
        # Common synonym expansions
        synonym_expansions = {
            'movie': ['film', 'movies', 'films', 'cinema'],
            'movies': ['film', 'films', 'cinema'],
            'film': ['movie', 'movies', 'films', 'cinema'],
            'user': ['users', 'person', 'people', 'account'],
            'customer': ['customers', 'client', 'clients', 'buyer'],
            'order': ['orders', 'purchase', 'purchases', 'transaction'],
            'product': ['products', 'item', 'items', 'goods'],
            'employee': ['employees', 'staff', 'worker', 'personnel'],
        }
        
        # Find synonyms and create expanded queries
        for word, synonyms in synonym_expansions.items():
            if word in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(word, synonym)
                    if expanded_query != query_lower:
                        expanded.append(expanded_query)
        
        # Also add query with common terms
        common_terms = {
            'show me': ['list', 'get', 'fetch', 'retrieve', 'display'],
            'find': ['search', 'get', 'retrieve', 'fetch'],
            'how many': ['count', 'total', 'number of'],
        }
        
        for term, alternatives in common_terms.items():
            if term in query_lower:
                for alt in alternatives:
                    expanded_query = query_lower.replace(term, alt)
                    if expanded_query != query_lower:
                        expanded.append(expanded_query)
        
        # Return unique expansions
        return list(set(expanded))[:5]  # Limit to 5 expansions


class NLToSQLAgent:
    """Agent to convert natural language queries to SQL"""
    
    def __init__(self, llm_service: LLMService, embedding_service: EmbeddingService, 
                 chromadb_service: ChromaDBService, model_config: Optional[Dict] = None):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.chromadb_service = chromadb_service
        self.model_config = model_config or {}
    
    def generate_sql(
        self, 
        user_query: str, 
        schema_details: Dict[str, Any],
        database_type: str,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language using semantic search
        
        Returns:
            {
                'sql': str,
                'confidence': float,
                'reasoning': str,
                'tables_used': List[str],
                'columns_used': List[str],
                'relevant_schema_parts': List[str]
            }
        """
        # Use semantic search to find relevant schema parts if collection exists
        relevant_schema_text = None
        relevant_schema_parts = []
        
        if collection_name:
            try:
                # Expand query with synonyms for better semantic matching
                expanded_queries = self._expand_query_with_synonyms(user_query)
                
                # Convert all expanded queries to embeddings for semantic search
                all_queries = [user_query] + expanded_queries
                query_embeddings = self.embedding_service.generate_embeddings(
                    all_queries,
                    model_name=embedding_model_name
                )
                
                # Use the first (original) query embedding for retrieval
                if query_embeddings.ndim == 1:
                    query_embeddings = query_embeddings.reshape(1, -1)
                else:
                    query_embeddings = query_embeddings[0:1]  # Use first query
                
                # Retrieve relevant schema chunks using semantic search
                results = self.chromadb_service.query_collection(
                    collection_name=collection_name,
                    query_embeddings=query_embeddings,
                    n_results=15  # Get more results for better coverage
                )
                
                # Extract relevant documents
                documents = results.get('documents', [[]])[0] if results.get('documents') else []
                metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
                distances = results.get('distances', [[]])[0] if results.get('distances') else []
                
                # Filter by relevance (lower distance = more relevant)
                # Use top-N results with reasonable distance filtering
                relevant_docs = []
                if documents:
                    # Use top results (up to 10), but prioritize by distance
                    # For cosine similarity: smaller distance = more similar
                    # Filter threshold: take results within reasonable similarity range
                    max_results_to_use = min(10, len(documents))
                    
                    # Sort by distance (ascending - most relevant first)
                    doc_dist_pairs = list(zip(documents, metadatas, distances))
                    doc_dist_pairs.sort(key=lambda x: x[2])  # Sort by distance
                    
                    # Use top results - take top 70% by distance, but at least top 5
                    threshold_index = max(5, int(len(doc_dist_pairs) * 0.7))
                    top_pairs = doc_dist_pairs[:max(max_results_to_use, threshold_index)]
                    
                    # Use all top results (they're already sorted by relevance)
                    relevant_docs = [doc for doc, meta, dist in top_pairs]
                    relevant_schema_parts = relevant_docs.copy()
                    
                    # Also check metadata for synonyms match
                    query_lower = user_query.lower()
                    for doc, meta, dist in zip(documents, metadatas, distances):
                        # Check if synonyms in metadata match query
                        synonyms_str = meta.get('synonyms', '')
                        if synonyms_str:
                            synonyms = [s.strip().lower() for s in synonyms_str.split(',') if s.strip()]
                            for synonym in synonyms:
                                if synonym in query_lower or query_lower in synonym:
                                    # Add this document if not already included
                                    if doc not in relevant_docs:
                                        relevant_docs.append(doc)
                                        relevant_schema_parts.append(doc)
                                        break
                
                # Combine relevant schema parts
                if relevant_docs:
                    relevant_schema_text = "\n\n".join(relevant_docs)
                    print(f"[NLToSQL] Using {len(relevant_docs)} relevant schema chunks from semantic search")
            except Exception as e:
                print(f"[NLToSQL] Semantic search failed: {str(e)}, falling back to full schema")
        
        # Fallback to full schema if semantic search didn't work
        if not relevant_schema_text:
            schema_text = self._format_schema_for_llm(schema_details)
            relevant_schema_text = schema_text
        
        prompt = f"""You are an expert SQL query generator. Convert the following natural language query into a valid {database_type.upper()} SQL query.

Database Type: {database_type.upper()}

Relevant Schema Details (retrieved via semantic search based on your query):
{relevant_schema_text}

User Query: "{user_query}"

Instructions:
1. Generate a valid {database_type.upper()} SQL query
2. Use only tables and columns that exist in the schema
3. Include proper JOINs if multiple tables are needed
4. Add appropriate WHERE clauses based on the query intent
5. Use appropriate aggregate functions if needed (COUNT, SUM, AVG, MAX, MIN)
6. Add GROUP BY if aggregations are used
7. Add ORDER BY if results should be sorted
8. Limit results to a reasonable number (use LIMIT or TOP)

Respond in JSON format:
{{
    "sql": "the SQL query here",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of the query",
    "tables_used": ["table1", "table2"],
    "columns_used": ["column1", "column2"]
}}
"""
        
        try:
            response = self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model_config=self.model_config,
                temperature=0.1
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return {
                        'sql': result.get('sql', '').strip(),
                        'confidence': result.get('confidence', 0.7),
                        'reasoning': result.get('reasoning', ''),
                        'tables_used': result.get('tables_used', []),
                        'columns_used': result.get('columns_used', []),
                        'relevant_schema_parts': relevant_schema_parts
                    }
                except json.JSONDecodeError:
                    # Try to extract SQL directly
                    sql_match = re.search(r'(?i)(SELECT|INSERT|UPDATE|DELETE)[^;]+', response, re.DOTALL)
                    if sql_match:
                        return {
                            'sql': sql_match.group().strip(),
                            'confidence': 0.6,
                            'reasoning': 'Extracted from response',
                            'tables_used': [],
                            'columns_used': [],
                            'relevant_schema_parts': relevant_schema_parts
                        }
        except Exception as e:
            print(f"[NLToSQL] Error: {str(e)}")
        
        return {
            'sql': '',
            'confidence': 0.0,
            'reasoning': f'Error generating SQL: {str(e)}',
            'tables_used': [],
            'columns_used': [],
            'relevant_schema_parts': []
        }
    
    def _format_schema_for_llm(self, schema_details: Dict[str, Any]) -> str:
        """Format schema details for LLM prompt"""
        if not schema_details or 'tables' not in schema_details:
            return "No schema information available"
        
        lines = []
        for table in schema_details.get('tables', []):
            lines.append(f"\nTable: {table.get('name', 'unknown')}")
            if table.get('schema'):
                lines.append(f"  Schema: {table['schema']}")
            
            # Columns
            if table.get('columns'):
                lines.append("  Columns:")
                for col in table['columns']:
                    col_line = f"    - {col['name']}: {col['type']}"
                    if not col.get('nullable', True):
                        col_line += " (NOT NULL)"
                    lines.append(col_line)
            
            # Primary Keys
            if table.get('primary_keys'):
                lines.append(f"  Primary Keys: {', '.join(table['primary_keys'])}")
            
            # Foreign Keys
            if table.get('foreign_keys'):
                lines.append("  Foreign Keys:")
                for fk in table['foreign_keys']:
                    lines.append(f"    - {', '.join(fk['constrained_columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}")
        
        return '\n'.join(lines)


class SQLValidationAgent:
    """Agent to validate and correct SQL queries against schema"""
    
    def __init__(self, llm_service: LLMService, schema_service: DatabaseSchemaService, model_config: Optional[Dict] = None):
        self.llm_service = llm_service
        self.schema_service = schema_service
        self.model_config = model_config or {}
    
    def validate_and_correct_sql(
        self,
        sql: str,
        schema_details: Dict[str, Any],
        database_type: str,
        relevant_schema_parts: Optional[List[str]] = None,
        connection_string: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database_name: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate SQL against schema and correct if needed
        
        Returns:
            {
                'valid': bool,
                'corrected_sql': str,
                'errors': List[str],
                'warnings': List[str],
                'validated_tables': List[str],
                'validated_columns': List[str]
            }
        """
        # Use relevant schema parts if available, otherwise use full schema
        if relevant_schema_parts and len(relevant_schema_parts) > 0:
            schema_text = "\n\n".join(relevant_schema_parts)
            print(f"[SQLValidation] Using {len(relevant_schema_parts)} relevant schema chunks for validation")
        else:
            schema_text = self._format_schema_for_llm(schema_details)
        
        prompt = f"""Validate and correct the following {database_type.upper()} SQL query against the provided schema.

Database Type: {database_type.upper()}

Relevant Schema Details (semantically matched):
{schema_text}

SQL Query to Validate:
{sql}

Instructions:
1. Check if all tables exist in the schema
2. Check if all columns exist in their respective tables
3. Validate JOIN conditions (foreign keys)
4. Check syntax for {database_type.upper()}
5. Correct any errors
6. Provide corrected SQL if needed

Respond in JSON format:
{{
    "valid": true/false,
    "corrected_sql": "corrected SQL or original if valid",
    "errors": ["error1", "error2"],
    "warnings": ["warning1", "warning2"],
    "validated_tables": ["table1", "table2"],
    "validated_columns": ["table1.column1", "table2.column2"]
}}
"""
        
        try:
            response = self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model_config=self.model_config,
                temperature=0.1
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    
                    # Sanitize errors - make them user-friendly
                    errors = result.get('errors', [])
                    user_friendly_errors = []
                    for error in errors:
                        error_lower = str(error).lower()
                        if 'table' in error_lower or 'does not exist' in error_lower:
                            user_friendly_errors.append("Table or column mismatch")
                        elif 'syntax' in error_lower or 'invalid' in error_lower:
                            user_friendly_errors.append("Syntax issue detected")
                        elif 'type' in error_lower or 'mismatch' in error_lower:
                            user_friendly_errors.append("Data type mismatch")
                        else:
                            user_friendly_errors.append("Validation issue")
                    
                    # Only return corrected SQL if validation passed
                    corrected_sql = None
                    if result.get('valid', False):
                        corrected_sql = result.get('corrected_sql', sql).strip()
                    
                    return {
                        'valid': result.get('valid', False),
                        'corrected_sql': corrected_sql,
                        'errors': user_friendly_errors if not result.get('valid', False) else [],
                        'warnings': result.get('warnings', []),
                        'validated_tables': result.get('validated_tables', []),
                        'validated_columns': result.get('validated_columns', [])
                    }
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"[SQLValidation] Error: {str(e)}")
        
        # Fallback: validate by basic syntax check
        # If SQL looks valid syntactically, assume it's valid
        try:
            sql_upper = sql.upper().strip()
            if sql_upper.startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')):
                return {
                    'valid': True,  # Give benefit of doubt
                    'corrected_sql': sql,
                    'errors': [],
                    'warnings': ['Validation could not be performed - using basic syntax check'],
                    'validated_tables': [],
                    'validated_columns': []
                }
            else:
                return {
                    'valid': False,
                    'corrected_sql': None,
                    'errors': ['Invalid SQL statement type'],
                    'warnings': [],
                    'validated_tables': [],
                    'validated_columns': []
                }
        except:
            # If even basic check fails, mark as invalid
            return {
                'valid': False,
                'corrected_sql': None,
                'errors': ['Unable to validate query'],
                'warnings': [],
                'validated_tables': [],
                'validated_columns': []
            }
    
    def _format_schema_for_llm(self, schema_details: Dict[str, Any]) -> str:
        """Format schema details for LLM prompt"""
        if not schema_details or 'tables' not in schema_details:
            return "No schema information available"
        
        lines = []
        for table in schema_details.get('tables', []):
            lines.append(f"\nTable: {table.get('name', 'unknown')}")
            
            # Columns
            if table.get('columns'):
                lines.append("  Columns:")
                for col in table['columns']:
                    lines.append(f"    - {col['name']}: {col['type']}")
            
            # Primary Keys
            if table.get('primary_keys'):
                lines.append(f"  Primary Keys: {', '.join(table['primary_keys'])}")
            
            # Foreign Keys
            if table.get('foreign_keys'):
                lines.append("  Foreign Keys:")
                for fk in table['foreign_keys']:
                    lines.append(f"    - {', '.join(fk['constrained_columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}")
        
        return '\n'.join(lines)


class DatabaseAgentSystem:
    """Main agent system orchestrator"""
    
    def __init__(self, llm_service: LLMService, schema_service: DatabaseSchemaService,
                 embedding_service: Optional[EmbeddingService] = None,
                 chromadb_service: Optional[ChromaDBService] = None,
                 model_config: Optional[Dict] = None):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.chromadb_service = chromadb_service
        self.model_config = model_config or {}
        self.intent_agent = IntentDetectionAgent(
            llm_service, 
            embedding_service, 
            chromadb_service, 
            model_config
        )
        self.nl_to_sql_agent = NLToSQLAgent(
            llm_service, 
            embedding_service, 
            chromadb_service, 
            model_config
        )
        self.sql_validation_agent = SQLValidationAgent(llm_service, schema_service, model_config)
    
    def process_query(
        self,
        user_query: str,
        schema_details: Dict[str, Any],
        schema_text: str,
        database_type: str,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        connection_string: Optional[str] = None,
        **connection_params
    ) -> Dict[str, Any]:
        """
        Process user query through agent pipeline with semantic search
        
        Returns:
            {
                'requires_sql': bool,
                'intent': Dict,
                'sql': Optional[str],
                'sql_validation': Optional[Dict],
                'should_execute': bool,
                'chat_response': Optional[str],
                'relevant_schema_parts': List[str]
            }
        """
        result = {
            'requires_sql': False,
            'intent': {},
            'sql': None,
            'sql_validation': None,
            'should_execute': False,
            'chat_response': None,
            'relevant_schema_parts': []
        }
        
        # Step 1: Detect intent using semantic search
        intent_result = self.intent_agent.detect_intent(
            user_query, 
            schema_text,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name
        )
        result['intent'] = intent_result
        
        if not intent_result.get('requires_sql'):
            # Step 2a: General chat response (no SQL needed)
            chat_response = self._generate_chat_response(
                user_query, 
                schema_details, 
                schema_text,
                previous_context=None
            )
            result['chat_response'] = chat_response
            return result
        
        # Step 2b: Generate SQL using semantic search for relevant schema parts
        result['requires_sql'] = True
        sql_result = self.nl_to_sql_agent.generate_sql(
            user_query,
            schema_details,
            database_type,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name
        )
        result['sql'] = sql_result.get('sql')
        result['relevant_schema_parts'] = sql_result.get('relevant_schema_parts', [])
        
        if not result['sql']:
            # Failed to generate SQL, provide helpful response
            result['chat_response'] = "I couldn't generate a SQL query for your request based on the available schema information. Could you try rephrasing your question or provide more specific details about what you're looking for?"
            return result
        
        # Step 3: Validate SQL using relevant schema parts
        validation_result = self.sql_validation_agent.validate_and_correct_sql(
            result['sql'],
            schema_details,
            database_type,
            relevant_schema_parts=result.get('relevant_schema_parts', []),
            connection_string=connection_string,
            **connection_params
        )
        result['sql_validation'] = validation_result
        
        if validation_result.get('valid'):
            result['sql'] = validation_result.get('corrected_sql')
            result['should_execute'] = True
        else:
            # SQL has errors, generate chat response explaining without exposing technical details
            errors = validation_result.get('errors', [])
            warnings = validation_result.get('warnings', [])
            
            # Format user-friendly error message
            if errors:
                # Filter out technical SQL details for user-friendly message
                user_friendly_errors = []
                for error in errors:
                    error_lower = error.lower()
                    if 'table' in error_lower or 'column' in error_lower:
                        user_friendly_errors.append("Some tables or columns in the query don't match the database schema.")
                    elif 'syntax' in error_lower:
                        user_friendly_errors.append("There's a syntax issue with the generated query.")
                    else:
                        user_friendly_errors.append("The query has validation issues.")
                
                error_message = user_friendly_errors[0] if user_friendly_errors else "The generated SQL query has some issues."
                result['chat_response'] = f"I tried to generate a SQL query, but {error_message.lower()} Could you rephrase your question or provide more specific details?"
            else:
                result['chat_response'] = "I generated a SQL query, but it needs verification. Could you provide more details about what you're looking for?"
        
        return result
    
    def _generate_chat_response(
        self,
        user_query: str,
        schema_details: Dict[str, Any],
        schema_text: str,
        previous_context: Optional[List[Dict]] = None
    ) -> str:
        """Generate chat response for non-SQL queries"""
        context = f"Database Schema Context:\n{schema_text[:1000]}"
        
        if previous_context:
            messages = previous_context + [{"role": "user", "content": user_query}]
        else:
            messages = [
                {"role": "system", "content": f"You are a helpful database assistant. {context}"},
                {"role": "user", "content": user_query}
            ]
        
        try:
            response = self.llm_service.generate_response(
                messages=messages, 
                model_config=self.model_config,
                temperature=0.7
            )
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

