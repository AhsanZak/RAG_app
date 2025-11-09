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
        Uses LLM and semantic search - no hardcoded keyword matching
        
        Returns:
            {
                'requires_sql': bool,
                'confidence': float,
                'intent_type': str,  # 'sql_query', 'schema_question', 'general_chat'
                'reasoning': str
            }
        """
        
        # Use semantic search to get relevant schema context if available
        relevant_context = schema_context
        if collection_name and self.embedding_service and self.chromadb_service:
            try:
                # Convert user query to embedding - rely purely on embedding model's semantic understanding
                # No query expansion - embedding models handle semantic similarity automatically
                query_embeddings = self.embedding_service.generate_embeddings(
                    [user_query],
                    model_name=embedding_model_name
                )
                
                # Ensure query_embeddings is a 2D array (1 x dimension)
                if query_embeddings.ndim == 1:
                    query_embeddings = query_embeddings.reshape(1, -1)
                
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
2. What type of intent is this?
   - "sql_query": Query needs to fetch actual data rows from tables (e.g., "show me all customers", "find orders from 2020")
   - "schema_question": Question about database structure/metadata (e.g., "how many tables", "list all tables", "what columns in table X", "does table Y exist")
   - "general_chat": General conversation not related to data or schema
3. Confidence level (0.0 to 1.0)

Examples:
- "how many tables in the database" → intent_type: "schema_question", requires_sql: false
- "show me all movies" → intent_type: "sql_query", requires_sql: true
- "what is a database" → intent_type: "general_chat", requires_sql: false

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
                try:
                    result = json.loads(json_match.group())
                    # Use LLM's determination - no hardcoded fallbacks
                    return {
                        'requires_sql': result.get('requires_sql', False),
                        'confidence': result.get('confidence', 0.7),
                        'intent_type': result.get('intent_type', 'general_chat'),
                        'reasoning': result.get('reasoning', '')
                    }
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"[IntentDetection] Error: {str(e)}")
        
        # Fallback: assume general chat if LLM analysis fails
        # No hardcoded keyword matching - rely on LLM understanding
        return {
            'requires_sql': False,
            'confidence': 0.3,
            'intent_type': 'general_chat',
            'reasoning': 'LLM analysis unavailable - defaulting to general chat'
        }
    
    def _expand_query_with_synonyms(self, query: str) -> List[str]:
        """
        No hardcoded query expansion - rely purely on embedding models
        Embedding models (BGE, MPNet, etc.) already understand semantic similarity
        """
        # Return empty list - no query expansion needed
        # Embedding models handle semantic matching automatically
        # They understand "movies" → "film", "show" → "list", etc. through training
        return []


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
        embedding_model_name: Optional[str] = None,
        conversation_summary: Optional[str] = None,
        additional_guidance: Optional[str] = None
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
                # Convert user query to embedding - rely purely on embedding model's semantic understanding
                # No query expansion - embedding models (BGE, MPNet, etc.) handle semantic similarity
                query_embeddings = self.embedding_service.generate_embeddings(
                    [user_query],
                    model_name=embedding_model_name
                )
                
                # Ensure query_embeddings is a 2D array (1 x dimension)
                if query_embeddings.ndim == 1:
                    query_embeddings = query_embeddings.reshape(1, -1)
                
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
                    
                    # Use semantic distance-based filtering only
                    # No hardcoded synonym matching - rely on embedding models
                    # The embedding model's semantic understanding will handle synonyms
                
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
        
        # Build enhanced schema context with structured metadata
        structured_schema_info = self._build_structured_schema_context(schema_details)
        
        # Condensed conversation context summary
        conversation_context_text = None
        if conversation_summary:
            conversation_context_text = conversation_summary.strip()
            if len(conversation_context_text) > 1200:
                conversation_context_text = conversation_context_text[-1200:]
        
        prompt = f"""You are an expert SQL query generator specializing in {database_type.upper()} databases. 
Convert the following natural language query into a valid {database_type.upper()} SQL query.

Database Type: {database_type.upper()}

=== STRUCTURED SCHEMA METADATA ===
{structured_schema_info}

=== SEMANTICALLY RELEVANT SCHEMA CHUNKS (from vectorized search) ===
{relevant_schema_text}

"""
        if conversation_context_text:
            prompt += f"""
=== CONVERSATION CONTEXT (recent messages) ===
{conversation_context_text}
"""

        if additional_guidance:
            prompt += f"""
=== VALIDATION / ADDITIONAL GUIDANCE ===
{additional_guidance}
"""

        prompt += f"""
=== USER QUERY ===
"{user_query}"

=== INSTRUCTIONS ===
1. **Schema Analysis**: Use BOTH the structured metadata (table names, relationships) AND the semantically matched chunks (detailed column info) to understand the database structure
2. **Table Selection**: Identify which tables are needed based on the query. Use the structured metadata to see all available tables and relationships
3. **Column Selection**: Use the semantically matched chunks to find exact column names, types, and constraints
4. **Relationships**: Use foreign key information from structured metadata to properly JOIN tables
5. **Query Generation**: 
   - Generate valid {database_type.upper()} SQL syntax
   - Use ONLY tables and columns that exist in the schema
   - Verify table/column names exactly (case-sensitive where applicable)
   - Include appropriate WHERE, JOIN, GROUP BY, ORDER BY, and LIMIT clauses
6. **Best Practices**:
   - Use aggregate functions (COUNT, SUM, AVG, MAX, MIN) when needed
   - Add GROUP BY when using aggregates with non-aggregated columns
   - Use proper JOIN syntax (INNER, LEFT, RIGHT) based on relationships
   - Respect data types and constraints
   - Limit results appropriately (LIMIT for PostgreSQL/MySQL, TOP for MSSQL)

=== IMPORTANT NOTES ===
- The structured metadata shows ALL tables and relationships in the database
- The semantic chunks show DETAILED information about columns relevant to your query
- Combine both sources to ensure accurate SQL generation
- If a table/column is mentioned in the query but not found in the schema, exclude it and note this in reasoning

=== OUTPUT FORMAT ===
Respond in JSON format:
{{
    "sql": "the SQL query here",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of how you used the schema information and generated the query",
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
            'sql': None,
            'confidence': 0.0,
            'reasoning': 'Failed to generate SQL',
            'tables_used': [],
            'columns_used': [],
            'relevant_schema_parts': relevant_schema_parts
        }
    
    def _build_structured_schema_context(self, schema_details: Dict[str, Any]) -> str:
        """
        Build structured schema context with metadata for better SQL generation
        Includes table names, relationships, and summary statistics
        """
        if not schema_details or 'tables' not in schema_details:
            return "No schema information available"
        
        lines = []
        metadata = schema_details.get('metadata', {})
        tables = schema_details.get('tables', [])
        
        # Summary
        lines.append(f"Database Summary:")
        lines.append(f"  - Total Tables: {metadata.get('total_tables', len(tables))}")
        lines.append(f"  - Total Columns: {metadata.get('total_columns', 0)}")
        lines.append(f"  - Database Type: {schema_details.get('database_type', 'unknown')}")
        lines.append("")
        
        # Table list
        table_names = [t.get('name', '') for t in tables if t.get('name')]
        if table_names:
            lines.append(f"All Tables in Database ({len(table_names)}):")
            lines.append(f"  {', '.join(sorted(table_names))}")
            lines.append("")
        
        # Relationships summary
        relationships = []
        for table in tables:
            fks = table.get('foreign_keys', [])
            for fk in fks:
                relationships.append(
                    f"{table.get('name')}.{', '.join(fk.get('constrained_columns', []))} -> "
                    f"{fk.get('referred_table')}.{', '.join(fk.get('referred_columns', []))}"
                )
        
        if relationships:
            lines.append("Key Relationships:")
            for rel in relationships[:20]:  # Limit to top 20 relationships
                lines.append(f"  - {rel}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_schema_for_llm(self, schema_details: Dict[str, Any]) -> str:
        """Format schema details for LLM prompt (detailed version)"""
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
                    raw_errors = result.get('errors', [])
                    raw_warnings = result.get('warnings', [])
                    user_friendly_errors = []
                    for error in raw_errors:
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
                        'warnings': raw_warnings or [],
                        'raw_errors': raw_errors or [],
                        'raw_warnings': raw_warnings or [],
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
                    'raw_errors': [],
                    'raw_warnings': ['Validation could not be performed - using basic syntax check'],
                    'validated_tables': [],
                    'validated_columns': []
                }
            else:
                return {
                    'valid': False,
                    'corrected_sql': None,
                    'errors': ['Invalid SQL statement type'],
                    'warnings': [],
                    'raw_errors': ['Invalid SQL statement type'],
                    'raw_warnings': [],
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
                'raw_errors': ['Unable to validate query'],
                'raw_warnings': [],
                'validated_tables': [],
                'validated_columns': []
            }
    
    def _build_structured_schema_context(self, schema_details: Dict[str, Any]) -> str:
        """
        Build structured schema context with metadata for better SQL generation
        Includes table names, relationships, and summary statistics
        """
        if not schema_details or 'tables' not in schema_details:
            return "No schema information available"
        
        lines = []
        metadata = schema_details.get('metadata', {})
        tables = schema_details.get('tables', [])
        
        # Summary
        lines.append(f"Database Summary:")
        lines.append(f"  - Total Tables: {metadata.get('total_tables', len(tables))}")
        lines.append(f"  - Total Columns: {metadata.get('total_columns', 0)}")
        lines.append(f"  - Database Type: {schema_details.get('database_type', 'unknown')}")
        lines.append("")
        
        # Table list
        table_names = [t.get('name', '') for t in tables if t.get('name')]
        if table_names:
            lines.append(f"All Tables in Database ({len(table_names)}):")
            lines.append(f"  {', '.join(sorted(table_names))}")
            lines.append("")
        
        # Relationships summary
        relationships = []
        for table in tables:
            fks = table.get('foreign_keys', [])
            for fk in fks:
                relationships.append(
                    f"{table.get('name')}.{', '.join(fk.get('constrained_columns', []))} -> "
                    f"{fk.get('referred_table')}.{', '.join(fk.get('referred_columns', []))}"
                )
        
        if relationships:
            lines.append("Key Relationships:")
            for rel in relationships[:20]:  # Limit to top 20 relationships
                lines.append(f"  - {rel}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_schema_for_llm(self, schema_details: Dict[str, Any]) -> str:
        """Format schema details for LLM prompt (detailed version)"""
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
        conversation_summary: Optional[str] = None,
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
            # Step 2a: Check if this is a schema metadata question
            intent_type = intent_result.get('intent_type', 'general_chat')
            
            # Handle schema metadata questions using full schema_details
            if intent_type == 'schema_question':
                schema_metadata_response = self._handle_schema_metadata_question(
                    user_query,
                    schema_details
                )
                if schema_metadata_response:
                    result['chat_response'] = schema_metadata_response
                    return result
            
            # Step 2b: General chat response (no SQL needed)
            chat_response = self._generate_chat_response(
                user_query,
                schema_details,
                schema_text,
                conversation_summary=conversation_summary
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
            embedding_model_name=embedding_model_name,
            conversation_summary=conversation_summary
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
            # Attempt a single refinement using validation feedback and conversation context
            guidance_text = self._build_validation_guidance(
                user_query=user_query,
                sql=result['sql'],
                validation_result=validation_result,
                schema_details=schema_details
            )
            
            if guidance_text:
                refined_summary = self._combine_summary_with_guidance(
                    conversation_summary,
                    guidance_text
                )
                summary_for_refinement = refined_summary or conversation_summary

                refined_sql_result = self.nl_to_sql_agent.generate_sql(
                    user_query,
                    schema_details,
                    database_type,
                    collection_name=collection_name,
                    embedding_model_name=embedding_model_name,
                    conversation_summary=summary_for_refinement,
                    additional_guidance=guidance_text
                )
                
                if refined_sql_result.get('sql') and refined_sql_result.get('sql') != result['sql']:
                    result['sql'] = refined_sql_result.get('sql')
                    if refined_sql_result.get('relevant_schema_parts'):
                        result['relevant_schema_parts'] = refined_sql_result.get('relevant_schema_parts')
                    
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
                if result.get('should_execute'):
                    # Query successfully refined; no need to send error response
                    pass
                else:
                    result['chat_response'] = f"I tried to generate a SQL query, but {error_message.lower()} Could you rephrase your question or provide more specific details?"
                    table_hint = self._build_table_column_summary(
                        self._extract_tables_from_sql(result['sql']),
                        schema_details
                    )
                    if table_hint:
                        result['chat_response'] += f"\n\nHere's what the schema shows for the tables involved:\n{table_hint}"
            else:
                if not result.get('should_execute'):
                    result['chat_response'] = "I generated a SQL query, but it needs verification. Could you provide more details about what you're looking for?"
        
        return result
    
    def _combine_summary_with_guidance(
        self,
        base_summary: Optional[str],
        guidance: Optional[str]
    ) -> Optional[str]:
        parts: List[str] = []
        if base_summary:
            cleaned = base_summary.strip()
            if cleaned:
                parts.append(cleaned)
        if guidance:
            cleaned_guidance = guidance.strip()
            if cleaned_guidance:
                parts.append(f"Validation feedback to incorporate:\n{cleaned_guidance}")
        if not parts:
            return None
        combined = "\n\n".join(parts)
        if len(combined) > 1500:
            combined = combined[-1500:]
        return combined

    def _build_validation_guidance(
        self,
        user_query: str,
        sql: Optional[str],
        validation_result: Dict[str, Any],
        schema_details: Dict[str, Any]
    ) -> Optional[str]:
        if not sql:
            return None
        
        raw_errors = validation_result.get('raw_errors') or validation_result.get('errors') or []
        raw_warnings = validation_result.get('raw_warnings') or validation_result.get('warnings') or []
        
        table_names = self._extract_tables_from_sql(sql)
        table_summary = self._build_table_column_summary(table_names, schema_details)
        relationship_summary = self._build_relationship_summary(table_names, schema_details)
        
        sections = []
        sections.append(f"User query: {user_query}")
        sections.append(f"Previous SQL attempt:\n{sql}")
        
        if raw_errors:
            sections.append("Validation feedback:")
            for err in raw_errors:
                sections.append(f"- {err}")
        if raw_warnings:
            sections.append("Warnings:")
            for warn in raw_warnings:
                sections.append(f"- {warn}")
        if table_summary:
            sections.append("Relevant table column details:")
            sections.append(table_summary)
        if relationship_summary:
            sections.append("Relevant relationships:")
            sections.append(relationship_summary)
        
        guidance = "\n".join(sections).strip()
        return guidance if guidance else None
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        if not sql:
            return []
        
        pattern = re.compile(r'\bFROM\s+([a-zA-Z0-9_."`]+)(?:\s+AS)?\s*(\w+)?|\bJOIN\s+([a-zA-Z0-9_."`]+)', re.IGNORECASE)
        matches = pattern.findall(sql)
        order_preserved = []
        seen = set()
        for match in matches:
            table_candidate = match[0] or match[2]
            if not table_candidate:
                continue
            cleaned = table_candidate.strip().strip('"`')
            # Remove schema prefix if present (e.g., public.table)
            if '.' in cleaned:
                cleaned = cleaned.split('.')[-1]
            if cleaned and cleaned.lower() not in seen:
                order_preserved.append(cleaned)
                seen.add(cleaned.lower())
        return order_preserved
    
    def _build_table_column_summary(self, table_names: List[str], schema_details: Dict[str, Any]) -> str:
        if not table_names or not schema_details:
            return ""
        
        tables = schema_details.get('tables', [])
        lines = []
        max_tables = 6
        max_columns = 15
        
        for table_name in table_names[:max_tables]:
            table_info = next(
                (t for t in tables if t.get('name', '').lower() == table_name.lower()),
                None
            )
            if not table_info:
                continue
            columns = table_info.get('columns', [])
            column_names = [col.get('name', '') for col in columns]
            if column_names:
                display_cols = column_names[:max_columns]
                col_text = ", ".join(display_cols)
                if len(column_names) > max_columns:
                    col_text += ", ..."
                lines.append(f"- {table_info.get('name')}: {col_text}")
            else:
                lines.append(f"- {table_info.get('name')}: (no column information available)")
        return "\n".join(lines)
    
    def _build_relationship_summary(self, table_names: List[str], schema_details: Dict[str, Any]) -> str:
        if not table_names or not schema_details:
            return ""
        
        tables = schema_details.get('tables', [])
        relevant_relationships = []
        
        # Outbound foreign keys
        for table in tables:
            if table.get('name', '').lower() not in [name.lower() for name in table_names]:
                continue
            for fk in table.get('foreign_keys', []):
                target_table = fk.get('referred_table')
                constrained_cols = ", ".join(fk.get('constrained_columns', []))
                referred_cols = ", ".join(fk.get('referred_columns', []))
                relevant_relationships.append(
                    f"{table.get('name')}({constrained_cols}) -> {target_table}({referred_cols})"
                )
        
        # Inbound foreign keys (other tables pointing to these tables)
        target_names = {name.lower() for name in table_names}
        for table in tables:
            for fk in table.get('foreign_keys', []):
                target_table = fk.get('referred_table', '').lower()
                if target_table in target_names:
                    constrained_cols = ", ".join(fk.get('constrained_columns', []))
                    referred_cols = ", ".join(fk.get('referred_columns', []))
                    relevant_relationships.append(
                        f"{table.get('name')}({constrained_cols}) -> {fk.get('referred_table')}({referred_cols})"
                    )
        
        if relevant_relationships:
            # Deduplicate while preserving order
            seen = set()
            unique_relationships = []
            for rel in relevant_relationships:
                if rel not in seen:
                    unique_relationships.append(rel)
                    seen.add(rel)
            max_relationships = 12
            trimmed = unique_relationships[:max_relationships]
            lines = [f"- {rel}" for rel in trimmed]
            if len(unique_relationships) > max_relationships:
                lines.append("- ...")
            return "\n".join(lines)
        
        return ""
    
    def _handle_schema_metadata_question(
        self,
        user_query: str,
        schema_details: Dict[str, Any]
    ) -> Optional[str]:
        """
        Handle schema metadata questions using full extracted schema details
        No hardcoded lists - uses actual schema_data structure
        """
        if not schema_details or 'tables' not in schema_details:
            return None
        
        query_lower = user_query.lower().strip()
        metadata = schema_details.get('metadata', {})
        tables = schema_details.get('tables', [])
        
        # Extract accurate counts from schema metadata
        total_tables = metadata.get('total_tables', len(tables))
        total_columns = metadata.get('total_columns', 0)
        
        # Count columns if not in metadata
        if total_columns == 0:
            for table in tables:
                total_columns += len(table.get('columns', []))
        
        # Get all table names from schema
        table_names = [table.get('name', '') for table in tables if table.get('name')]
        
        # Handle "how many tables" questions
        if any(phrase in query_lower for phrase in ['how many table', 'count of table', 'number of table', 'total table']):
            return f"There are **{total_tables} tables** in the database."
        
        # Handle "list all tables" questions
        if any(phrase in query_lower for phrase in ['list all table', 'show all table', 'what table', 'which table', 'name of table', 'table name', 'table available']):
            if table_names:
                table_list = ', '.join(sorted(table_names))
                return f"The database contains **{total_tables} tables**:\n\n{table_list}"
            else:
                return f"The database contains **{total_tables} tables**, but I couldn't retrieve their names."
        
        # Handle "how many columns" questions
        if any(phrase in query_lower for phrase in ['how many column', 'count of column', 'number of column', 'total column']):
            return f"There are **{total_columns} columns** across all tables in the database."
        
        # Handle "what columns in table X" questions
        # Extract table name from query
        for table_name in table_names:
            if table_name.lower() in query_lower:
                table_info = next((t for t in tables if t.get('name', '').lower() == table_name.lower()), None)
                if table_info:
                    columns = table_info.get('columns', [])
                    if columns:
                        column_list = ', '.join([col.get('name', '') for col in columns])
                        return f"The **{table_name}** table has **{len(columns)} columns**:\n\n{column_list}"
                    else:
                        return f"The **{table_name}** table exists but I couldn't retrieve its column information."
        
        # If no specific pattern matched, return None to fall back to general chat
        return None
    
    def _generate_chat_response(
        self,
        user_query: str,
        schema_details: Dict[str, Any],
        schema_text: str,
        conversation_summary: Optional[str] = None
    ) -> str:
        """Generate chat response for non-SQL queries"""
        # Build context with full schema metadata for accuracy
        context_parts = []
        
        # Include schema metadata for accurate answers
        if schema_details and 'metadata' in schema_details:
            metadata = schema_details['metadata']
            context_parts.append(
                f"Schema Metadata:\n"
                f"- Total Tables: {metadata.get('total_tables', 0)}\n"
                f"- Total Columns: {metadata.get('total_columns', 0)}\n"
                f"- Database Type: {schema_details.get('database_type', 'unknown')}\n"
            )
        
        # Include schema text
        if schema_text:
            context_parts.append(f"Database Schema Context:\n{schema_text[:1000]}")
        
        context = "\n\n".join(context_parts) if context_parts else "Database Schema Context: Not available"
        
        if conversation_summary:
            context = f"{context}\n\nConversation Summary:\n{conversation_summary}\n\nUse this summary instead of the full chat history."

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

