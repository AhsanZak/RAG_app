# Database Chat System - Key Code Samples

This document shows the exact code locations and samples of major processing steps.

---

## Main Entry Point

**File:** `backend/main.py`  
**Function:** `database_chat()`  
**Line:** 1322

```python
@app.post("/api/database/chat")
async def database_chat(chat_data: dict, db: Session = Depends(get_db)):
    """
    Main endpoint for database chat processing
    
    Flow:
    1. Validate input and get session/schema
    2. Initialize agent system
    3. Process query through agents
    4. Execute SQL if needed
    5. Save messages and vectorize history
    6. Return formatted response
    """
    try:
        # EXTRACT PARAMETERS (Lines 1328-1331)
        session_id = chat_data.get("session_id")
        message = chat_data.get("message")
        model_id = chat_data.get("model_id")
        user_id = chat_data.get("user_id", 1)
        
        # RETRIEVE SESSION & SCHEMA (Lines 1333-1363)
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
        schema = db.query(DatabaseSchema).filter(DatabaseSchema.session_id == session_id).first()
        connection = db.query(DatabaseConnection).filter(
            DatabaseConnection.id == schema.connection_id
        ).first()
        
        # INITIALIZE AGENT SYSTEM (Lines 1530-1539)
        agent_llm_service = LLMService()
        db_agent_system = DatabaseAgentSystem(
            agent_llm_service, 
            database_schema_service,
            embedding_service=embedding_service,
            chromadb_service=chromadb_service,
            model_config=model_config
        )
        
        # PROCESS QUERY (Lines 1558-1570)
        agent_result = db_agent_system.process_query(
            user_query=message,
            schema_details=schema.schema_data,
            schema_text=schema.schema_text or "",
            database_type=connection.database_type,
            collection_name=f"db_session_{session_id}",
            embedding_model_name=get_default_embedding_model(),
            # ... connection parameters
        )
        
        # EXECUTE SQL & FORMAT RESPONSE (Lines 1408-1508)
        # ... (see detailed sections below)
        
        return response_data
```

---

## Agent System Orchestration

**File:** `backend/services/db_agents.py`  
**Class:** `DatabaseAgentSystem`  
**Method:** `process_query()`  
**Line:** 596

```python
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
    Main orchestration method that coordinates all agents
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
    
    # STEP 1: INTENT DETECTION (Lines 622-631)
    intent_result = self.intent_agent.detect_intent(
        user_query, 
        schema_text,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name
    )
    result['intent'] = intent_result
    
    # STEP 2A: If no SQL needed, generate chat response
    if not intent_result.get('requires_sql'):
        result['chat_response'] = self._generate_chat_response(
            user_query, 
            schema_details, 
            schema_text,
            previous_context=None
        )
        return result
    
    # STEP 2B: Generate SQL (Lines 639-648)
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
        result['chat_response'] = "I couldn't generate a SQL query..."
        return result
    
    # STEP 3: Validate SQL (Lines 653-663)
    validation_result = self.sql_validation_agent.validate_and_correct_sql(
        result['sql'],
        schema_details,
        database_type,
        relevant_schema_parts=result.get('relevant_schema_parts', []),
        connection_string=connection_string,
        **connection_params
    )
    result['sql_validation'] = validation_result
    
    # STEP 4: Mark for execution if valid
    if validation_result.get('valid'):
        result['sql'] = validation_result.get('corrected_sql')
        result['should_execute'] = True
    else:
        # Generate helpful error message
        result['chat_response'] = "I tried to generate a SQL query, but..."
    
    return result
```

---

## Intent Detection - Detailed Code

**File:** `backend/services/db_agents.py`  
**Class:** `IntentDetectionAgent`  
**Method:** `detect_intent()`  
**Line:** 36

```python
def detect_intent(self, user_query: str, schema_context: Optional[str] = None,
                  collection_name: Optional[str] = None, 
                  embedding_model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Detects if user query requires SQL or is general chat
    
    Process:
    1. Quick keyword check
    2. Semantic search for relevant schema context
    3. LLM analysis for intent
    """
    # Quick keyword-based check (Line 49)
    user_query_lower = user_query.lower()
    has_sql_keywords = any(keyword in user_query_lower 
                          for keyword in ['query', 'select', 'find', 'get', ...])
    
    # SEMANTIC SEARCH FOR SCHEMA CONTEXT (Lines 54-86)
    relevant_context = schema_context
    if collection_name and self.embedding_service and self.chromadb_service:
        try:
            # Convert query to embedding
            query_embeddings = self.embedding_service.generate_embeddings(
                [user_query],
                model_name=embedding_model_name
            )
            
            # Ensure 2D array
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            
            # Query ChromaDB for similar schema chunks
            results = self.chromadb_service.query_collection(
                collection_name=collection_name,
                query_embeddings=query_embeddings,
                n_results=8
            )
            
            # Extract documents (schema chunks)
            documents = results.get('documents', [[]])[0]
            if documents:
                relevant_context = "\n\n".join(documents)
                print(f"[IntentDetection] Using {len(documents)} relevant schema chunks")
        except Exception as e:
            print(f"[IntentDetection] Semantic search failed: {str(e)}")
    
    # LLM-BASED INTENT ANALYSIS (Lines 88-126)
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
                'confidence': result.get('confidence', 0.8),
                'intent_type': result.get('intent_type', 'sql_query'),
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
```

---

## NL-to-SQL Conversion - Detailed Code

**File:** `backend/services/db_agents.py`  
**Class:** `NLToSQLAgent`  
**Method:** `generate_sql()`  
**Line:** 191

```python
def generate_sql(
    self, 
    user_query: str, 
    schema_details: Dict[str, Any],
    database_type: str,
    collection_name: Optional[str] = None,
    embedding_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Converts natural language to SQL using semantic search
    
    Key Steps:
    1. Semantic search for relevant schema parts
    2. Filter by relevance (distance-based)
    3. Generate SQL using LLM with relevant context
    """
    relevant_schema_text = None
    relevant_schema_parts = []
    
    # SEMANTIC SEARCH (Lines 216-274)
    if collection_name:
        try:
            # Expand query with linguistic variations (Line 219)
            expanded_queries = self._expand_query_with_synonyms(user_query)
            # Example: "movies" → ["movie", "films"] (simple variations)
            
            # Generate embeddings for all query variations (Lines 221-232)
            all_queries = [user_query] + expanded_queries
            query_embeddings = self.embedding_service.generate_embeddings(
                all_queries,
                model_name=embedding_model_name
            )
            
            # Use original query embedding for retrieval
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            else:
                query_embeddings = query_embeddings[0:1]
            
            # Query ChromaDB for similar schema chunks (Lines 235-238)
            results = self.chromadb_service.query_collection(
                collection_name=collection_name,
                query_embeddings=query_embeddings,
                n_results=15  # Get top 15 most similar
            )
            
            # Extract results (Lines 241-243)
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            # FILTER BY RELEVANCE (Lines 246-264)
            relevant_docs = []
            if documents:
                max_results_to_use = min(10, len(documents))
                
                # Sort by distance (ascending = most relevant first)
                doc_dist_pairs = list(zip(documents, metadatas, distances))
                doc_dist_pairs.sort(key=lambda x: x[2])
                
                # Take top 70% by distance, but at least top 5
                threshold_index = max(5, int(len(doc_dist_pairs) * 0.7))
                top_pairs = doc_dist_pairs[:max(max_results_to_use, threshold_index)]
                
                # Extract relevant documents
                relevant_docs = [doc for doc, meta, dist in top_pairs]
                relevant_schema_parts = relevant_docs.copy()
            
            # Combine into single context (Lines 270-273)
            if relevant_docs:
                relevant_schema_text = "\n\n".join(relevant_docs)
                print(f"[NLToSQL] Using {len(relevant_docs)} relevant schema chunks from semantic search")
        except Exception as e:
            print(f"[NLToSQL] Semantic search failed: {str(e)}, falling back to full schema")
    
    # FALLBACK TO FULL SCHEMA (Lines 277-280)
    if not relevant_schema_text:
        schema_text = self._format_schema_for_llm(schema_details)
        relevant_schema_text = schema_text
    
    # GENERATE SQL USING LLM (Lines 282-316)
    prompt = f"""You are an expert SQL query generator specializing in {database_type.upper()} databases. 
Convert the following natural language query into a valid {database_type.upper()} SQL query.

Database Type: {database_type.upper()}

Relevant Schema Information (semantically matched from database):
{relevant_schema_text}

User Query: "{user_query}"

Instructions:
1. Carefully analyze the schema structure including table names, columns, data types, constraints, and relationships
2. Generate a valid {database_type.upper()} SQL query that matches the query intent
3. Use ONLY tables and columns that exist in the provided schema - verify names exactly
4. Pay attention to foreign key relationships to properly JOIN related tables
...
"""
    
    try:
        response = self.llm_service.generate_response(
            messages=[{"role": "user", "content": prompt}],
            model_config=self.model_config,
            temperature=0.1
        )
        
        # EXTRACT SQL FROM RESPONSE (Lines 318-340)
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
                # Try to extract SQL directly from response
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
```

---

## SQL Execution - Detailed Code

**File:** `backend/services/sql_execution_service.py`  
**Class:** `SQLExecutionService`  
**Method:** `execute_query()`  
**Line:** 20

```python
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
    Executes SQL query on database and returns results
    
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
        # BUILD CONNECTION STRING (Lines 48-52)
        if not connection_string:
            connection_string = self._build_connection_string(
                database_type, host, port, database_name, username, password
            )
        
        # CREATE ENGINE (Line 55)
        engine = create_engine(connection_string, echo=False)
        
        # EXECUTE QUERY (Lines 57-87)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            
            # Process SELECT queries (Lines 61-76)
            if result.returns_rows:
                columns = list(result.keys())
                rows = result.fetchall()
                
                # Convert rows to dictionaries
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
                # Process INSERT/UPDATE/DELETE (Lines 77-87)
                rowcount = result.rowcount
                return {
                    'success': True,
                    'data': None,
                    'columns': None,
                    'row_count': rowcount,
                    'error': None,
                    'execution_time': time.time() - start_time
                }
    
    # ERROR HANDLING (Lines 89-120)
    except SQLAlchemyError as e:
        error_msg = str(e)
        
        # Format user-friendly error message
        user_friendly_error = self._format_error_message(error_msg)
        
        return {
            'success': False,
            'data': None,
            'columns': None,
            'row_count': 0,
            'error': user_friendly_error,  # User-friendly, not technical
            'error_type': 'sql_error',
            'execution_time': time.time() - start_time
        }
    except Exception as e:
        error_msg = str(e)
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
        # Cleanup
        try:
            engine.dispose()
        except:
            pass
```

---

## Response Formatting - Detailed Code

**File:** `backend/main.py`  
**Function:** `database_chat()`  
**Section:** SQL Execution and Response Formatting  
**Lines:** 1422-1508

```python
# EXECUTE SQL IF NEEDED (Lines 1422-1508)
if agent_result.get('should_execute') and agent_result.get('sql'):
    # Execute SQL (Lines 1425-1434)
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
    
    sql_executed = True
    query_result = sql_result
    
    # FORMAT SUCCESSFUL RESULTS (Lines 1439-1483)
    if sql_result.get('success'):
        query_data = sql_result.get('data', [])
        row_count = sql_result.get('row_count', 0)
        
        # Prepare context for LLM response generation
        if row_count > 0 and len(query_data) <= 100:
            results_summary = json.dumps(query_data, indent=2, default=str)
            context_message = f"SQL Query executed successfully and returned {row_count} rows.\n\nQuery Results:\n{results_summary[:2000]}"
        else:
            context_message = f"SQL Query executed successfully and returned {row_count} rows."
        
        # Generate natural language response using LLM
        response_text = db_agent_system._generate_chat_response(
            user_query=message,
            schema_details=schema.schema_data,
            schema_text=schema.schema_text or "",
            previous_context=previous_context + [
                {"role": "assistant", "content": context_message}
            ] if previous_context else [
                {"role": "assistant", "content": context_message}
            ]
        )
        
        # Add query results to metadata (Lines 1463-1475)
        if row_count > 0:
            metadata['query_result'] = {
                'row_count': row_count,
                'columns': sql_result.get('columns', []),
                'sample_data': query_data[:10] if len(query_data) <= 100 else []
            }
    
    # HANDLE EXECUTION ERRORS (Lines 1484-1508)
    else:
        error_msg = sql_result.get('error', 'Unknown error occurred')
        error_type = sql_result.get('error_type', 'execution_error')
        
        # Generate user-friendly error messages (no technical details)
        if 'table' in error_msg.lower() or 'column' in error_msg.lower():
            response_text = "I couldn't execute the query because some tables or columns referenced don't exist in the database. Could you rephrase your question?"
        elif 'syntax' in error_msg.lower():
            response_text = "There was an issue with the generated SQL query. Let me try to understand your question better - could you rephrase it?"
        elif 'permission' in error_msg.lower():
            response_text = "I don't have permission to execute this type of query on the database."
        else:
            response_text = "I encountered an issue executing the query. Could you try rephrasing your question?"
        
        # Store error in metadata (for backend logging only)
        metadata['query_error'] = {
            'type': error_type,
            'message': error_msg  # Technical error for backend only
        }
        metadata['sql'] = None  # Don't expose failed SQL
```

---

## Chat History Vectorization - Detailed Code

**File:** `backend/main.py`  
**Function:** `database_chat()`  
**Section:** Chat History Vectorization  
**Lines:** 1565-1604

```python
# VECTORIZE CHAT HISTORY (Lines 1565-1604)
collection_name = f"db_session_{session_id}"

try:
    # Get all messages for this session
    all_messages = db.query(ChatMessageModel).filter(
        ChatMessageModel.session_id == session_id
    ).order_by(ChatMessageModel.created_at).all()
    
    # Prepare recent messages for vectorization (last 10)
    recent_messages = all_messages[-10:]
    chat_texts = []
    chat_metadatas = []
    chat_ids = []
    
    # Format messages for embedding (Lines 1577-1587)
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
    
    # Generate embeddings (Lines 1589-1594)
    if chat_texts:
        embeddings = embedding_service.generate_embeddings(
            chat_texts, 
            model_name=get_default_embedding_model()
        )
        
        # Add to ChromaDB (Lines 1596-1613)
        try:
            # Get existing collection or create
            collections = chromadb_service.list_collections()
            if collection_name not in collections:
                chromadb_service.create_collection(collection_name)
            
            # Add messages to collection
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
```

**What Happens:**
- Last 10 messages are vectorized and stored
- Each message gets an embedding vector
- Stored in ChromaDB collection for semantic search
- Future queries can use this context for better understanding

---

## Schema Chunking - Detailed Code

**File:** `backend/main.py`  
**Function:** `_create_enriched_schema_chunks()`  
**Line:** 16

```python
def _create_enriched_schema_chunks(schema_data: dict, schema_text: str) -> List[dict]:
    """
    Creates comprehensive schema chunks based on actual database structure
    No hardcoded mappings - purely schema-driven
    """
    chunks = []
    
    if schema_data and 'tables' in schema_data:
        tables = schema_data['tables']
        
        # Create comprehensive table-level chunks (Lines 26-102)
        for table in tables:
            table_name = table.get('name', '')
            
            description_parts = []
            
            # Table header (Line 34)
            description_parts.append(f"Database Table: {table_name}")
            
            # Schema context (Lines 37-39)
            schema_name = table.get('schema') or schema_data.get('schema_name')
            if schema_name:
                description_parts.append(f"Schema: {schema_name}")
            
            # Detailed columns (Lines 41-55)
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
            
            # Primary keys (Lines 57-60)
            if table.get('primary_keys'):
                pk_columns = ', '.join(table['primary_keys'])
                description_parts.append(f"Primary Key: {pk_columns}")
            
            # Foreign keys (Lines 62-69)
            if table.get('foreign_keys'):
                description_parts.append("Foreign Key Relationships:")
                for fk in table['foreign_keys']:
                    fk_cols = ', '.join(fk.get('constrained_columns', []))
                    ref_table = fk.get('referred_table', '')
                    ref_cols = ', '.join(fk.get('referred_columns', []))
                    description_parts.append(f"  {fk_cols} references {ref_table}({ref_cols})")
            
            # Indexes (Lines 71-77)
            if table.get('indexes'):
                description_parts.append("Indexes:")
                for idx in table['indexes']:
                    idx_cols = ', '.join(idx.get('columns', []))
                    unique = "UNIQUE " if idx.get('unique') else ""
                    description_parts.append(f"  {unique}Index: {idx_cols}")
            
            # Create table chunk (Lines 79-102)
            table_chunk_text = '\n'.join(description_parts)
            
            # Add column names list for better semantic matching (Lines 84-86)
            column_names = [col.get('name', '') for col in table.get('columns', [])]
            if column_names:
                table_chunk_text += f"\nColumn Names: {', '.join(column_names)}"
            
            # Add related tables (Lines 88-92)
            if table.get('foreign_keys'):
                related_tables = set([fk.get('referred_table', '') 
                                     for fk in table.get('foreign_keys', [])])
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
            
            # Create individual column chunks (Lines 104-126)
            # Create relationship chunks (Lines 128-149)
            # ... (see full code)
    
    return chunks
```

**Example Chunk Output:**
```
Database Table: film
Schema: public
Table Columns:
  film_id: integer (NOT NULL)
  title: varchar (NOT NULL)
  description: text
  release_year: integer
  language_id: smallint (NOT NULL)
Primary Key: film_id
Foreign Key Relationships:
  language_id references language(language_id)
Column Names: film_id, title, description, release_year, language_id
Related Tables: language
```

---

## Complete Processing Flow Example

### Example: "How many movies were released in 2006?"

**Step 1: Intent Detection**
```python
# Query: "How many movies were released in 2006?"
# Semantic search finds: film table chunks, release_year column chunks
# LLM determines: requires_sql=True, confidence=0.95
```

**Step 2: NL-to-SQL**
```python
# Semantic search retrieves:
# - "Database Table: film" chunk
# - "Column: film.release_year" chunk
# - Related chunks about film table

# LLM generates SQL:
sql = "SELECT COUNT(*) FROM film WHERE release_year = 2006;"
```

**Step 3: SQL Validation**
```python
# Validates: film table exists ✓, release_year column exists ✓
# Result: valid=True, corrected_sql="SELECT COUNT(*) FROM film WHERE release_year = 2006;"
```

**Step 4: SQL Execution**
```python
# Executes on PostgreSQL
# Returns: {'success': True, 'data': [{'count': 42}], 'row_count': 1}
```

**Step 5: Response Generation**
```python
# LLM formats response:
response_text = "There are 42 movies that were released in 2006."
```

**Step 6: Save & Vectorize**
```python
# Save user message and assistant response
# Vectorize last 10 messages for future context
```

**Step 7: Return to Frontend**
```json
{
    "response": "There are 42 movies that were released in 2006.",
    "sql": "SELECT COUNT(*) FROM film WHERE release_year = 2006;",
    "sql_executed": true,
    "query_result": [{"count": 42}]
}
```

---

## Key Data Structures

### Agent Result Structure
```python
{
    'requires_sql': bool,
    'intent': {
        'requires_sql': bool,
        'confidence': float,
        'intent_type': str,
        'reasoning': str
    },
    'sql': str or None,
    'sql_validation': {
        'valid': bool,
        'corrected_sql': str or None,
        'errors': List[str],
        'warnings': List[str]
    },
    'should_execute': bool,
    'chat_response': str or None,
    'relevant_schema_parts': List[str]
}
```

### SQL Execution Result Structure
```python
{
    'success': bool,
    'data': List[Dict] or None,  # [{col1: val1, col2: val2}, ...]
    'columns': List[str] or None,  # ['col1', 'col2', ...]
    'row_count': int,
    'error': str or None,
    'error_type': str or None,
    'execution_time': float
}
```

### Final Response Structure
```python
{
    "response": str,  # Natural language response
    "message_id": int,
    "session_id": int,
    "requires_sql": bool,
    "sql_executed": bool,
    "sql": str or None,  # Only if successful
    "query_result": List[Dict] or None,  # Only if successful
    "metadata": {
        "intent": Dict,
        "used_semantic_search": bool,
        "relevant_schema_parts_count": int
    }
}
```

---

## Semantic Search Deep Dive

### How Embeddings Work

**1. Schema Vectorization (During Processing):**
```python
# Each chunk is converted to embedding vector
chunk_text = "Database Table: film\nColumns: film_id, title, ..."
embedding = embedding_model.encode(chunk_text)
# Result: [0.123, -0.456, 0.789, ..., 0.234] (384 or 768 dimensions)
```

**2. Query Vectorization (During Query):**
```python
query = "How many movies are there?"
query_embedding = embedding_model.encode(query)
# Result: [0.125, -0.458, 0.791, ..., 0.236] (similar to film table!)
```

**3. Similarity Calculation (ChromaDB):**
```python
# Cosine similarity between query and schema chunks
similarity = cosine_similarity(query_embedding, schema_chunk_embedding)
distance = 1 - similarity  # Lower distance = more similar

# "movies" query vs "film" table chunk: distance ≈ 0.15 (very similar!)
# "movies" query vs "customer" table chunk: distance ≈ 0.85 (not similar)
```

**4. Retrieval:**
```python
# ChromaDB returns top N most similar chunks
results = chromadb.query_collection(
    query_embeddings=query_embedding,
    n_results=15
)
# Returns chunks sorted by similarity (distance)
```

---

## Performance Metrics

### Typical Processing Times:
- **Intent Detection**: 0.5-1.5 seconds (LLM + semantic search)
- **NL-to-SQL**: 1-3 seconds (semantic search + LLM generation)
- **SQL Validation**: 0.5-1.5 seconds (LLM validation)
- **SQL Execution**: 0.01-0.5 seconds (depends on query complexity)
- **Response Generation**: 0.5-2 seconds (LLM formatting)
- **Total**: ~3-8 seconds for complete processing

### Optimization:
- Schema chunks are pre-vectorized (cached in ChromaDB)
- Chat history vectorized incrementally (only last 10 messages)
- LLM calls use temperature=0.1 for consistency
- Semantic search limited to top 15 chunks, filtered to top 70%

---

## Error Handling Flow

### SQL Generation Fails:
```python
# NL-to-SQL agent returns: {'sql': '', 'confidence': 0.0}
# System generates: "I couldn't generate a SQL query for your request..."
# No SQL exposed to frontend
```

### SQL Validation Fails:
```python
# Validation agent returns: {'valid': False, 'errors': [...]}
# System generates: "I tried to generate a SQL query, but some tables or columns don't match..."
# No SQL exposed to frontend
```

### SQL Execution Fails:
```python
# Execution service returns: {'success': False, 'error': 'user-friendly message'}
# System generates helpful error message
# Technical error stored in metadata (backend only)
# No failed SQL exposed to frontend
```

---

This completes the detailed explanation of the database chat processing flow with code samples and examples.

