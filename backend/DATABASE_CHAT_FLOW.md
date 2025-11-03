# Database Chat Processing Flow - Detailed Explanation

## Overview

This document explains in detail how the database chat system processes user prompts, from receiving the query to returning SQL results. The system uses a multi-agent architecture with semantic search, intent detection, NL-to-SQL conversion, SQL validation, and execution.

---

## Architecture Overview

```
User Query
    ↓
[1. Intent Detection Agent] → Determines if SQL is needed
    ↓
[2. NL-to-SQL Agent] → Converts natural language to SQL (uses semantic search)
    ↓
[3. SQL Validation Agent] → Validates and corrects SQL against schema
    ↓
[4. SQL Execution Service] → Executes SQL on database
    ↓
[5. Response Generation] → Formats results for user
    ↓
[6. Chat History Vectorization] → Stores conversation context
    ↓
Response to Frontend
```

---

## Step-by-Step Processing Flow

### Step 1: User Sends Query

**Frontend Request:**
```javascript
// frontend/src/services/api.js
const response = await databaseChatAPI.chat(
    sessionId,
    messageText,
    modelId
);
```

**Backend Endpoint:**
```python
# backend/main.py - Line 1322
@app.post("/api/database/chat")
async def database_chat(
    chat_data: dict,
    db: Session = Depends(get_db)
):
```

**Extracted Parameters:**
```python
session_id = chat_data.get("session_id")
message = chat_data.get("message")  # User's natural language query
model_id = chat_data.get("model_id")  # Selected LLM model
user_id = chat_data.get("user_id", 1)
```

---

### Step 2: Retrieve Session and Schema Information

**Code Location:** `backend/main.py` lines 1333-1363

```python
# Get chat session
session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

# Get LLM model configuration
model = db.query(LLMModel).filter(LLMModel.id == model_id).first()
model_config = {
    "provider": model.provider.value,
    "model_name": model.model_name,
    "base_url": model.base_url,
    "api_key": model.api_key if hasattr(model, 'api_key') else None
}

# Get database schema associated with this session
schema = db.query(DatabaseSchema).filter(
    DatabaseSchema.session_id == session_id
).first()

# Get database connection details
connection = db.query(DatabaseConnection).filter(
    DatabaseConnection.id == schema.connection_id
).first()
```

**What Happens:**
- Retrieves the chat session from database
- Gets the LLM model configuration (Ollama, OpenAI, etc.)
- Retrieves the vectorized schema stored for this session
- Gets database connection details (host, port, credentials)

---

### Step 3: Initialize Agent System

**Code Location:** `backend/main.py` lines 1365-1374

```python
# Create agent system with services
agent_llm_service = LLMService()
db_agent_system = DatabaseAgentSystem(
    agent_llm_service,           # LLM service for generating responses
    database_schema_service,      # Schema extraction service
    embedding_service=embedding_service,      # For semantic search
    chromadb_service=chromadb_service,       # Vector database
    model_config=model_config                # LLM model config
)
```

**Components Initialized:**
- `IntentDetectionAgent`: Detects if query needs SQL
- `NLToSQLAgent`: Converts natural language to SQL
- `SQLValidationAgent`: Validates SQL against schema
- All agents share the same embedding/chromadb services for semantic search

---

### Step 4: Get Chat History Context

**Code Location:** `backend/main.py` lines 1376-1378

```python
# Get last 10 messages for conversation context
previous_messages = db.query(ChatMessageModel).filter(
    ChatMessageModel.session_id == session_id
).order_by(ChatMessageModel.created_at).limit(10).all()

previous_context = [
    {"role": msg.role, "content": msg.message} 
    for msg in previous_messages
] if previous_messages else None
```

**What Happens:**
- Retrieves last 10 messages from the conversation
- Formats them for LLM context
- Provides conversational continuity

---

### Step 5: Process Query Through Agent System

**Code Location:** `backend/main.py` lines 1380-1406

```python
# Collection name for semantic search
collection_name = f"db_session_{session_id}"

# Get embedding model name (default or selected)
embedding_model_name = get_default_embedding_model()

# Process query through agent pipeline
agent_result = db_agent_system.process_query(
    user_query=message,
    schema_details=schema.schema_data,        # Full schema JSON
    schema_text=schema.schema_text,          # Human-readable schema text
    database_type=connection.database_type,    # postgresql, mysql, etc.
    collection_name=collection_name,          # ChromaDB collection for semantic search
    embedding_model_name=embedding_model_name, # Embedding model for semantic search
    connection_string=connection.connection_string,
    host=connection.host,
    port=connection.port,
    database_name=connection.database_name,
    username=connection.username,
    password=connection.password
)
```

**Key Parameters:**
- `collection_name`: Points to ChromaDB collection with vectorized schema
- `embedding_model_name`: Model used for semantic similarity (e.g., BGE, MPNet)
- `schema_details`: Complete schema structure as JSON
- `schema_text`: Human-readable schema representation

---

## Detailed Agent Processing

### 5.1 Intent Detection Agent

**Code Location:** `backend/services/db_agents.py` lines 19-178

**Process Flow:**

```python
class IntentDetectionAgent:
    def detect_intent(self, user_query: str, schema_context: Optional[str] = None,
                      collection_name: Optional[str] = None, 
                      embedding_model_name: Optional[str] = None) -> Dict[str, Any]:
        
        # Step 1: Quick keyword check
        has_sql_keywords = any(keyword in user_query.lower() 
                              for keyword in ['query', 'select', 'find', 'get', ...])
        
        # Step 2: Semantic search for relevant schema context
        if collection_name and self.embedding_service:
            # Convert user query to embedding
            query_embeddings = self.embedding_service.generate_embeddings(
                [user_query],
                model_name=embedding_model_name
            )
            
            # Retrieve relevant schema chunks (top 8)
            results = self.chromadb_service.query_collection(
                collection_name=collection_name,
                query_embeddings=query_embeddings,
                n_results=8
            )
            
            # Extract documents (schema chunks)
            documents = results.get('documents', [[]])[0]
            if documents:
                relevant_context = "\n\n".join(documents)
        
        # Step 3: Use LLM to analyze intent
        prompt = f"""Analyze the following user query...
        User Query: "{user_query}"
        Relevant Schema Context: {relevant_context[:800]}
        
        Determine:
        1. Does this require SQL? (requires_sql: true/false)
        2. Intent type (sql_query, schema_question, general_chat)
        3. Confidence level (0.0 to 1.0)
        """
        
        response = self.llm_service.generate_response(
            messages=[{"role": "user", "content": prompt}],
            model_config=self.model_config,
            temperature=0.1
        )
        
        # Step 4: Parse LLM response
        result = {
            'requires_sql': True/False,
            'confidence': 0.0-1.0,
            'intent_type': 'sql_query'|'schema_question'|'general_chat',
            'reasoning': 'explanation'
        }
```

**Example:**
- Query: "Show me all movies"
- Semantic search finds: `film` table chunks (movies→film semantic match)
- LLM analyzes: `requires_sql: true`, `intent_type: sql_query`, `confidence: 0.9`

---

### 5.2 NL-to-SQL Agent

**Code Location:** `backend/services/db_agents.py` lines 180-340

**Process Flow:**

```python
class NLToSQLAgent:
    def generate_sql(self, user_query: str, schema_details: Dict[str, Any],
                     database_type: str, collection_name: Optional[str] = None,
                     embedding_model_name: Optional[str] = None) -> Dict[str, Any]:
        
        relevant_schema_text = None
        relevant_schema_parts = []
        
        # Step 1: Semantic search for relevant schema chunks
        if collection_name:
            # Expand query with linguistic variations (not hardcoded synonyms)
            expanded_queries = self._expand_query_with_synonyms(user_query)
            # Example: "movies" → ["movie", "films"] (simple plural variations)
            
            # Convert queries to embeddings
            all_queries = [user_query] + expanded_queries
            query_embeddings = self.embedding_service.generate_embeddings(
                all_queries,
                model_name=embedding_model_name
            )
            
            # Use original query embedding for retrieval
            query_embeddings = query_embeddings[0:1]
            
            # Retrieve top 15 most semantically similar schema chunks
            results = self.chromadb_service.query_collection(
                collection_name=collection_name,
                query_embeddings=query_embeddings,
                n_results=15
            )
            
            # Extract results
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            # Step 2: Filter by relevance
            doc_dist_pairs = list(zip(documents, metadatas, distances))
            doc_dist_pairs.sort(key=lambda x: x[2])  # Sort by distance (lower = more similar)
            
            # Take top 70% most relevant, but at least top 5
            threshold_index = max(5, int(len(doc_dist_pairs) * 0.7))
            top_pairs = doc_dist_pairs[:threshold_index]
            
            relevant_docs = [doc for doc, meta, dist in top_pairs]
            relevant_schema_parts = relevant_docs.copy()
            
            # Combine into single text
            if relevant_docs:
                relevant_schema_text = "\n\n".join(relevant_docs)
                print(f"[NLToSQL] Using {len(relevant_docs)} relevant schema chunks")
        
        # Step 3: Fallback to full schema if semantic search failed
        if not relevant_schema_text:
            schema_text = self._format_schema_for_llm(schema_details)
            relevant_schema_text = schema_text
        
        # Step 4: Generate SQL using LLM
        prompt = f"""You are an expert SQL query generator specializing in {database_type.upper()} databases.

Database Type: {database_type.upper()}

Relevant Schema Information (semantically matched from database):
{relevant_schema_text}

User Query: "{user_query}"

Instructions:
1. Carefully analyze the schema structure including table names, columns, data types, constraints, and relationships
2. Generate a valid {database_type.upper()} SQL query that matches the query intent
3. Use ONLY tables and columns that exist in the provided schema
4. Pay attention to foreign key relationships to properly JOIN related tables
5. Include appropriate WHERE clauses based on the query intent
6. Use aggregate functions when the query requires calculations
7. Add GROUP BY when using aggregate functions
8. Add ORDER BY when the query should return sorted results
9. Limit results appropriately using LIMIT or TOP clauses

Respond in JSON format:
{{
    "sql": "the SQL query here",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of the query",
    "tables_used": ["table1", "table2"],
    "columns_used": ["column1", "column2"]
}}
"""
        
        # Step 5: Call LLM
        response = self.llm_service.generate_response(
            messages=[{"role": "user", "content": prompt}],
            model_config=self.model_config,
            temperature=0.1  # Low temperature for consistent SQL generation
        )
        
        # Step 6: Parse LLM response and extract SQL
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'sql': result.get('sql', '').strip(),
                'confidence': result.get('confidence', 0.7),
                'reasoning': result.get('reasoning', ''),
                'tables_used': result.get('tables_used', []),
                'columns_used': result.get('columns_used', []),
                'relevant_schema_parts': relevant_schema_parts
            }
```

**Example Flow:**
1. User query: "How many movies are there?"
2. Semantic search finds:
   - `film` table chunk (movies→film semantic match via embeddings)
   - Column chunks: `film.film_id`, `film.title`
   - Relevant schema context about film table
3. LLM generates SQL:
   ```sql
   SELECT COUNT(*) FROM film;
   ```
4. Returns:
   ```python
   {
       'sql': 'SELECT COUNT(*) FROM film;',
       'confidence': 0.95,
       'tables_used': ['film'],
       'columns_used': []
   }
   ```

---

### 5.3 SQL Validation Agent

**Code Location:** `backend/services/db_agents.py` lines 329-477

**Process Flow:**

```python
class SQLValidationAgent:
    def validate_and_correct_sql(self, sql: str, schema_details: Dict[str, Any],
                                 database_type: str,
                                 relevant_schema_parts: Optional[List[str]] = None) -> Dict[str, Any]:
        
        # Step 1: Use relevant schema parts if available
        if relevant_schema_parts and len(relevant_schema_parts) > 0:
            schema_text = "\n\n".join(relevant_schema_parts)
        else:
            schema_text = self._format_schema_for_llm(schema_details)
        
        # Step 2: Create validation prompt
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
        
        # Step 3: Call LLM for validation
        response = self.llm_service.generate_response(
            messages=[{"role": "user", "content": prompt}],
            model_config=self.model_config,
            temperature=0.1
        )
        
        # Step 4: Parse validation result
        result = json.loads(json_match.group())
        
        # Step 5: Sanitize errors (make user-friendly)
        errors = result.get('errors', [])
        user_friendly_errors = []
        for error in errors:
            error_lower = str(error).lower()
            if 'table' in error_lower or 'does not exist' in error_lower:
                user_friendly_errors.append("Table or column mismatch")
            elif 'syntax' in error_lower:
                user_friendly_errors.append("Syntax issue detected")
            else:
                user_friendly_errors.append("Validation issue")
        
        return {
            'valid': result.get('valid', False),
            'corrected_sql': result.get('corrected_sql') if result.get('valid') else None,
            'errors': user_friendly_errors if not result.get('valid') else [],
            'warnings': result.get('warnings', []),
            'validated_tables': result.get('validated_tables', []),
            'validated_columns': result.get('validated_columns', [])
        }
```

**Example:**
- Input SQL: `SELECT * FROM films;` (wrong table name)
- Validation finds: Table should be `film`, not `films`
- Returns:
  ```python
  {
      'valid': False,
      'corrected_sql': None,
      'errors': ['Table or column mismatch']
  }
  ```

---

### 5.4 SQL Execution Service

**Code Location:** `backend/services/sql_execution_service.py`

**Process Flow:**

```python
class SQLExecutionService:
    def execute_query(self, sql: str, database_type: str, 
                     connection_string: Optional[str] = None,
                     host: Optional[str] = None, port: Optional[int] = None,
                     database_name: Optional[str] = None,
                     username: Optional[str] = None,
                     password: Optional[str] = None) -> Dict[str, Any]:
        
        # Step 1: Build connection string if not provided
        if not connection_string:
            connection_string = self._build_connection_string(
                database_type, host, port, database_name, username, password
            )
        
        # Step 2: Create SQLAlchemy engine
        engine = create_engine(connection_string, echo=False)
        
        # Step 3: Execute SQL query
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            
            # Step 4: Process results
            if result.returns_rows:
                # SELECT query
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
                # INSERT/UPDATE/DELETE
                return {
                    'success': True,
                    'data': None,
                    'columns': None,
                    'row_count': result.rowcount,
                    'error': None,
                    'execution_time': time.time() - start_time
                }
```

**Error Handling:**
```python
except SQLAlchemyError as e:
    error_msg = str(e)
    
    # Format user-friendly error messages
    user_friendly_error = self._format_error_message(error_msg)
    # Converts technical errors to user-friendly messages
    
    return {
        'success': False,
        'error': user_friendly_error,  # User-friendly, not technical
        'error_type': 'sql_error'
    }
```

---

### 5.5 Response Generation and Formatting

**Code Location:** `backend/main.py` lines 1408-1508

**Process Flow:**

```python
# Step 1: Prepare response metadata
relevant_schema_parts = agent_result.get('relevant_schema_parts', [])
metadata = {
    "intent": agent_result.get('intent', {}),
    "requires_sql": agent_result.get('requires_sql', False),
    "sql": agent_result.get('sql'),
    "sql_validation": agent_result.get('sql_validation', {}),
    "relevant_schema_parts_count": len(relevant_schema_parts),
    "used_semantic_search": len(relevant_schema_parts) > 0
}

# Step 2: Execute SQL if needed
if agent_result.get('should_execute') and agent_result.get('sql'):
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
    
    # Step 3: Format successful results
    if sql_result.get('success'):
        query_data = sql_result.get('data', [])
        row_count = sql_result.get('row_count', 0)
        
        # Generate natural language response using LLM
        if row_count > 0 and len(query_data) <= 100:
            results_summary = json.dumps(query_data, indent=2, default=str)
            context_message = f"SQL Query executed successfully and returned {row_count} rows.\n\nQuery Results:\n{results_summary[:2000]}"
        else:
            context_message = f"SQL Query executed successfully and returned {row_count} rows."
        
        # Use LLM to format response
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
        
        # Add query results to metadata
        metadata['query_result'] = {
            'row_count': row_count,
            'columns': sql_result.get('columns', []),
            'sample_data': query_data[:10] if len(query_data) <= 100 else []
        }
    
    # Step 4: Handle errors
    else:
        # Generate user-friendly error message (no technical details exposed)
        error_msg = sql_result.get('error', 'Unknown error occurred')
        
        if 'table' in error_msg.lower() or 'column' in error_msg.lower():
            response_text = "I couldn't execute the query because some tables or columns referenced don't exist in the database. Could you rephrase your question?"
        elif 'syntax' in error_msg.lower():
            response_text = "There was an issue with the generated SQL query. Let me try to understand your question better - could you rephrase it?"
        else:
            response_text = "I encountered an issue executing the query. Could you try rephrasing your question?"
        
        metadata['sql'] = None  # Don't expose failed SQL

# Step 5: Handle non-SQL queries
elif agent_result.get('chat_response'):
    # General chat response (no SQL needed)
    response_text = agent_result['chat_response']
else:
    # Fallback to RAG
    rag_result = rag_service.query(
        query_text=message,
        collection_name=collection_name,
        model_config=model_config,
        embedding_model_name=embedding_model_name,
        n_results=5
    )
    response_text = rag_result["response"]
```

---

### Step 6: Save Messages and Vectorize Chat History

**Code Location:** `backend/main.py` lines 1533-1604

**Process Flow:**

```python
# Step 1: Save user message
user_message = ChatMessageModel(
    session_id=session_id,
    role="user",
    message=message,
    created_at=datetime.utcnow()
)
db.add(user_message)

# Step 2: Save assistant message with metadata
assistant_message = ChatMessageModel(
    session_id=session_id,
    role="assistant",
    message=response_text,
    meta_data=metadata,
    created_at=datetime.utcnow()
)
db.add(assistant_message)
db.commit()

# Step 3: Vectorize chat history for future context
collection_name = f"db_session_{session_id}"

# Get all messages for this session
all_messages = db.query(ChatMessageModel).filter(
    ChatMessageModel.session_id == session_id
).order_by(ChatMessageModel.created_at).all()

# Prepare recent messages for vectorization (last 10)
recent_messages = all_messages[-10:]
chat_texts = []
chat_metadatas = []
chat_ids = []

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

# Step 4: Generate embeddings and store in ChromaDB
if chat_texts:
    embeddings = embedding_service.generate_embeddings(
        chat_texts, 
        model_name=get_default_embedding_model()
    )
    
    # Add to ChromaDB collection
    chromadb_service.add_documents(
        collection_name=collection_name,
        texts=chat_texts,
        embeddings=embeddings,
        metadatas=chat_metadatas,
        ids=chat_ids
    )
```

**What Happens:**
- Saves user and assistant messages to database
- Vectorizes last 10 messages for conversational context
- Stores embeddings in ChromaDB for semantic search
- Future queries can use this context for better understanding

---

### Step 7: Prepare and Return Response

**Code Location:** `backend/main.py` lines 1606-1635

**Process Flow:**

```python
# Prepare response - only include SQL if execution was successful
response_data = {
    "response": response_text,  # Natural language response
    "message_id": assistant_message.id,
    "session_id": session_id,
    "requires_sql": agent_result.get('requires_sql', False),
    "sql_executed": sql_executed,
    "metadata": {
        "intent": metadata.get('intent', {}),
        "used_semantic_search": metadata.get('used_semantic_search', False),
        "relevant_schema_parts_count": metadata.get('relevant_schema_parts_count', 0)
    }
}

# Only include SQL in response if execution was successful
if sql_executed and query_result and query_result.get('success'):
    response_data["sql"] = agent_result.get('sql')
    if query_result.get('data'):
        response_data["query_result"] = query_result.get('data', [])[:10]  # Limit to 10 rows
    else:
        response_data["query_result"] = []
elif agent_result.get('sql') and not sql_executed:
    # SQL was generated but not executed (validation failed)
    # Don't expose the SQL to frontend
    response_data["sql"] = None
else:
    # No SQL generated or errors occurred
    response_data["sql"] = None

return response_data
```

**Response Structure:**
```json
{
    "response": "There are 1000 movies in the database.",
    "message_id": 123,
    "session_id": 45,
    "requires_sql": true,
    "sql_executed": true,
    "sql": "SELECT COUNT(*) FROM film;",
    "query_result": [
        {"count": 1000}
    ],
    "metadata": {
        "intent": {
            "requires_sql": true,
            "confidence": 0.95,
            "intent_type": "sql_query"
        },
        "used_semantic_search": true,
        "relevant_schema_parts_count": 3
    }
}
```

---

## Key Processing Components

### Semantic Search Mechanism

**How It Works:**

1. **Schema Vectorization** (during schema processing):
   ```python
   # Each table/column becomes a chunk with embedding
   chunks = _create_enriched_schema_chunks(schema_data, schema_text)
   # Creates chunks like:
   # - Table chunks: "Database Table: film\nColumns: film_id, title, ..."
   # - Column chunks: "Column: film.title\nType: varchar\n..."
   # - Relationship chunks: "Source Table: film_actor\nReferences: film"
   
   embeddings = embedding_service.generate_embeddings(
       all_texts, 
       model_name=embedding_model_name
   )
   
   chromadb_service.add_documents(
       collection_name=collection_name,
       texts=all_texts,
       embeddings=embeddings,
       metadatas=all_metadatas,
       ids=all_ids
   )
   ```

2. **Query Vectorization** (during user query):
   ```python
   # Convert user query to embedding vector
   query_embeddings = embedding_service.generate_embeddings(
       [user_query],
       model_name=embedding_model_name
   )
   # Example: "movies" → [0.123, -0.456, 0.789, ...] (384 or 768 dim vector)
   ```

3. **Semantic Search**:
   ```python
   # Find most similar schema chunks using cosine similarity
   results = chromadb_service.query_collection(
       collection_name=collection_name,
       query_embeddings=query_embeddings,
       n_results=15  # Top 15 most similar
   )
   # Returns: documents (schema chunks), distances (similarity scores), metadatas
   ```

4. **Distance-Based Filtering**:
   ```python
   # Lower distance = higher similarity
   doc_dist_pairs = list(zip(documents, metadatas, distances))
   doc_dist_pairs.sort(key=lambda x: x[2])  # Sort by distance
   
   # Take top 70% most relevant
   threshold_index = max(5, int(len(doc_dist_pairs) * 0.7))
   top_pairs = doc_dist_pairs[:threshold_index]
   ```

**Example:**
- Query: "movies" → embedding vector `[0.1, -0.3, 0.5, ...]`
- Schema chunk: "Database Table: film" → embedding vector `[0.12, -0.28, 0.52, ...]`
- Cosine distance: `0.15` (very similar!)
- Result: `film` table chunk is retrieved

---

### Schema Chunk Structure

**Table Chunk Example:**
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

**Column Chunk Example:**
```
Column Definition:
Table: film
Column Name: title
Data Type: varchar
Constraint: NOT NULL
```

**Relationship Chunk Example:**
```
Table Relationship:
Source Table: film_actor
Source Columns: film_id
References Table: film
Referenced Columns: film_id
Relationship Type: Foreign Key Constraint
```

---

## Error Handling and User-Friendly Messages

### SQL Execution Errors

**Code Location:** `backend/services/sql_execution_service.py` lines 200-233

```python
def _format_error_message(self, error_msg: str) -> str:
    """Convert technical errors to user-friendly messages"""
    error_lower = error_msg.lower()
    
    if 'table' in error_lower or 'does not exist' in error_lower:
        return "The table you're trying to query doesn't exist in the database."
    elif 'column' in error_lower:
        return "One or more columns don't exist in the database."
    elif 'syntax error' in error_lower:
        return "There's a syntax error in the generated SQL query."
    elif 'permission denied' in error_lower:
        return "You don't have permission to execute this query."
    else:
        return "The database query encountered an error. Please try rephrasing your question."
```

**Example:**
- Technical error: `relation "films" does not exist`
- User sees: "The table you're trying to query doesn't exist in the database."

---

## Complete Example Flow

### User Query: "Show me all movies released after 2000"

**Step-by-Step Processing:**

1. **Intent Detection:**
   - Query: "Show me all movies released after 2000"
   - Semantic search finds: `film` table chunks
   - LLM determines: `requires_sql: true`, `confidence: 0.95`

2. **NL-to-SQL:**
   - Semantic search retrieves:
     - `film` table chunk (movies→film match)
     - `film.release_year` column chunk
     - Related chunks about film table structure
   - LLM generates:
     ```sql
     SELECT * FROM film WHERE release_year > 2000 LIMIT 100;
     ```
   - Confidence: 0.92

3. **SQL Validation:**
   - Checks: Table `film` exists ✓
   - Checks: Column `release_year` exists ✓
   - Checks: Syntax valid ✓
   - Result: `valid: true`, `corrected_sql: "SELECT * FROM film WHERE release_year > 2000 LIMIT 100;"`

4. **SQL Execution:**
   - Executes SQL on PostgreSQL database
   - Returns: 850 rows
   - Execution time: 0.023 seconds

5. **Response Generation:**
   - Formats results using LLM:
     ```
     "I found 850 movies released after 2000. Here are the results..."
     ```
   - Includes SQL and first 10 rows in response

6. **Chat History:**
   - Saves messages
   - Vectorizes conversation for future context

7. **Response to Frontend:**
   ```json
   {
       "response": "I found 850 movies released after 2000...",
       "sql": "SELECT * FROM film WHERE release_year > 2000 LIMIT 100;",
       "sql_executed": true,
       "query_result": [
           {"film_id": 1, "title": "ACADEMY DINOSAUR", "release_year": 2006},
           ...
       ]
   }
   ```

---

## Performance Considerations

### Embedding Model Selection

**Recommended Models for NL-to-SQL:**
- **BGE Large EN v1.5** (1024 dim): Highest accuracy, slower
- **BGE Base EN** (768 dim): Good balance
- **MPNet Base v2** (768 dim): Fast, good quality

### Semantic Search Optimization

- **Chunk Granularity**: Table-level + Column-level + Relationship chunks
- **Result Limiting**: Top 15 chunks for NL-to-SQL, filtered to top 70%
- **Distance Thresholding**: Adaptive based on result distribution

### Caching

- Schema embeddings are cached in ChromaDB
- LLM responses use temperature=0.1 for consistency
- Chat history vectorized incrementally (last 10 messages)

---

## Conclusion

The system processes queries through multiple intelligent agents:
1. **Semantic understanding** via embedding models (no hardcoded mappings)
2. **Intent detection** via LLM analysis
3. **SQL generation** via LLM with schema context
4. **SQL validation** via LLM checking
5. **Execution** via SQLAlchemy
6. **Response formatting** via LLM
7. **Context preservation** via vectorized chat history

All steps use actual schema structure and semantic similarity rather than hardcoded rules, making it adaptable to any database schema.

