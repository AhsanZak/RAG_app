# Embedding Models Usage in Database Chat

## Overview
When a user sends a chat message to the database chat system, **ONE embedding model** is used, but it's called **3 times** for different purposes during the query processing pipeline.

## The Single Embedding Model

The system uses **ONE embedding model** throughout the entire chat process. The model name is determined at the start of the chat endpoint and passed to all agents.

**Default Model Selection:**
- Primary choice: `BAAI/bge-base-en` (if available) - recommended for NL-to-SQL accuracy
- Fallback: System default embedding model (typically `all-MiniLM-L6-v2`)

**Location:** `backend/main.py` line 1584
```python
embedding_model_name = get_default_embedding_model()
```

---

## Three Uses of the Embedding Model During Chat

### 1. **Intent Detection Agent** (Semantic Search)
**Purpose:** Determine if the user query requires SQL execution or is just a general question

**What it does:**
- Converts the user query to embeddings
- Performs semantic search in ChromaDB to find relevant schema chunks
- Uses top 8 most relevant schema chunks to help LLM determine intent

**Location:** `backend/services/db_agents.py` lines 46-68
```python
# Convert user query to embedding
query_embeddings = self.embedding_service.generate_embeddings(
    [user_query],
    model_name=embedding_model_name
)

# Retrieve relevant schema chunks
results = self.chromadb_service.query_collection(
    collection_name=collection_name,
    query_embeddings=query_embeddings,
    n_results=8  # Get more results for intent detection
)
```

**Why it's needed:**
- Understands semantic similarity between user query and schema structure
- "movies" → matches "film" table (semantic understanding, no hardcoding)
- "show me data" → matches tables with "list" or "display" patterns
- Helps LLM make better intent decisions with relevant context

---

### 2. **NL-to-SQL Agent** (Semantic Search)
**Purpose:** Convert natural language query to SQL by finding relevant schema parts

**What it does:**
- Converts the user query to embeddings (same query, same model)
- Performs semantic search to find top 15 relevant schema chunks
- Filters to use top 70% most relevant chunks (minimum top 5)
- Provides only relevant schema parts to LLM for SQL generation

**Location:** `backend/services/db_agents.py` lines 174-227
```python
# Convert user query to embedding
query_embeddings = self.embedding_service.generate_embeddings(
    [user_query],
    model_name=embedding_model_name
)

# Retrieve relevant schema chunks using semantic search
results = self.chromadb_service.query_collection(
    collection_name=collection_name,
    query_embeddings=query_embeddings,
    n_results=15  # Get more results for better coverage
)
```

**Why it's needed:**
- Finds semantically relevant tables/columns for the query
- "movies" → finds "film" table and related columns
- "customer orders" → finds "customer" and "order" tables with relationships
- Reduces token usage by only sending relevant schema parts to LLM
- Improves SQL accuracy by focusing on relevant schema elements

---

### 3. **Chat History Vectorization** (After Response)
**Purpose:** Store chat messages as embeddings for conversational context

**What it does:**
- Takes the last 10 messages (user + assistant pairs)
- Converts each message to embeddings
- Stores them in ChromaDB for future semantic search

**Location:** `backend/main.py` lines 1749-1796
```python
# Get all messages for this session
recent_messages = all_messages[-10:]  # Last 10 messages

# Prepare texts for vectorization
for msg in recent_messages:
    text = f"{msg.role}: {msg.message}"
    chat_texts.append(text)

# Generate embeddings
embeddings = embedding_service.generate_embeddings(
    chat_texts, 
    model_name=get_default_embedding_model()
)

# Add to ChromaDB
chromadb_service.add_documents(
    collection_name=collection_name,
    texts=chat_texts,
    embeddings=embeddings,
    metadatas=chat_metadatas,
    ids=chat_ids
)
```

**Why it's needed:**
- Maintains conversation context across multiple queries
- "What about the previous result?" → can find previous messages semantically
- "Show me more details" → understands context from previous messages
- Enables the system to reference past conversation for better responses

---

## Summary Table

| Step | Agent/Process | Purpose | Embeddings Generated | Model Used |
|------|---------------|---------|---------------------|------------|
| 1 | Intent Detection | Find relevant schema chunks for intent analysis | 1 (user query) | Same model |
| 2 | NL-to-SQL | Find relevant schema chunks for SQL generation | 1 (user query) | Same model |
| 3 | Chat History | Vectorize conversation messages | Up to 10 (last 10 messages) | Same model |

**Total Embedding Calls:** 3 separate calls (1 query + 1 query + up to 10 messages)
**Model Instances:** 1 (same model name used throughout)
**Total Embeddings Generated:** Up to 12 embeddings per chat message

---

## Why Only One Model?

1. **Consistency:** Same semantic space for all operations ensures consistent understanding
2. **Efficiency:** Model is loaded once and reused
3. **Accuracy:** Using the same model for schema search and chat history ensures semantic compatibility
4. **Simplicity:** No need to manage multiple models or coordinate between them

---

## Model Selection Priority

When processing schema (one-time setup):
1. User-selected model (if provided)
2. `BAAI/bge-base-en` (if available - recommended for NL-to-SQL)
3. System default model

When chatting (per query):
1. Uses default embedding model (same as schema processing for consistency)

**Note:** The model used for schema processing should match the model used for chat queries to ensure semantic compatibility.

---

## Example Flow

**User Query:** "Show me all movies from 2020"

1. **Intent Detection:**
   - Query → Embedding: `[0.123, -0.456, ...]`
   - Search ChromaDB → Finds: "film" table chunks
   - LLM determines: `requires_sql: true`

2. **NL-to-SQL:**
   - Query → Embedding: `[0.123, -0.456, ...]` (same embedding)
   - Search ChromaDB → Finds: "film" table, "release_year" column
   - LLM generates: `SELECT * FROM film WHERE release_year = 2020;`

3. **Chat History:**
   - Messages: ["user: Show me all movies from 2020", "assistant: Found 15 movies..."]
   - Messages → Embeddings: `[[0.123, ...], [0.789, ...]]`
   - Stored in ChromaDB for future context

---

## Key Points

✅ **ONE embedding model** is used throughout the entire chat process
✅ The model is called **3 times** for different purposes (intent, SQL generation, history)
✅ All operations use the **same semantic space** for consistency
✅ The model is selected once at the start and reused
✅ No hardcoded mappings - pure semantic understanding through embeddings

