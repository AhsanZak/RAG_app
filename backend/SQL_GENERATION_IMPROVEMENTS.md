# SQL Generation Improvements - Using Online APIs with Vectorized Schema

## Overview

The database chat system has been enhanced to:
1. **Pass both vectorized schema chunks AND structured metadata** to the LLM
2. **Support online APIs** (OpenAI, Anthropic) for better SQL generation
3. **Improve SQL accuracy** by combining semantic search results with complete schema context

---

## Current Implementation: What Gets Passed to LLM?

### ✅ YES - Vectorized Schema IS Passed to LLM

**The system currently:**
1. **Semantic Search**: User query → embeddings → searches ChromaDB for relevant schema chunks
2. **Retrieves Top Chunks**: Gets top 10-15 most relevant schema chunks (vectorized)
3. **Passes to LLM**: Both vectorized chunks AND structured metadata are included in the prompt

### What the LLM Receives:

```
=== STRUCTURED SCHEMA METADATA ===
- Total Tables: 15
- Total Columns: 150
- All Tables: actor, address, category, city, country, customer, film, ...
- Key Relationships: film.actor_id -> actor.actor_id, ...

=== SEMANTICALLY RELEVANT SCHEMA CHUNKS (from vectorized search) ===
Table: film
  Columns: film_id (INTEGER), title (VARCHAR), release_year (INTEGER), ...
  Primary Keys: film_id
  Foreign Keys: language_id -> language.language_id

Table: actor
  Columns: actor_id (INTEGER), first_name (VARCHAR), last_name (VARCHAR), ...
  ...

=== USER QUERY ===
"Show me all movies from 2020"
```

**Location**: `backend/services/db_agents.py` lines 244-293

---

## Improvements Made

### 1. **Enhanced Schema Context**

**Before**: Only semantic search results (top chunks)
**After**: Both structured metadata + semantic chunks

**New Method**: `_build_structured_schema_context()`
- Provides complete table list
- Shows all relationships
- Includes summary statistics

**Benefits**:
- LLM sees ALL tables (not just relevant ones)
- Better understanding of relationships
- More accurate JOIN generation

### 2. **Improved Prompt Structure**

The prompt now clearly separates:
- **Structured Metadata**: All tables, relationships, summary
- **Semantic Chunks**: Detailed column info for relevant tables
- **Instructions**: Clear guidance on using both sources

### 3. **Online API Support**

Added support for:
- ✅ **OpenAI** (GPT-3.5, GPT-4, GPT-4 Turbo)
- ✅ **Anthropic** (Claude 3 Sonnet, Opus, Haiku)
- ✅ **Ollama** (existing support)

---

## How to Use Online APIs for SQL Generation

### Step 1: Add an LLM Model in Frontend

1. Go to **LLM Model Management** in the frontend
2. Click **"Add New Model"**
3. Select Provider:
   - **OpenAI**: For GPT models
   - **Anthropic**: For Claude models
4. Fill in details:
   - **Model Name**: `gpt-4-turbo-preview` (OpenAI) or `claude-3-opus-20240229` (Anthropic)
   - **Base URL**: 
     - OpenAI: `https://api.openai.com/v1`
     - Anthropic: `https://api.anthropic.com/v1`
   - **API Key**: Your API key (starts with `sk-` for OpenAI, `sk-ant-` for Anthropic)
5. Click **"Test Connection"** to verify
6. Click **"Save"**

### Step 2: Select Model in Database Chat

1. Open **Database Chat**
2. In the header, select your OpenAI/Anthropic model from the **LLM Model** dropdown
3. The system will use this model for SQL generation

### Step 3: Verify SQL Generation

When you ask a question:
- The system uses semantic search to find relevant schema chunks
- Passes BOTH structured metadata + semantic chunks to your selected LLM
- The LLM generates SQL using the complete context

---

## Code Flow

### 1. Query Processing (`backend/services/db_agents.py`)

```python
# Step 1: Semantic Search
query_embeddings = embedding_service.generate_embeddings([user_query])
results = chromadb_service.query_collection(
    collection_name=collection_name,
    query_embeddings=query_embeddings,
    n_results=15
)
relevant_schema_chunks = results['documents'][0]  # Top 15 chunks

# Step 2: Build Structured Context
structured_metadata = _build_structured_schema_context(schema_details)
# Includes: all tables, relationships, summary

# Step 3: Create Enhanced Prompt
prompt = f"""
=== STRUCTURED SCHEMA METADATA ===
{structured_metadata}

=== SEMANTICALLY RELEVANT SCHEMA CHUNKS ===
{relevant_schema_chunks}

=== USER QUERY ===
{user_query}
"""

# Step 4: Send to LLM (OpenAI/Anthropic/Ollama)
response = llm_service.generate_response(
    messages=[{"role": "user", "content": prompt}],
    model_config=model_config  # Includes provider, api_key, base_url
)
```

### 2. LLM Service (`backend/services/llm_service.py`)

```python
def generate_response(messages, model_config, temperature):
    provider = model_config.get('provider')  # 'openai', 'anthropic', 'ollama'
    
    if provider == 'openai':
        # OpenAI API call
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model_name, "messages": messages}
        )
    
    elif provider == 'anthropic':
        # Anthropic API call
        response = requests.post(
            f"{base_url}/messages",
            headers={"x-api-key": api_key},
            json={"model": model_name, "messages": messages}
        )
```

---

## Best Practices for SQL Generation

### 1. **Model Selection**

**For Best SQL Accuracy**:
- **OpenAI GPT-4 Turbo**: Best overall performance, understands complex relationships
- **Anthropic Claude 3 Opus**: Excellent for complex multi-table queries
- **OpenAI GPT-3.5 Turbo**: Good balance of speed and accuracy
- **Ollama Models**: Useful for local/private deployments, but may have lower accuracy

### 2. **Schema Context**

The system automatically provides:
- ✅ **Complete table list** (structured metadata)
- ✅ **All relationships** (structured metadata)
- ✅ **Relevant column details** (semantic chunks)

**No manual intervention needed** - the system combines both sources automatically.

### 3. **Prompt Engineering**

The enhanced prompt includes:
- Clear separation of structured vs semantic information
- Instructions to use BOTH sources
- Examples of best practices
- Database-specific syntax guidance

---

## Troubleshooting

### Issue: SQL Generated Doesn't Use All Relevant Tables

**Solution**: The structured metadata now includes ALL tables, so the LLM can see relationships even if they weren't in semantic chunks.

### Issue: OpenAI/Anthropic API Errors

**Check**:
1. API key is valid and has sufficient credits
2. Base URL is correct
3. Model name is valid for the provider
4. Network connectivity

**Error Messages**:
- `"OpenAI API key is required"` → Add API key in model config
- `"Cannot connect to API"` → Check base_url and network
- `"Model not found"` → Verify model name is correct

### Issue: Still Getting Inaccurate SQL

**Try**:
1. Use GPT-4 Turbo or Claude 3 Opus (better models)
2. Ensure schema extraction captured all tables correctly
3. Check that semantic search is finding relevant chunks (check logs)

---

## Example: Complete Flow

**User Query**: "Show me all movies from 2020 with their actors"

1. **Intent Detection**: `requires_sql: true`, `intent_type: sql_query`
2. **Semantic Search**: 
   - Query → embeddings
   - Finds: `film` table, `actor` table, `film_actor` junction table
3. **Schema Context Building**:
   - Structured: All 15 tables, relationships (film ↔ film_actor ↔ actor)
   - Semantic: Detailed columns for film, actor, film_actor
4. **Prompt to LLM**:
   ```
   === STRUCTURED SCHEMA METADATA ===
   All Tables: actor, address, category, city, country, customer, film, film_actor, ...
   Relationships: film.film_id -> film_actor.film_id, film_actor.actor_id -> actor.actor_id
   
   === SEMANTICALLY RELEVANT CHUNKS ===
   Table: film
     Columns: film_id, title, release_year, ...
   Table: actor
     Columns: actor_id, first_name, last_name, ...
   Table: film_actor
     Columns: film_id, actor_id, ...
   
   === USER QUERY ===
   "Show me all movies from 2020 with their actors"
   ```
5. **LLM (GPT-4) Generates**:
   ```sql
   SELECT f.title, f.release_year, 
          a.first_name, a.last_name
   FROM film f
   JOIN film_actor fa ON f.film_id = fa.film_id
   JOIN actor a ON fa.actor_id = a.actor_id
   WHERE f.release_year = 2020
   ORDER BY f.title, a.last_name;
   ```
6. **SQL Validation**: Validates against schema
7. **Execution**: Runs query and returns results

---

## Summary

✅ **Vectorized schema IS passed to LLM** - semantic search results are included
✅ **Structured metadata ALSO passed** - complete table list and relationships
✅ **Online APIs supported** - OpenAI and Anthropic work out of the box
✅ **Enhanced prompts** - better instructions for using both schema sources
✅ **No hardcoding** - all schema information comes from extraction

The system now provides the LLM with the **best of both worlds**:
- **Semantic understanding** (vectorized chunks for relevant details)
- **Complete context** (structured metadata for all tables and relationships)

This ensures accurate SQL generation, especially when using powerful models like GPT-4 or Claude 3 Opus.

