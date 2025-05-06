# Vector Database Setup

This directory contains database-related files for the LLM Comparison Tool.

## PostgreSQL with pgvector

The LLM Comparison Tool uses PostgreSQL with the pgvector extension for vector storage and similarity search.

### Setup Options

#### Docker Compose (Recommended)

The easiest way to set up the database is using Docker Compose. The configuration is already provided in the `docker-compose.yml` file at the root of the project.

To start the database:

```bash
docker-compose up -d vectordb
```

This will start a PostgreSQL instance with pgvector extension installed and configured according to the settings in `.env` (or default values if not set).

#### Manual Setup

If you prefer to use an existing PostgreSQL instance:

1. Install the pgvector extension:
   ```sql
   CREATE EXTENSION vector;
   ```

2. Update your `.env` file with the connection details:
   ```
   DB_PROVIDER=postgres
   POSTGRES_HOST=your_host
   POSTGRES_PORT=5432
   POSTGRES_USER=your_user
   POSTGRES_PASSWORD=your_password
   POSTGRES_DB=your_database
   ```

### Testing the Database Connection

The project includes a CLI tool for testing the database connection:

```bash
# Ensure you're in the project root directory
python -m src.shared.db_tools init

# Test with random vectors
python -m src.shared.db_tools test --num-vectors 5 --dimension 768

# Benchmark the database performance
python -m src.shared.db_tools benchmark --num-vectors 100 --dimension 768
```

### Database Schema

The database uses a simple schema with a single table:

```sql
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    text TEXT,
    embedding FLOAT[] NOT NULL,  -- Uses pgvector's vector type
    metadata JSONB
);
```

Vector similarity searches use cosine distance:

```sql
SELECT id, text, 1 - (embedding <=> query_embedding) AS similarity
FROM embeddings
ORDER BY embedding <=> query_embedding
LIMIT num_results;
```

### Integration with RAG

The vector database is integrated with the RAG system in the LLM Comparison Tool. When documents are indexed, they are also stored in the vector database for efficient retrieval.

The system will automatically fall back to the vector database if the in-memory index is not available, providing consistent access to document embeddings. 