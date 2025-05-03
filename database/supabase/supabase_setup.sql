-- This SQL script sets up the necessary tables and functions for Supabase pgvector integration

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for storing embeddings
CREATE TABLE IF NOT EXISTS embeddings (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  embedding VECTOR(1536),  -- Adjust dimension based on your embedding model
  metadata JSONB
);

-- Create an index for faster similarity searches
CREATE INDEX IF NOT EXISTS embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops);

-- Create a stored procedure for similarity search
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(1536),
  match_count INT DEFAULT 5,
  table_name TEXT DEFAULT 'embeddings'
)
RETURNS TABLE (
  id TEXT,
  text TEXT,
  similarity FLOAT,
  metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY EXECUTE format(
    'SELECT id, text, 1 - (embedding <=> $1) AS similarity, metadata
     FROM %I
     ORDER BY embedding <=> $1
     LIMIT $2',
     table_name
  ) USING query_embedding, match_count;
END;
$$; 