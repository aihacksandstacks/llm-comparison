-- 001_initial_schema.sql
-- Initial schema for LLM Comparison Tool

-- Create pgvector extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table for vector storage
-- Using 768 as default dimension (common for many models)
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    embedding_model TEXT,
    embedding_dimensions INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);

-- Create a trigger to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER embeddings_update_timestamp
BEFORE UPDATE ON embeddings
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- Create collections table to organize embeddings
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create a trigger for collections
CREATE TRIGGER collections_update_timestamp
BEFORE UPDATE ON collections
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- Create a table to track collection membership
CREATE TABLE IF NOT EXISTS collection_embeddings (
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    embedding_id TEXT NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
    PRIMARY KEY (collection_id, embedding_id)
);

-- Create indices for faster lookups
CREATE INDEX IF NOT EXISTS idx_collection_embeddings_collection_id ON collection_embeddings(collection_id);
CREATE INDEX IF NOT EXISTS idx_collection_embeddings_embedding_id ON collection_embeddings(embedding_id); 