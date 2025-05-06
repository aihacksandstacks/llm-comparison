"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Database providers for the LLM Comparison Tool.
Supports PostgreSQL with pgvector and Supabase.
"""

import os
from typing import Dict, Any, Union, Optional, List
import httpx
from sqlalchemy import create_engine, Column, String, JSON, Float, MetaData, Table
from sqlalchemy.sql import text as sql_text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.pool import QueuePool
import numpy as np
import json
import time

from src.shared.config import DB_CONFIG
from src.shared.logger import get_logger, log_db_operation

# Set up module logger
logger = get_logger(__name__)

class DBProvider:
    """Base class for database providers."""
    
    def __init__(self):
        """Initialize the database provider."""
        self.connection = None
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing {self.__class__.__name__}")
    
    def connect(self) -> bool:
        """
        Connect to the database.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement connect()")
    
    def disconnect(self) -> None:
        """Disconnect from the database."""
        raise NotImplementedError("Subclasses must implement disconnect()")
    
    def store_embedding(self, doc_id: str, text: str, embedding: list, metadata: Dict[str, Any]) -> bool:
        """
        Store an embedding in the database.
        
        Args:
            doc_id: Document ID.
            text: The text that was embedded.
            embedding: The embedding vector.
            metadata: Additional metadata for the document.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement store_embedding()")
    
    def search_similar(self, embedding: list, top_k: int = 5) -> list:
        """
        Search for similar documents by embedding.
        
        Args:
            embedding: The query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            list: List of similar documents with their similarity scores.
        """
        raise NotImplementedError("Subclasses must implement search_similar()")
    
    def health_check(self) -> bool:
        """
        Check if the database connection is healthy.
        
        Returns:
            bool: True if healthy, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement health_check()")
    
    def delete_embedding(self, doc_id: str) -> bool:
        """
        Delete an embedding from the database.
        
        Args:
            doc_id: Document ID to delete.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement delete_embedding()")


class PostgresVectorProvider(DBProvider):
    """PostgreSQL with pgvector database provider."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PostgreSQL provider.
        
        Args:
            config: Database configuration. If None, uses the default configuration.
        """
        super().__init__()
        self.config = config or DB_CONFIG
        self.engine = None
        self.metadata = MetaData()
        self.embeddings_table = None
        self.connection_retries = 3
        self.connection_retry_delay = 2  # seconds
        self.logger.debug(f"PostgreSQL config: host={self.config['host']}, port={self.config['port']}, db={self.config['database']}")
    
    def connect(self) -> bool:
        """
        Connect to the PostgreSQL database with retry logic.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        for attempt in range(1, self.connection_retries + 1):
            try:
                # Create connection URL
                connection_url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
                
                self.logger.info(f"Connecting to PostgreSQL at {self.config['host']}:{self.config['port']}/{self.config['database']} (attempt {attempt}/{self.connection_retries})")
                
                # Create engine with connection pooling
                self.engine = create_engine(
                    connection_url,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_pre_ping=True  # Test connections before using them
                )
                
                # Note: We don't need to define the table structure here anymore
                # since it's created by migrations now
                
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(sql_text("SELECT 1"))
                
                self.logger.info("Successfully connected to PostgreSQL database")
                return True
                
            except Exception as e:
                self.logger.error(f"Error connecting to PostgreSQL (attempt {attempt}/{self.connection_retries}): {e}")
                
                if attempt < self.connection_retries:
                    self.logger.info(f"Retrying in {self.connection_retry_delay} seconds...")
                    time.sleep(self.connection_retry_delay)
                else:
                    self.logger.error("Max retries reached, giving up")
                    return False
    
    def disconnect(self) -> None:
        """Disconnect from the PostgreSQL database."""
        if self.engine:
            self.logger.info("Disconnecting from PostgreSQL database")
            self.engine.dispose()
    
    def store_embedding(self, doc_id: str, text: str, embedding: list, metadata: Dict[str, Any], 
                        embedding_model: str = "default") -> bool:
        """
        Store an embedding in the PostgreSQL database.
        
        Args:
            doc_id: Document ID.
            text: The text that was embedded.
            embedding: The embedding vector.
            metadata: Additional metadata for the document.
            embedding_model: The name of the embedding model used (optional).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.engine:
            self.logger.error("Database connection not established")
            return False
            
        try:
            self.logger.debug(f"Storing embedding for document '{doc_id}'")
            
            # Get embedding dimensions
            embedding_dimensions = len(embedding)
            
            with self.engine.connect() as conn:
                # Check if document already exists
                query = sql_text("SELECT id FROM embeddings WHERE id = :id")
                result = conn.execute(query, {"id": doc_id}).fetchone()
                
                if result:
                    self.logger.debug(f"Document '{doc_id}' already exists, updating")
                    # Update existing document
                    update_query = sql_text(
                        """
                        UPDATE embeddings 
                        SET text = :text, 
                            embedding = array_to_vector(:embedding), 
                            metadata = :metadata,
                            embedding_model = :model,
                            embedding_dimensions = :dimensions
                        WHERE id = :id
                        """
                    )
                    conn.execute(update_query, {
                        "id": doc_id,
                        "text": text,
                        "embedding": embedding,
                        "metadata": json.dumps(metadata) if metadata else None,
                        "model": embedding_model,
                        "dimensions": embedding_dimensions
                    })
                else:
                    self.logger.debug(f"Document '{doc_id}' is new, inserting")
                    # Insert new document
                    
                    # First create a function to convert array to vector if it doesn't exist
                    conn.execute(sql_text(
                        """
                        CREATE OR REPLACE FUNCTION array_to_vector(arr float[]) 
                        RETURNS vector 
                        AS $$ 
                        BEGIN
                          RETURN arr::vector;
                        END; 
                        $$ LANGUAGE plpgsql IMMUTABLE;
                        """
                    ))
                    
                    # Now use the function in our insert
                    insert_query = sql_text(
                        """
                        INSERT INTO embeddings 
                        (id, text, embedding, metadata, embedding_model, embedding_dimensions) 
                        VALUES 
                        (:id, :text, array_to_vector(:embedding), :metadata, :model, :dimensions)
                        """
                    )
                    conn.execute(insert_query, {
                        "id": doc_id,
                        "text": text,
                        "embedding": embedding,
                        "metadata": json.dumps(metadata) if metadata else None,
                        "model": embedding_model,
                        "dimensions": embedding_dimensions
                    })
                
                conn.commit()
                
            log_db_operation(self.logger, "store", {
                "doc_id": doc_id,
                "text_length": len(text),
                "embedding_dimensions": embedding_dimensions,
                "embedding_model": embedding_model,
                "metadata_keys": list(metadata.keys()) if metadata else []
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing embedding in PostgreSQL: {e}", exc_info=True)
            return False
    
    def search_similar(self, embedding: list, top_k: int = 5) -> list:
        """
        Search for similar documents in the PostgreSQL database.
        
        Args:
            embedding: The query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            list: List of similar documents with their similarity scores.
        """
        if not self.engine:
            self.logger.error("Database connection not established")
            return []
            
        try:
            self.logger.debug(f"Searching for similar documents, top_k={top_k}")
            with self.engine.connect() as conn:
                # Make sure array_to_vector function exists
                conn.execute(sql_text(
                    """
                    CREATE OR REPLACE FUNCTION array_to_vector(arr float[]) 
                    RETURNS vector 
                    AS $$ 
                    BEGIN
                      RETURN arr::vector;
                    END; 
                    $$ LANGUAGE plpgsql IMMUTABLE;
                    """
                ))
                
                # Use cosine similarity with pgvector
                query = sql_text("""
                SELECT 
                    id, 
                    text, 
                    1 - (embedding <=> array_to_vector(:embedding)) AS similarity,
                    metadata,
                    embedding_model,
                    embedding_dimensions
                FROM 
                    embeddings
                ORDER BY 
                    embedding <=> array_to_vector(:embedding)
                LIMIT 
                    :top_k
                """)
                
                results = conn.execute(query, {
                    "embedding": embedding,
                    "top_k": top_k
                }).fetchall()
                
                # Format results
                formatted_results = []
                for row in results:
                    formatted_results.append({
                        "id": row[0],
                        "text": row[1],
                        "similarity": float(row[2]),
                        "metadata": row[3],
                        "embedding_model": row[4],
                        "embedding_dimensions": row[5]
                    })
                
                self.logger.debug(f"Found {len(formatted_results)} similar documents")
                log_db_operation(self.logger, "search", {
                    "embedding_dimensions": len(embedding),
                    "top_k": top_k,
                    "results_count": len(formatted_results)
                })
                
                return formatted_results
                
        except Exception as e:
            self.logger.error(f"Error searching for similar documents in PostgreSQL: {e}", exc_info=True)
            return []
    
    def health_check(self) -> bool:
        """
        Check if the PostgreSQL connection is healthy.
        
        Returns:
            bool: True if healthy, False otherwise.
        """
        if not self.engine:
            self.logger.error("Database connection not established")
            return False
            
        try:
            with self.engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def delete_embedding(self, doc_id: str) -> bool:
        """
        Delete an embedding from the PostgreSQL database.
        
        Args:
            doc_id: Document ID to delete.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.engine:
            self.logger.error("Database connection not established")
            return False
            
        try:
            self.logger.debug(f"Deleting embedding for document '{doc_id}'")
            
            with self.engine.connect() as conn:
                # Delete the document
                delete_query = sql_text("DELETE FROM embeddings WHERE id = :id")
                result = conn.execute(delete_query, {"id": doc_id})
                conn.commit()
                
                if result.rowcount > 0:
                    self.logger.debug(f"Successfully deleted document '{doc_id}'")
                    return True
                else:
                    self.logger.debug(f"Document '{doc_id}' not found, nothing to delete")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error deleting document '{doc_id}': {e}", exc_info=True)
            return False
    
    def batch_store_embeddings(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Store multiple embeddings in a batch operation.
        
        Args:
            documents: List of documents with id, text, embedding, and metadata.
                       Optionally can include embedding_model.
            
        Returns:
            bool: True if all documents were stored successfully, False otherwise.
        """
        if not self.engine:
            self.logger.error("Database connection not established")
            return False
            
        try:
            self.logger.debug(f"Batch storing {len(documents)} embeddings")
            
            with self.engine.connect() as conn:
                # Create the array_to_vector function if it doesn't exist
                conn.execute(sql_text(
                    """
                    CREATE OR REPLACE FUNCTION array_to_vector(arr float[]) 
                    RETURNS vector 
                    AS $$ 
                    BEGIN
                      RETURN arr::vector;
                    END; 
                    $$ LANGUAGE plpgsql IMMUTABLE;
                    """
                ))
                
                for doc in documents:
                    # Get embedding model name and dimensions
                    embedding_model = doc.get('embedding_model', 'default')
                    embedding_dimensions = len(doc['embedding'])
                    
                    # Check if document already exists
                    select_query = sql_text("SELECT id FROM embeddings WHERE id = :id")
                    result = conn.execute(select_query, {"id": doc['id']}).fetchone()
                    
                    if result:
                        # Update existing document
                        update_query = sql_text(
                            """
                            UPDATE embeddings 
                            SET text = :text, 
                                embedding = array_to_vector(:embedding), 
                                metadata = :metadata,
                                embedding_model = :model,
                                embedding_dimensions = :dimensions
                            WHERE id = :id
                            """
                        )
                        conn.execute(update_query, {
                            "id": doc['id'],
                            "text": doc['text'],
                            "embedding": doc['embedding'],
                            "metadata": json.dumps(doc['metadata']) if 'metadata' in doc and doc['metadata'] else None,
                            "model": embedding_model,
                            "dimensions": embedding_dimensions
                        })
                    else:
                        # Insert new document
                        insert_query = sql_text(
                            """
                            INSERT INTO embeddings 
                            (id, text, embedding, metadata, embedding_model, embedding_dimensions) 
                            VALUES 
                            (:id, :text, array_to_vector(:embedding), :metadata, :model, :dimensions)
                            """
                        )
                        conn.execute(insert_query, {
                            "id": doc['id'],
                            "text": doc['text'],
                            "embedding": doc['embedding'],
                            "metadata": json.dumps(doc['metadata']) if 'metadata' in doc and doc['metadata'] else None,
                            "model": embedding_model,
                            "dimensions": embedding_dimensions
                        })
                
                conn.commit()
            
            self.logger.debug(f"Successfully stored {len(documents)} embeddings in batch")
            return True
            
        except Exception as e:
            self.logger.error(f"Error batch storing embeddings in PostgreSQL: {e}", exc_info=True)
            return False
    
    def count_embeddings(self) -> int:
        """
        Count the number of embeddings in the database.
        
        Returns:
            int: Number of embeddings, or -1 if error.
        """
        if not self.engine:
            self.logger.error("Database connection not established")
            return -1
            
        try:
            with self.engine.connect() as conn:
                query = sql_text("SELECT COUNT(*) FROM embeddings")
                result = conn.execute(query).fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error counting embeddings in PostgreSQL: {e}")
            return -1


class SupabaseProvider(DBProvider):
    """Supabase database provider for vector embeddings."""
    
    def __init__(self):
        """Initialize the Supabase provider."""
        super().__init__()
        self.url = os.getenv('SUPABASE_URL', '')
        self.key = os.getenv('SUPABASE_KEY', '')
        self.table = os.getenv('SUPABASE_TABLE', 'embeddings')
        self.client = None
        self.logger.debug(f"Supabase config: url={self.url[:20]}..., table={self.table}")
    
    def connect(self) -> bool:
        """
        Connect to Supabase.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            if not self.url or not self.key:
                self.logger.error("Supabase URL or key not provided")
                return False
            
            self.logger.info(f"Connecting to Supabase at {self.url[:20]}...")
            
            # Create HTTP client for Supabase
            self.client = httpx.Client(
                base_url=self.url,
                headers={
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            
            # Test connection
            self.logger.debug(f"Testing connection to table '{self.table}'")
            response = self.client.get(f"/rest/v1/{self.table}?limit=1")
            response.raise_for_status()
            
            self.logger.info("Successfully connected to Supabase")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to Supabase: {e}", exc_info=True)
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Supabase."""
        if self.client:
            self.logger.info("Disconnecting from Supabase")
            self.client.close()
    
    def store_embedding(self, doc_id: str, text: str, embedding: list, metadata: Dict[str, Any]) -> bool:
        """
        Store an embedding in Supabase.
        
        Args:
            doc_id: Document ID.
            text: The text that was embedded.
            embedding: The embedding vector.
            metadata: Additional metadata for the document.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.logger.debug(f"Storing embedding for document '{doc_id}'")
            
            # Check if document already exists
            response = self.client.get(
                f"/rest/v1/{self.table}",
                params={"id": f"eq.{doc_id}", "select": "id"}
            )
            response.raise_for_status()
            
            data = {
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": json.dumps(metadata)
            }
            
            if response.json():
                self.logger.debug(f"Document '{doc_id}' already exists, updating")
                # Update existing document
                response = self.client.patch(
                    f"/rest/v1/{self.table}",
                    params={"id": f"eq.{doc_id}"},
                    json=data
                )
            else:
                self.logger.debug(f"Document '{doc_id}' is new, inserting")
                # Insert new document
                response = self.client.post(
                    f"/rest/v1/{self.table}",
                    json=data
                )
            
            response.raise_for_status()
            
            log_db_operation(self.logger, "store", {
                "doc_id": doc_id,
                "text_length": len(text),
                "embedding_dimensions": len(embedding),
                "metadata_keys": list(metadata.keys()) if metadata else []
            })
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing embedding in Supabase: {e}", exc_info=True)
            return False
    
    def search_similar(self, embedding: list, top_k: int = 5) -> list:
        """
        Search for similar documents in Supabase using vector similarity.
        
        Args:
            embedding: The query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            list: List of similar documents with their similarity scores.
        """
        try:
            self.logger.debug(f"Searching for similar documents, top_k={top_k}")
            
            # Supabase uses RPC for vector similarity search
            response = self.client.post(
                "/rest/v1/rpc/match_documents",
                json={
                    "query_embedding": embedding,
                    "match_count": top_k,
                    "table_name": self.table
                }
            )
            response.raise_for_status()
            
            results = response.json()
            
            # Format results
            formatted_results = []
            for item in results:
                formatted_results.append({
                    "id": item["id"],
                    "text": item["text"],
                    "similarity": item["similarity"],
                    "metadata": json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]
                })
            
            self.logger.debug(f"Found {len(formatted_results)} similar documents")
            log_db_operation(self.logger, "search", {
                "embedding_dimensions": len(embedding),
                "top_k": top_k,
                "results_count": len(formatted_results)
            })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error searching similar documents in Supabase: {e}", exc_info=True)
            return []


def get_db_provider() -> DBProvider:
    """
    Get the configured database provider.
    
    Returns:
        DBProvider: The configured database provider.
    """
    provider_name = os.getenv('DB_PROVIDER', 'postgres').lower()
    
    logger.info(f"Using database provider: {provider_name}")
    
    if provider_name == 'supabase':
        return SupabaseProvider()
    else:
        return PostgresVectorProvider() 