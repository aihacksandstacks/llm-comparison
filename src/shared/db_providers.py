"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Database providers for the LLM Comparison Tool.
Supports PostgreSQL with pgvector and Supabase.
"""

import os
from typing import Dict, Any, Union, Optional
import httpx
from sqlalchemy import create_engine, Column, String, JSON, Float, MetaData, Table, text
from sqlalchemy.dialects.postgresql import ARRAY
import numpy as np
import json

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
        self.logger.debug(f"PostgreSQL config: host={self.config['host']}, port={self.config['port']}, db={self.config['database']}")
    
    def connect(self) -> bool:
        """
        Connect to the PostgreSQL database.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            # Create connection URL
            connection_url = f"postgresql://{self.config['user']}:{self.config['password']}@{self.config['host']}:{self.config['port']}/{self.config['database']}"
            
            self.logger.info(f"Connecting to PostgreSQL at {self.config['host']}:{self.config['port']}/{self.config['database']}")
            
            # Create engine
            self.engine = create_engine(connection_url)
            
            # Create pgvector extension if it doesn't exist
            with self.engine.connect() as conn:
                self.logger.debug("Creating pgvector extension if it doesn't exist")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            
            # Define embeddings table
            self.embeddings_table = Table(
                'embeddings', 
                self.metadata,
                Column('id', String, primary_key=True),
                Column('text', String),
                Column('embedding', ARRAY(Float)),
                Column('metadata', JSON)
            )
            
            # Create tables if they don't exist
            self.logger.debug("Creating tables if they don't exist")
            self.metadata.create_all(self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info("Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to PostgreSQL: {e}", exc_info=True)
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the PostgreSQL database."""
        if self.engine:
            self.logger.info("Disconnecting from PostgreSQL database")
            self.engine.dispose()
    
    def store_embedding(self, doc_id: str, text: str, embedding: list, metadata: Dict[str, Any]) -> bool:
        """
        Store an embedding in the PostgreSQL database.
        
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
            with self.engine.connect() as conn:
                # Check if document already exists
                result = conn.execute(
                    self.embeddings_table.select().where(self.embeddings_table.c.id == doc_id)
                ).fetchone()
                
                if result:
                    self.logger.debug(f"Document '{doc_id}' already exists, updating")
                    # Update existing document
                    conn.execute(
                        self.embeddings_table.update().where(
                            self.embeddings_table.c.id == doc_id
                        ).values(
                            text=text,
                            embedding=embedding,
                            metadata=metadata
                        )
                    )
                else:
                    self.logger.debug(f"Document '{doc_id}' is new, inserting")
                    # Insert new document
                    conn.execute(
                        self.embeddings_table.insert().values(
                            id=doc_id,
                            text=text,
                            embedding=embedding,
                            metadata=metadata
                        )
                    )
                
                conn.commit()
                
            log_db_operation(self.logger, "store", {
                "doc_id": doc_id,
                "text_length": len(text),
                "embedding_dimensions": len(embedding),
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
        try:
            self.logger.debug(f"Searching for similar documents, top_k={top_k}")
            with self.engine.connect() as conn:
                # Use cosine similarity with pgvector
                query = text(f"""
                SELECT 
                    id, 
                    text, 
                    1 - (embedding <=> ARRAY{embedding}::float[]) AS similarity,
                    metadata
                FROM 
                    embeddings
                ORDER BY 
                    embedding <=> ARRAY{embedding}::float[]
                LIMIT 
                    {top_k}
                """)
                
                results = conn.execute(query).fetchall()
                
                # Format results
                formatted_results = []
                for row in results:
                    formatted_results.append({
                        "id": row[0],
                        "text": row[1],
                        "similarity": float(row[2]),
                        "metadata": row[3]
                    })
                
                self.logger.debug(f"Found {len(formatted_results)} similar documents")
                log_db_operation(self.logger, "search", {
                    "embedding_dimensions": len(embedding),
                    "top_k": top_k,
                    "results_count": len(formatted_results)
                })
                
                return formatted_results
        except Exception as e:
            self.logger.error(f"Error searching similar documents in PostgreSQL: {e}", exc_info=True)
            return []


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