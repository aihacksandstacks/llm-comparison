"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

RAG (Retrieval-Augmented Generation) module for the LLM Comparison Tool.
Provides functionality for document ingestion, indexing, and retrieval.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import pickle
import logging

from llama_index.core import Settings, ServiceContext
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
# Import the common embeddings interface
from llama_index.core.embeddings import BaseEmbedding

from src.shared.config import DATA_DIR, CACHE_DIR, get_config
from src.features.llm_compare.embeddings import get_embedding_provider
from src.shared.db_providers import get_db_provider
from src.shared.logger import get_logger

# Create our own implementation of BaseEmbedding
class AdapterEmbedding(BaseEmbedding):
    """Simple adapter class that implements the BaseEmbedding interface."""
    
    def __init__(self, provider):
        """Initialize with the embedding provider."""
        self._provider = provider
        self.model_name = getattr(provider, 'model_name', 'custom')
        # Don't call super().__init__ as it may use Pydantic validation
        
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query."""
        return self._provider.get_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        return self._provider.get_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        return self._provider.get_embeddings(texts)

class RAGProcessor:
    """Class for handling Retrieval-Augmented Generation workflows."""
    
    def __init__(self):
        """Initialize the RAG processor."""
        # Set up logger early so we can use it during initialization
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize the database connection
        self.db_provider = get_db_provider()
        self.db_connected = self.db_provider.connect()
        
        # Set storage preference flag - database is now preferred
        self.use_db_as_primary = True
        
        if self.db_connected:
            self.logger.info("Successfully connected to vector database - using as primary storage")
        else:
            self.logger.warning("Database connection failed. Falling back to in-memory storage only.")
        
        # Get embedding provider from app if available or create new one
        import streamlit as st
        if hasattr(st, 'session_state') and 'embedding_provider' in st.session_state and st.session_state.embedding_provider is not None:
            self.embedding_provider = st.session_state.embedding_provider
            self.logger.info("Using embedding provider from session state")
        else:
            # Fallback to creating a new one 
            self.embedding_provider = get_embedding_provider()
            self.logger.info("Created new embedding provider")
        
        # Default prompt template
        self.DEFAULT_PROMPT_TEMPLATE = (
            "You are a helpful, respectful and honest AI assistant. "
            "Answer the question based on the context provided.\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n"
            "Answer: "
        )
        
        # Get configuration
        rag_config = get_config("rag")
        self.chunk_size = rag_config.get("chunk_size", 512)
        self.chunk_overlap = rag_config.get("chunk_overlap", 128)
        self.similarity_top_k = rag_config.get("similarity_top_k", 5)
        
        # Create cache directory for indices
        self.index_cache_dir = Path(CACHE_DIR) / "indices"
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup llama_index with our embedding provider
        self._setup_llama_index()
    
    def _setup_llama_index(self):
        """Configure llama_index with custom embedding model."""
        self.logger.info("Setting up llama_index with custom embedding model")
        
        # Try patching the OpenAIEmbedding class directly
        try:
            # Try importing from the new location first
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
                self.logger.info("Imported OpenAIEmbedding from llama_index.embeddings.openai")
            except ImportError:
                # Fall back to older location
                from llama_index.core.embeddings.openai import OpenAIEmbedding
                self.logger.info("Imported OpenAIEmbedding from llama_index.core.embeddings.openai")
            
            # Create an instance to use with patched methods
            embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key="dummy")
            
            # Store the original async methods for reference
            _original_aget_query_embedding = embed_model._aget_query_embedding
            _original_aget_text_embedding = embed_model._aget_text_embedding
            _original_aget_text_embeddings = embed_model._aget_text_embeddings
            
            # Create our provider
            provider = self.embedding_provider
            
            # Replace the embedding methods with our versions that use our provider
            def _monkey_get_query_embedding(self, query):
                self.logger.debug(f"Using custom embedding for query: {query[:20]}...")
                return provider.get_embedding(query)
            
            def _monkey_get_text_embedding(self, text):
                self.logger.debug(f"Using custom embedding for text: {text[:20]}...")
                return provider.get_embedding(text)
                
            def _monkey_get_text_embeddings(self, texts):
                self.logger.debug(f"Using custom embeddings for {len(texts)} texts")
                return provider.get_embeddings(texts)
            
            # Patch the instance methods directly
            import types
            embed_model._get_query_embedding = types.MethodType(_monkey_get_query_embedding, embed_model)
            embed_model._get_text_embedding = types.MethodType(_monkey_get_text_embedding, embed_model)
            embed_model._get_text_embeddings = types.MethodType(_monkey_get_text_embeddings, embed_model)
            
            # Also define the async methods to call our sync methods
            async def _monkey_aget_query_embedding(self, query):
                self.logger.debug(f"Using async custom embedding for query: {query[:20]}...")
                return provider.get_embedding(query)
                
            async def _monkey_aget_text_embedding(self, text):
                self.logger.debug(f"Using async custom embedding for text: {text[:20]}...")
                return provider.get_embedding(text)
                
            async def _monkey_aget_text_embeddings(self, texts):
                self.logger.debug(f"Using async custom embeddings for {len(texts)} texts")
                return provider.get_embeddings(texts)
            
            # Patch the async methods
            embed_model._aget_query_embedding = types.MethodType(_monkey_aget_query_embedding, embed_model)
            embed_model._aget_text_embedding = types.MethodType(_monkey_aget_text_embedding, embed_model)
            embed_model._aget_text_embeddings = types.MethodType(_monkey_aget_text_embeddings, embed_model)
            
            self.logger.info("Successfully patched OpenAIEmbedding methods")
            
        except Exception as e:
            self.logger.error(f"Error setting up custom embedding model: {e}", exc_info=True)
            # If we can't patch OpenAIEmbedding, fall back to direct function
            from llama_index.core.embeddings import resolve_embed_model
            
            # Create a simple embedding function
            def get_embedding_func(text):
                return self.embedding_provider.get_embedding(text)
            
            embed_model = get_embedding_func
            self.logger.info("Using simple embedding function as fallback")
        
        # Set up global llama_index settings
        Settings.embed_model = embed_model
        self.logger.info(f"Set embed_model to {embed_model}")
        
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.logger.info("Llama_index settings configured successfully")
    
    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """
        Load documents from a directory.
        
        Args:
            directory: Path to the directory containing documents.
            
        Returns:
            List of Document objects.
        """
        try:
            directory_path = Path(directory)
            reader = SimpleDirectoryReader(input_dir=str(directory_path))
            documents = reader.load_data()
            self.logger.info(f"Loaded {len(documents)} documents from {directory}")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading documents from {directory}: {e}", exc_info=True)
            return []
    
    def load_documents_from_files(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from a list of file paths.
        
        Args:
            file_paths: List of paths to document files.
            
        Returns:
            List of Document objects.
        """
        try:
            reader = SimpleDirectoryReader(input_files=file_paths)
            documents = reader.load_data()
            self.logger.info(f"Loaded {len(documents)} documents from {len(file_paths)} files")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading documents from files: {e}", exc_info=True)
            return []
    
    def create_index(self, documents: List[Document], index_name: str) -> VectorStoreIndex:
        """
        Create a new index from documents, prioritizing database storage.
        
        Args:
            documents: List of Document objects to index.
            index_name: Name of the index.
            
        Returns:
            The created VectorStoreIndex.
        """
        try:
            self.logger.info(f"Creating index '{index_name}' with {len(documents)} documents")
            
            # Store documents in the vector database first if connection is available
            db_success = False
            if self.db_connected:
                self.logger.info("Storing documents in database as primary storage")
                db_success = self._store_documents_in_db(documents, index_name)
                if db_success:
                    self.logger.info("Documents successfully stored in database")
                else:
                    self.logger.warning("Failed to store documents in database, falling back to in-memory index")
            
            # Also create an in-memory index as a fallback
            index = VectorStoreIndex.from_documents(documents)
            
            # Save the index to disk for persistence between sessions
            index_path = self.index_cache_dir / index_name
            index_path.mkdir(parents=True, exist_ok=True)
            
            with open(index_path / "index.pkl", "wb") as f:
                pickle.dump(index, f)
            
            self.logger.info(f"In-memory index '{index_name}' created and saved to disk as backup")
            
            return index
        except Exception as e:
            self.logger.error(f"Error creating index: {e}", exc_info=True)
            return None
    
    def _store_documents_in_db(self, documents: List[Document], collection_name: str) -> bool:
        """
        Store documents in the vector database.
        
        Args:
            documents: List of Document objects to store.
            collection_name: Name of the collection (used as metadata).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        self.logger.info(f"Storing {len(documents)} documents in vector database under collection '{collection_name}'")
        
        try:
            # Process all documents into a format for batch storage
            db_documents = []
            
            for i, doc in enumerate(documents):
                # Get document text and metadata
                text = doc.get_content()
                doc_id = getattr(doc, "doc_id", None) or f"{collection_name}_{i}"
                
                # Get document metadata or create empty dict
                metadata = getattr(doc, "metadata", {}) or {}
                metadata["collection"] = collection_name
                metadata["index"] = i
                
                # Get embedding for the document
                embedding = self.embedding_provider.get_embedding(text)
                
                # Add to batch
                db_documents.append({
                    "id": doc_id,
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata
                })
            
            # Store documents in batch
            if hasattr(self.db_provider, "batch_store_embeddings"):
                success = self.db_provider.batch_store_embeddings(db_documents)
            else:
                # Fall back to individual storage
                success = True
                for doc in db_documents:
                    result = self.db_provider.store_embedding(
                        doc_id=doc["id"],
                        text=doc["text"],
                        embedding=doc["embedding"],
                        metadata=doc["metadata"]
                    )
                    if not result:
                        success = False
            
            self.logger.info(f"{'Successfully' if success else 'Failed to'} store documents in vector database")
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing documents in vector database: {e}", exc_info=True)
            return False
    
    def load_index(self, index_name: str) -> Optional[VectorStoreIndex]:
        """
        Load a cached index by name.
        
        Args:
            index_name: Name of the index to load.
            
        Returns:
            VectorStoreIndex object or None if not found.
        """
        self.logger.info(f"Attempting to load index '{index_name}' from cache")
        
        # Check if we have a directory for this index
        index_dir = self.index_cache_dir / index_name
        if not index_dir.exists() or not index_dir.is_dir():
            self.logger.info(f"Index '{index_name}' not found in cache")
            return None
        
        try:
            # First try loading the index directly from the index.pkl file
            index_file = index_dir / "index.pkl"
            if index_file.exists():
                try:
                    self.logger.info(f"Found index.pkl for '{index_name}', loading directly")
                    with open(index_file, "rb") as f:
                        index = pickle.load(f)
                    self.logger.info(f"Successfully loaded index '{index_name}' directly from index.pkl")
                    return index
                except Exception as e:
                    self.logger.warning(f"Failed to load index from {index_file}: {e}")
                    # Continue to fallback methods below
            
            # If we didn't succeed with direct loading, try reconstructing from documents
            # Load documents
            docs_dir = index_dir / "documents"
            if docs_dir.exists() and docs_dir.is_dir():
                # Get all document batch files
                doc_files = list(docs_dir.glob("documents_batch_*.pkl"))
                all_documents = []
                
                for doc_file in doc_files:
                    try:
                        with open(doc_file, "rb") as f:
                            doc_batch = pickle.load(f)
                            all_documents.extend(doc_batch)
                        self.logger.info(f"Loaded {len(doc_batch)} documents from {doc_file}")
                    except Exception as e:
                        self.logger.error(f"Error loading documents from {doc_file}: {e}", exc_info=True)
                
                # Check if we found any documents
                if all_documents:
                    # Recreate the index from documents
                    self.logger.info(f"Recreating index from {len(all_documents)} loaded documents")
                    index = VectorStoreIndex.from_documents(all_documents)
                    self.logger.info(f"Successfully recreated index '{index_name}' from cache")
                    return index
                else:
                    self.logger.info(f"No documents found for index '{index_name}' in cache")
                    return None
            else:
                # Try loading from legacy single file format as fallback
                docs_path = index_dir / "documents.pkl"
                if docs_path.exists():
                    with open(docs_path, "rb") as f:
                        documents = pickle.load(f)
                    
                    if isinstance(documents, list) and documents:
                        self.logger.info(f"Loaded {len(documents)} documents from legacy cache format")
                        index = VectorStoreIndex.from_documents(documents)
                        self.logger.info(f"Successfully recreated index '{index_name}' from legacy cache")
                        return index
                
                self.logger.info(f"Documents for index '{index_name}' not found in cache")
                return None
        except Exception as e:
            self.logger.error(f"Error loading index '{index_name}' from cache: {e}", exc_info=True)
            return None
    
    def retrieve(self, index: VectorStoreIndex, query: str, top_k: Optional[int] = None) -> List[Any]:
        """
        Retrieve relevant documents from an index.
        
        Args:
            index: The VectorStoreIndex to query (used as fallback).
            query: The query string.
            top_k: Number of results to return. If None, uses the default.
            
        Returns:
            List of retrieved documents/nodes.
        """
        top_k = top_k or self.similarity_top_k
        
        try:
            # Try using the database first if connected
            if self.db_connected:
                self.logger.info(f"Retrieving documents from database for query: '{query}'")
                query_embedding = self.embedding_provider.get_embedding(query)
                results = self.db_provider.search_similar(query_embedding, top_k=top_k)
                
                # If we got results from the database, use them
                if results:
                    # Convert to a format similar to retriever nodes
                    nodes = []
                    for result in results:
                        # Create a simple node-like object
                        node = {
                            "id": result["id"],
                            "text": result["text"],
                            "metadata": result["metadata"],
                            "similarity": result["similarity"]
                        }
                        nodes.append(node)
                    
                    self.logger.info(f"Retrieved {len(nodes)} documents from database")
                    return nodes
                else:
                    self.logger.info("No results from database, falling back to in-memory index")
                
            # Fall back to in-memory index if database not available or no results
            if index:
                self.logger.info(f"Retrieving documents from in-memory index for query: '{query}'")
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=top_k
                )
                nodes = retriever.retrieve(query)
                self.logger.info(f"Retrieved {len(nodes)} documents from in-memory index")
                return nodes
            
            # No retrieval methods available
            self.logger.warning("No retrieval methods available. Database returned no results and in-memory index is not available.")
            return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}", exc_info=True)
            return []
    
    def query(self, index: VectorStoreIndex, query: str) -> Dict[str, Any]:
        """
        Query for information and return the result and context.
        
        Args:
            index: VectorStoreIndex to query (used as fallback).
            query: Query string.
            
        Returns:
            Dictionary with response and context information.
        """
        try:
            # Try using the database first if connected
            if self.db_connected:
                self.logger.info(f"Querying database for: '{query}'")
                query_embedding = self.embedding_provider.get_embedding(query)
                results = self.db_provider.search_similar(query_embedding, top_k=self.similarity_top_k)
                
                # If we got results from the database, use them
                if results:
                    # Extract text from results
                    context_texts = [result["text"] for result in results]
                    context = "\n\n".join(context_texts)
                    
                    return {
                        "response": context,
                        "source_nodes": results,
                        "metadata": {
                            "query": query,
                            "node_count": len(results),
                            "from_database": True
                        }
                    }
                else:
                    self.logger.info("No results from database, falling back to in-memory index")
            
            # Fall back to in-memory index if database not available or no results
            if index:
                self.logger.info(f"Querying in-memory index for: '{query}'")
                query_engine = index.as_query_engine()
                response = query_engine.query(query)
                
                return {
                    "response": str(response),
                    "source_nodes": response.source_nodes,
                    "metadata": {
                        "query": query,
                        "node_count": len(response.source_nodes),
                        "from_memory": True
                    }
                }
            
            # No query methods available
            self.logger.warning("Query failed. Database returned no results and in-memory index is not available.")
            return {
                "response": "Error: No data sources available for query.",
                "source_nodes": [],
                "metadata": {
                    "query": query,
                    "node_count": 0,
                    "error": True
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "response": f"Error processing query: {str(e)}",
                "source_nodes": [],
                "metadata": {
                    "query": query,
                    "node_count": 0,
                    "error": True,
                    "error_message": str(e)
                }
            }

    def __del__(self):
        """Clean up resources when the processor is deleted."""
        try:
            if hasattr(self, 'db_provider') and self.db_provider:
                self.db_provider.disconnect()
        except Exception as e:
            # Don't use logger here as it might be destructed already
            self.logger.error(f"Error disconnecting from database: {e}", exc_info=True)

def get_rag_processor() -> RAGProcessor:
    """
    Factory function to get a RAG processor instance.
    
    Returns:
        RAGProcessor instance.
    """
    return RAGProcessor() 