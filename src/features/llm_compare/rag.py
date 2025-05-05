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
        # Initialize the database connection
        self.db_provider = get_db_provider()
        
        # Get embedding provider from app if available or create new one
        import streamlit as st
        if hasattr(st, 'session_state') and 'embedding_provider' in st.session_state and st.session_state.embedding_provider is not None:
            self.embedding_provider = st.session_state.embedding_provider
            print("Using embedding provider from session state")
        else:
            # Fallback to creating a new one 
            self.embedding_provider = get_embedding_provider()
            print("Created new embedding provider")
        
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
        print("Setting up llama_index with custom embedding model")
        
        # Try patching the OpenAIEmbedding class directly
        try:
            # Try importing from the new location first
            try:
                from llama_index.embeddings.openai import OpenAIEmbedding
                print("Imported OpenAIEmbedding from llama_index.embeddings.openai")
            except ImportError:
                # Fall back to older location
                from llama_index.core.embeddings.openai import OpenAIEmbedding
                print("Imported OpenAIEmbedding from llama_index.core.embeddings.openai")
            
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
                print(f"Using custom embedding for query: {query[:20]}...")
                return provider.get_embedding(query)
            
            def _monkey_get_text_embedding(self, text):
                print(f"Using custom embedding for text: {text[:20]}...")
                return provider.get_embedding(text)
                
            def _monkey_get_text_embeddings(self, texts):
                print(f"Using custom embeddings for {len(texts)} texts")
                return provider.get_embeddings(texts)
            
            # Patch the instance methods directly
            import types
            embed_model._get_query_embedding = types.MethodType(_monkey_get_query_embedding, embed_model)
            embed_model._get_text_embedding = types.MethodType(_monkey_get_text_embedding, embed_model)
            embed_model._get_text_embeddings = types.MethodType(_monkey_get_text_embeddings, embed_model)
            
            # Also define the async methods to call our sync methods
            async def _monkey_aget_query_embedding(self, query):
                print(f"Using async custom embedding for query: {query[:20]}...")
                return provider.get_embedding(query)
                
            async def _monkey_aget_text_embedding(self, text):
                print(f"Using async custom embedding for text: {text[:20]}...")
                return provider.get_embedding(text)
                
            async def _monkey_aget_text_embeddings(self, texts):
                print(f"Using async custom embeddings for {len(texts)} texts")
                return provider.get_embeddings(texts)
            
            # Patch the async methods
            embed_model._aget_query_embedding = types.MethodType(_monkey_aget_query_embedding, embed_model)
            embed_model._aget_text_embedding = types.MethodType(_monkey_aget_text_embedding, embed_model)
            embed_model._aget_text_embeddings = types.MethodType(_monkey_aget_text_embeddings, embed_model)
            
            print("Successfully patched OpenAIEmbedding methods")
            
        except Exception as e:
            print(f"Error setting up custom embedding model: {e}")
            # If we can't patch OpenAIEmbedding, fall back to direct function
            from llama_index.core.embeddings import resolve_embed_model
            
            # Create a simple embedding function
            def get_embedding_func(text):
                return self.embedding_provider.get_embedding(text)
            
            embed_model = get_embedding_func
            print("Using simple embedding function as fallback")
        
        # Set up global llama_index settings
        Settings.embed_model = embed_model
        print(f"Set embed_model to {embed_model}")
        
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        print("Llama_index settings configured successfully")
    
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
            print(f"Loaded {len(documents)} documents from {directory}")
            return documents
        except Exception as e:
            print(f"Error loading documents from {directory}: {e}")
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
            print(f"Loaded {len(documents)} documents from {len(file_paths)} files")
            return documents
        except Exception as e:
            print(f"Error loading documents from files: {e}")
            return []
    
    def create_index(self, documents: List[Document], index_name: str) -> VectorStoreIndex:
        """
        Create a vector store index from documents.
        
        Args:
            documents: List of Document objects.
            index_name: Name for the index (used for caching).
            
        Returns:
            VectorStoreIndex object.
        """
        import sys
        import traceback
        
        print(f"Starting to create index '{index_name}' with {len(documents)} documents")
        
        # First, let's save the documents separately in case indexing fails
        try:
            # Create a safety backup of documents
            backup_dir = self.index_cache_dir / f"{index_name}_backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each document separately to avoid one large serialization that could fail
            for i, doc in enumerate(documents):
                try:
                    doc_path = backup_dir / f"doc_{i}.pkl"
                    with open(doc_path, "wb") as f:
                        pickle.dump(doc, f)
                except Exception as e:
                    print(f"Warning: Failed to save document {i} backup: {e}")
            
            print(f"Created document backups in {backup_dir}")
        except Exception as e:
            print(f"Warning: Failed to create document backups: {e}")
            print(traceback.format_exc())
        
        try:
            # Try to create the index in smaller batches to reduce memory usage
            # and make it easier to diagnose where a failure might occur
            
            print("Splitting documents into batches for indexing...")
            batch_size = min(10, max(1, len(documents) // 5))  # Use smaller batches for larger document sets
            batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
            print(f"Created {len(batches)} batches with approximately {batch_size} documents each")
            
            # Process each batch
            all_nodes = []
            for i, batch in enumerate(batches):
                try:
                    print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
                    # Create an index for this batch
                    batch_index = VectorStoreIndex.from_documents(batch)
                    print(f"Successfully created index for batch {i+1}")
                    
                    # Extract nodes from this batch's index
                    if hasattr(batch_index, "as_retriever") and hasattr(batch_index.as_retriever(), "get_nodes"):
                        nodes = batch_index.as_retriever().get_nodes()
                        all_nodes.extend(nodes)
                        print(f"Added {len(nodes)} nodes from batch {i+1}")
                    elif hasattr(batch_index, "index_struct") and hasattr(batch_index.index_struct, "all_nodes"):
                        batch_nodes = list(batch_index.index_struct.all_nodes.values())
                        all_nodes.extend(batch_nodes)
                        print(f"Added {len(batch_nodes)} nodes from batch {i+1} (older API)")
                    
                    # Clean up batch index to free memory
                    del batch_index
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing batch {i+1}: {e}")
                    print(traceback.format_exc())
                    # Continue with next batch
            
            # Now create the full index from all documents
            print(f"Creating final index from all {len(documents)} documents")
            index = VectorStoreIndex.from_documents(documents)
            print(f"Successfully created full index with {len(documents)} documents")
            
            # Cache the index for later use - but do it in a way that avoids pickling issues
            # Instead of pickling the entire index, save the essential components
            try:
                # Create directory for this index
                index_dir = self.index_cache_dir / index_name
                index_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created index directory at {index_dir}")
                
                # Save the documents separately - use smaller batches
                docs_dir = index_dir / "documents"
                docs_dir.mkdir(parents=True, exist_ok=True)
                
                for i, doc_batch in enumerate(batches):
                    try:
                        docs_path = docs_dir / f"documents_batch_{i}.pkl"
                        with open(docs_path, "wb") as f:
                            pickle.dump(doc_batch, f)
                        print(f"Saved document batch {i+1} to {docs_path}")
                    except Exception as e:
                        print(f"Error saving document batch {i+1}: {e}")
                        print(traceback.format_exc())
                
                # Save the index nodes separately
                try:
                    # Try the newer API
                    if hasattr(index, "as_retriever") and hasattr(index.as_retriever(), "get_nodes"):
                        nodes = index.as_retriever().get_nodes()
                        print(f"Retrieved {len(nodes)} nodes from index (newer API)")
                    elif hasattr(index, "index_struct") and hasattr(index.index_struct, "all_nodes"):
                        # Try the older API
                        nodes = list(index.index_struct.all_nodes.values())
                        print(f"Retrieved {len(nodes)} nodes from index (older API)")
                    else:
                        # Fallback to empty nodes if we can't access them
                        nodes = []
                        print("Could not retrieve nodes from index - API incompatibility")
                    
                    # Check if nodes is not empty before saving
                    if nodes:
                        # Save in smaller batches
                        nodes_dir = index_dir / "nodes"
                        nodes_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Split nodes into smaller batches for saving
                        node_batch_size = min(100, max(1, len(nodes) // 5))
                        node_batches = [nodes[i:i+node_batch_size] for i in range(0, len(nodes), node_batch_size)]
                        
                        for i, node_batch in enumerate(node_batches):
                            try:
                                nodes_path = nodes_dir / f"nodes_batch_{i}.pkl"
                                with open(nodes_path, "wb") as f:
                                    pickle.dump(node_batch, f)
                                print(f"Saved nodes batch {i+1}/{len(node_batches)} to {nodes_path}")
                            except Exception as e:
                                print(f"Error saving nodes batch {i+1}: {e}")
                                print(traceback.format_exc())
                    else:
                        print("No nodes to save - this is unusual and might indicate a problem")
                except Exception as e:
                    print(f"Warning: Could not save index nodes: {e}")
                    print(traceback.format_exc())
                
                # Also save the index ID for reference
                try:
                    index_id_path = index_dir / "index_id.txt"
                    with open(index_id_path, "w") as f:
                        f.write(str(getattr(index, "index_id", "unknown")))
                    print(f"Saved index ID to {index_id_path}")
                except Exception as e:
                    print(f"Error saving index ID: {e}")
                
                print(f"Created and cached index '{index_name}' with {len(documents)} documents")
                
                # Record in a manifest file that we've cached this index
                try:
                    manifest_path = self.index_cache_dir / "manifest.txt"
                    with open(manifest_path, "a") as f:
                        f.write(f"{index_name}\n")
                    print(f"Updated index manifest at {manifest_path}")
                except Exception as e:
                    print(f"Error updating manifest: {e}")
                    
            except Exception as e:
                print(f"Warning: Failed to cache index: {e}")
                print(traceback.format_exc())
            
            return index
            
        except Exception as e:
            print(f"CRITICAL ERROR creating index: {e}")
            print(traceback.format_exc())
            
            # Create a dummy index to return so the UI doesn't crash
            try:
                # Create a simple empty index as a fallback
                print("Creating fallback empty index")
                from llama_index.core import Document
                fallback_doc = Document(text="Error creating index. Please try again with fewer documents.")
                fallback_index = VectorStoreIndex.from_documents([fallback_doc])
                print("Successfully created fallback index")
                return fallback_index
            except Exception as fallback_error:
                print(f"Failed to create fallback index: {fallback_error}")
                # Re-raise the original error if we can't even create a fallback
                raise e
    
    def load_index(self, index_name: str) -> Optional[VectorStoreIndex]:
        """
        Load a cached index by name.
        
        Args:
            index_name: Name of the index to load.
            
        Returns:
            VectorStoreIndex object or None if not found.
        """
        print(f"Attempting to load index '{index_name}' from cache")
        
        # Check if we have a directory for this index
        index_dir = self.index_cache_dir / index_name
        if not index_dir.exists() or not index_dir.is_dir():
            print(f"Index '{index_name}' not found in cache")
            return None
        
        try:
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
                        print(f"Loaded {len(doc_batch)} documents from {doc_file}")
                    except Exception as e:
                        print(f"Error loading documents from {doc_file}: {e}")
                
                # Check if we found any documents
                if all_documents:
                    # Recreate the index from documents
                    print(f"Recreating index from {len(all_documents)} loaded documents")
                    index = VectorStoreIndex.from_documents(all_documents)
                    print(f"Successfully recreated index '{index_name}' from cache")
                    return index
                else:
                    print(f"No documents found for index '{index_name}' in cache")
                    return None
            else:
                # Try loading from legacy single file format as fallback
                docs_path = index_dir / "documents.pkl"
                if docs_path.exists():
                    with open(docs_path, "rb") as f:
                        documents = pickle.load(f)
                    
                    if isinstance(documents, list) and documents:
                        print(f"Loaded {len(documents)} documents from legacy cache format")
                        index = VectorStoreIndex.from_documents(documents)
                        print(f"Successfully recreated index '{index_name}' from legacy cache")
                        return index
                
                print(f"Documents for index '{index_name}' not found in cache")
                return None
        except Exception as e:
            print(f"Error loading index '{index_name}' from cache: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def retrieve(self, index: VectorStoreIndex, query: str, top_k: Optional[int] = None) -> List[Any]:
        """
        Retrieve relevant nodes from an index based on a query.
        
        Args:
            index: VectorStoreIndex to query.
            query: Query string.
            top_k: Number of results to return. If None, uses config default.
            
        Returns:
            List of retrieved nodes.
        """
        top_k = top_k or self.similarity_top_k
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k
        )
        
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes for query: {query}")
        return nodes
    
    def query(self, index: VectorStoreIndex, query: str) -> Dict[str, Any]:
        """
        Query an index and return the result and context.
        
        Args:
            index: VectorStoreIndex to query.
            query: Query string.
            
        Returns:
            Dictionary with response and context information.
        """
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        
        return {
            "response": str(response),
            "nodes": response.source_nodes,
            "metadata": {
                "query": query,
                "node_count": len(response.source_nodes)
            }
        }


def get_rag_processor() -> RAGProcessor:
    """
    Factory function to get a RAG processor instance.
    
    Returns:
        RAGProcessor instance.
    """
    return RAGProcessor() 