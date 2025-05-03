"""
RAG (Retrieval-Augmented Generation) module for the LLM Comparison Tool.
Provides functionality for document ingestion, indexing, and retrieval.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import pickle

from llama_index.core import Settings
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter

from src.shared.config import DATA_DIR, CACHE_DIR, get_config
from src.features.llm_compare.embeddings import get_embedding_provider

class RAGProcessor:
    """Class for handling Retrieval-Augmented Generation workflows."""
    
    def __init__(self):
        """Initialize the RAG processor."""
        # Get configuration
        rag_config = get_config("rag")
        self.chunk_size = rag_config.get("chunk_size", 512)
        self.chunk_overlap = rag_config.get("chunk_overlap", 128)
        self.similarity_top_k = rag_config.get("similarity_top_k", 5)
        
        # Create cache directory for indices
        self.index_cache_dir = Path(CACHE_DIR) / "indices"
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding provider
        self.embedding_provider = get_embedding_provider()
        
        # Setup llama_index with our embedding provider
        self._setup_llama_index()
    
    def _setup_llama_index(self):
        """Configure llama_index with custom embedding model."""
        # Create a wrapper for the embedding provider to match llama_index's expected API
        def get_text_embedding(text: str) -> List[float]:
            return self.embedding_provider.get_embedding(text)
            
        def get_text_embeddings(texts: List[str]) -> List[List[float]]:
            return self.embedding_provider.get_embeddings(texts)
        
        # Set up global llama_index settings
        Settings.embed_model = get_text_embedding
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
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
        # Create the index
        index = VectorStoreIndex.from_documents(documents)
        
        # Cache the index for later use
        cache_path = self.index_cache_dir / f"{index_name}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(index, f)
        
        print(f"Created and cached index '{index_name}' with {len(documents)} documents")
        return index
    
    def load_index(self, index_name: str) -> Optional[VectorStoreIndex]:
        """
        Load a cached index by name.
        
        Args:
            index_name: Name of the index to load.
            
        Returns:
            VectorStoreIndex object or None if not found.
        """
        cache_path = self.index_cache_dir / f"{index_name}.pkl"
        
        if not cache_path.exists():
            print(f"Index '{index_name}' not found in cache")
            return None
        
        try:
            with open(cache_path, "rb") as f:
                index = pickle.load(f)
            print(f"Loaded index '{index_name}' from cache")
            return index
        except Exception as e:
            print(f"Error loading index '{index_name}' from cache: {e}")
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