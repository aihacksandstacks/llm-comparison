"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Tests for the RAG module.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from src.features.llm_compare.rag import RAGProcessor, get_rag_processor


@pytest.fixture
def mock_index_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "cache" / "indices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider for testing."""
    provider = MagicMock()
    provider.get_embedding.return_value = [0.1, 0.2, 0.3]
    provider.get_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return provider

@pytest.fixture
def mock_rag_config():
    """Mock RAG configuration for testing."""
    return {
        "chunk_size": 256,
        "chunk_overlap": 64,
        "similarity_top_k": 3
    }


class TestRAGProcessor:
    """Tests for the RAGProcessor class."""
    
    @patch("src.features.llm_compare.rag.get_config")
    @patch("src.features.llm_compare.rag.get_embedding_provider")
    @patch("src.features.llm_compare.rag.CACHE_DIR")
    def test_init(self, mock_cache_dir, mock_get_embedding_provider, mock_get_config, 
                 mock_embedding_provider, mock_rag_config, mock_index_cache_dir):
        """Test initialization of RAGProcessor."""
        # Setup
        mock_cache_dir.return_value = str(mock_index_cache_dir)
        mock_get_embedding_provider.return_value = mock_embedding_provider
        mock_get_config.return_value = mock_rag_config
        
        # Execute
        processor = RAGProcessor()
        
        # Assert
        mock_get_config.assert_called_once_with("rag")
        mock_get_embedding_provider.assert_called_once()
        assert processor.chunk_size == 256
        assert processor.chunk_overlap == 64
        assert processor.similarity_top_k == 3
        assert processor.embedding_provider == mock_embedding_provider
    
    @patch("src.features.llm_compare.rag.get_config")
    @patch("src.features.llm_compare.rag.get_embedding_provider")
    @patch("src.features.llm_compare.rag.CACHE_DIR")
    @patch("src.features.llm_compare.rag.SimpleDirectoryReader")
    def test_load_documents_from_directory(self, mock_reader, mock_cache_dir, 
                                          mock_get_embedding_provider, mock_get_config,
                                          mock_embedding_provider, mock_rag_config):
        """Test loading documents from a directory."""
        # Setup
        mock_cache_dir.return_value = "cache"
        mock_get_embedding_provider.return_value = mock_embedding_provider
        mock_get_config.return_value = mock_rag_config
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = [
            Document(text="Document 1"),
            Document(text="Document 2")
        ]
        mock_reader.return_value = mock_reader_instance
        
        # Execute
        processor = RAGProcessor()
        documents = processor.load_documents_from_directory("/test/dir")
        
        # Assert
        mock_reader.assert_called_once_with(input_dir="/test/dir")
        mock_reader_instance.load_data.assert_called_once()
        assert len(documents) == 2
        assert documents[0].text == "Document 1"
        assert documents[1].text == "Document 2"
    
    @patch("src.features.llm_compare.rag.get_config")
    @patch("src.features.llm_compare.rag.get_embedding_provider")
    @patch("src.features.llm_compare.rag.CACHE_DIR")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_create_index(self, mock_pickle_dump, mock_file_open, mock_cache_dir,
                         mock_get_embedding_provider, mock_get_config,
                         mock_embedding_provider, mock_rag_config, mock_index_cache_dir):
        """Test creating an index."""
        # Setup
        mock_cache_dir.return_value = str(mock_index_cache_dir)
        mock_get_embedding_provider.return_value = mock_embedding_provider
        mock_get_config.return_value = mock_rag_config
        
        documents = [
            Document(text="Document 1"),
            Document(text="Document 2")
        ]
        
        # Mock VectorStoreIndex.from_documents
        mock_index = MagicMock()
        
        # Execute
        with patch("src.features.llm_compare.rag.VectorStoreIndex") as mock_vector_store_index:
            mock_vector_store_index.from_documents.return_value = mock_index
            processor = RAGProcessor()
            result = processor.create_index(documents, "test_index")
        
        # Assert
        mock_vector_store_index.from_documents.assert_called_once_with(documents)
        assert result == mock_index
        mock_file_open.assert_called_once()
        mock_pickle_dump.assert_called_once_with(mock_index, mock_file_open())
    
    @patch("src.features.llm_compare.rag.get_config")
    @patch("src.features.llm_compare.rag.get_embedding_provider")
    @patch("src.features.llm_compare.rag.CACHE_DIR")
    def test_retrieve(self, mock_cache_dir, mock_get_embedding_provider, mock_get_config,
                     mock_embedding_provider, mock_rag_config):
        """Test retrieving nodes from an index."""
        # Setup
        mock_cache_dir.return_value = "cache"
        mock_get_embedding_provider.return_value = mock_embedding_provider
        mock_get_config.return_value = mock_rag_config
        
        mock_index = MagicMock()
        mock_nodes = [MagicMock(), MagicMock()]
        
        # Execute
        with patch("src.features.llm_compare.rag.VectorIndexRetriever") as mock_retriever_class:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = mock_nodes
            mock_retriever_class.return_value = mock_retriever
            
            processor = RAGProcessor()
            result = processor.retrieve(mock_index, "test query")
        
        # Assert
        mock_retriever_class.assert_called_once_with(
            index=mock_index,
            similarity_top_k=3
        )
        mock_retriever.retrieve.assert_called_once_with("test query")
        assert result == mock_nodes

    @patch("src.features.llm_compare.rag.get_rag_processor")
    def test_get_rag_processor(self, mock_get_rag_processor):
        """Test the get_rag_processor factory function."""
        # Setup
        mock_processor = MagicMock()
        mock_get_rag_processor.return_value = mock_processor
        
        # Execute
        result = get_rag_processor()
        
        # Assert
        assert result == mock_processor 