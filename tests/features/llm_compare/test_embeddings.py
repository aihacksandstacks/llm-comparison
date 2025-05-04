"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Tests for the embeddings module.
"""

import os
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.features.llm_compare.embeddings import (
    EmbeddingProvider,
    NomicEmbeddingProvider,
    OpenAIEmbeddingProvider,
    LocalEmbeddingProvider,
    NomicLocalEmbeddingProvider,
    get_embedding_provider,
)

@pytest.fixture
def mock_cache_dir(tmp_path):
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "cache" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "embeddings": {
            "provider": "nomic",
            "model": "test-model",
            "cache_enabled": True
        }
    }

class TestNomicEmbeddingProvider:
    """Tests for NomicEmbeddingProvider."""
    
    @patch("src.features.llm_compare.embeddings.NOMIC_API_KEY", "test-key")
    @patch("nomic.login")
    def test_init(self, mock_login):
        """Test initialization."""
        provider = NomicEmbeddingProvider()
        mock_login.assert_called_once_with("test-key")
        assert provider.model == "nomic-embed-text-v1.5"
        assert provider.use_cache is True
        
    @patch("src.features.llm_compare.embeddings.NOMIC_API_KEY", "test-key")
    @patch("nomic.login")
    @patch("nomic.embed")
    def test_get_embeddings(self, mock_embed, mock_login):
        """Test get_embeddings without cache."""
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        provider = NomicEmbeddingProvider(use_cache=False)
        result = provider.get_embeddings(["text1", "text2"])
        
        mock_embed.assert_called_once_with(["text1", "text2"], model="nomic-embed-text-v1.5")
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        
    @patch("src.features.llm_compare.embeddings.NOMIC_API_KEY", "test-key")
    @patch("nomic.login")
    @patch("nomic.embed")
    def test_get_embedding(self, mock_embed, mock_login):
        """Test get_embedding."""
        mock_embed.return_value = [[0.1, 0.2]]
        
        provider = NomicEmbeddingProvider(use_cache=False)
        result = provider.get_embedding("text1")
        
        mock_embed.assert_called_once_with(["text1"], model="nomic-embed-text-v1.5")
        assert result == [0.1, 0.2]


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider."""
    
    @patch("src.features.llm_compare.embeddings.OPENAI_API_KEY", "test-key")
    @patch("openai.OpenAI")
    def test_init(self, mock_openai):
        """Test initialization."""
        provider = OpenAIEmbeddingProvider()
        mock_openai.assert_called_once_with(api_key="test-key")
        assert provider.model == "text-embedding-3-small"
        assert provider.use_cache is True
        
    @patch("src.features.llm_compare.embeddings.OPENAI_API_KEY", "test-key")
    @patch("openai.OpenAI")
    def test_get_embeddings(self, mock_openai):
        """Test get_embeddings without cache."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        provider = OpenAIEmbeddingProvider(use_cache=False)
        result = provider.get_embeddings(["text1", "text2"])
        
        mock_client.embeddings.create.assert_called_once_with(
            input=["text1", "text2"],
            model="text-embedding-3-small"
        )
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        
    @patch("src.features.llm_compare.embeddings.OPENAI_API_KEY", "test-key")
    @patch("openai.OpenAI")
    def test_get_embedding(self, mock_openai):
        """Test get_embedding."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response
        
        provider = OpenAIEmbeddingProvider(use_cache=False)
        result = provider.get_embedding("text1")
        
        mock_client.embeddings.create.assert_called_once_with(
            input=["text1"],
            model="text-embedding-3-small"
        )
        assert result == [0.1, 0.2]


class TestLocalEmbeddingProvider:
    """Tests for LocalEmbeddingProvider."""
    
    @patch("sentence_transformers.SentenceTransformer")
    def test_init(self, mock_st):
        """Test initialization."""
        provider = LocalEmbeddingProvider()
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")
        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.use_cache is True
        
    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embeddings(self, mock_st):
        """Test get_embeddings without cache."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        import numpy as np
        mock_model.encode.return_value = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ])
        
        provider = LocalEmbeddingProvider(use_cache=False)
        result = provider.get_embeddings(["text1", "text2"])
        
        mock_model.encode.assert_called_once_with(["text1", "text2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        
    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embedding(self, mock_st):
        """Test get_embedding."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        import numpy as np
        mock_model.encode.return_value = np.array([
            [0.1, 0.2]
        ])
        
        provider = LocalEmbeddingProvider(use_cache=False)
        result = provider.get_embedding("text1")
        
        mock_model.encode.assert_called_once_with(["text1"])
        assert result == [0.1, 0.2]


class TestNomicLocalEmbeddingProvider:
    """Tests for NomicLocalEmbeddingProvider."""
    
    @patch("sentence_transformers.SentenceTransformer")
    def test_init(self, mock_st):
        """Test initialization."""
        provider = NomicLocalEmbeddingProvider()
        mock_st.assert_called_once_with("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        assert provider.model_name == "nomic-ai/nomic-embed-text-v1"
        assert provider.task_type == "search_document"
        assert provider.use_cache is True
        
    @patch("sentence_transformers.SentenceTransformer")
    def test_init_with_invalid_task_type(self, mock_st):
        """Test initialization with invalid task type."""
        with pytest.raises(ValueError):
            NomicLocalEmbeddingProvider(task_type="invalid_task")
            
    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embeddings(self, mock_st):
        """Test get_embeddings without cache."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        import numpy as np
        mock_model.encode.return_value = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ])
        
        provider = NomicLocalEmbeddingProvider(use_cache=False)
        result = provider.get_embeddings(["text1", "text2"])
        
        # Should add the task prefix
        mock_model.encode.assert_called_once_with(["search_document: text1", "search_document: text2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        
    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embeddings_with_existing_prefix(self, mock_st):
        """Test get_embeddings with existing prefix."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        import numpy as np
        mock_model.encode.return_value = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ])
        
        provider = NomicLocalEmbeddingProvider(use_cache=False)
        result = provider.get_embeddings(["search_query: text1", "clustering: text2"])
        
        # Should not modify texts that already have prefixes
        mock_model.encode.assert_called_once_with(["search_query: text1", "clustering: text2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        
    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embedding(self, mock_st):
        """Test get_embedding."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        import numpy as np
        mock_model.encode.return_value = np.array([
            [0.1, 0.2]
        ])
        
        provider = NomicLocalEmbeddingProvider(use_cache=False, task_type="search_query")
        result = provider.get_embedding("text1")
        
        mock_model.encode.assert_called_once_with(["search_query: text1"])
        assert result == [0.1, 0.2]


class TestGetEmbeddingProvider:
    """Tests for get_embedding_provider."""
    
    @patch("src.features.llm_compare.embeddings.get_config")
    @patch("src.features.llm_compare.embeddings.NomicEmbeddingProvider")
    def test_get_nomic_provider(self, mock_nomic, mock_get_config):
        """Test getting a Nomic provider."""
        mock_get_config.return_value = {
            "provider": "nomic",
            "model": "test-model",
            "cache_enabled": True
        }
        
        provider = get_embedding_provider()
        
        mock_nomic.assert_called_once_with(
            model="test-model",
            use_cache=True
        )
        assert provider == mock_nomic.return_value
        
    @patch("src.features.llm_compare.embeddings.get_config")
    @patch("src.features.llm_compare.embeddings.OpenAIEmbeddingProvider")
    def test_get_openai_provider(self, mock_openai, mock_get_config):
        """Test getting an OpenAI provider."""
        mock_get_config.return_value = {
            "provider": "openai",
            "model": "test-model",
            "cache_enabled": True
        }
        
        provider = get_embedding_provider("openai")
        
        mock_openai.assert_called_once_with(
            model="test-model",
            use_cache=True
        )
        assert provider == mock_openai.return_value
        
    @patch("src.features.llm_compare.embeddings.get_config")
    @patch("src.features.llm_compare.embeddings.LocalEmbeddingProvider")
    def test_get_local_provider(self, mock_local, mock_get_config):
        """Test getting a Local provider."""
        mock_get_config.return_value = {
            "provider": "local",
            "model": "test-model",
            "cache_enabled": True
        }
        
        provider = get_embedding_provider("local")
        
        mock_local.assert_called_once_with(
            model="test-model",
            use_cache=True
        )
        assert provider == mock_local.return_value
        
    @patch("src.features.llm_compare.embeddings.get_config")
    @patch("src.features.llm_compare.embeddings.NomicLocalEmbeddingProvider")
    def test_get_nomic_local_provider(self, mock_nomic_local, mock_get_config):
        """Test getting a Nomic Local provider."""
        mock_get_config.return_value = {
            "provider": "nomic_local",
            "model": "test-model",
            "task_type": "search_query",
            "cache_enabled": True
        }
        
        provider = get_embedding_provider("nomic_local")
        
        mock_nomic_local.assert_called_once_with(
            model="test-model",
            task_type="search_query",
            use_cache=True
        )
        assert provider == mock_nomic_local.return_value 