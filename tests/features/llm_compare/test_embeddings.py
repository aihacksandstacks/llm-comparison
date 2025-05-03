"""
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
    """Tests for the NomicEmbeddingProvider class."""
    
    @patch("src.features.llm_compare.embeddings.NOMIC_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.embeddings.CACHE_DIR")
    @patch("nomic.login")
    @patch("nomic.embed")
    def test_get_embeddings_no_cache(self, mock_embed, mock_login, mock_cache_dir_patch):
        """Test getting embeddings without caching."""
        # Setup
        mock_cache_dir_patch.return_value = str(mock_cache_dir_patch)
        mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Execute
        provider = NomicEmbeddingProvider(use_cache=False)
        result = provider.get_embeddings(["test1", "test2"])
        
        # Assert
        mock_login.assert_called_once_with("test_api_key")
        mock_embed.assert_called_once_with(["test1", "test2"], model="nomic-embed-text-v1.5")
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    @patch("src.features.llm_compare.embeddings.NOMIC_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.embeddings.CACHE_DIR")
    @patch("nomic.login")
    @patch("nomic.embed")
    def test_get_embedding_single_text(self, mock_embed, mock_login, mock_cache_dir_patch):
        """Test getting embedding for a single text."""
        # Setup
        mock_cache_dir_patch.return_value = str(mock_cache_dir_patch)
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        # Execute
        provider = NomicEmbeddingProvider(use_cache=False)
        result = provider.get_embedding("test")
        
        # Assert
        mock_login.assert_called_once_with("test_api_key")
        mock_embed.assert_called_once_with(["test"], model="nomic-embed-text-v1.5")
        assert result == [0.1, 0.2, 0.3]

class TestOpenAIEmbeddingProvider:
    """Tests for the OpenAIEmbeddingProvider class."""
    
    @patch("src.features.llm_compare.embeddings.OPENAI_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.embeddings.CACHE_DIR")
    def test_get_embeddings_no_cache(self, mock_cache_dir_patch):
        """Test getting embeddings without caching."""
        # Setup
        mock_cache_dir_patch.return_value = str(mock_cache_dir_patch)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        # Execute
        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIEmbeddingProvider(use_cache=False)
            result = provider.get_embeddings(["test1", "test2"])
        
        # Assert
        mock_client.embeddings.create.assert_called_once_with(
            input=["test1", "test2"],
            model="text-embedding-3-small"
        )
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

class TestGetEmbeddingProvider:
    """Tests for the get_embedding_provider function."""
    
    @patch("src.features.llm_compare.embeddings.get_config")
    @patch("src.features.llm_compare.embeddings.NomicEmbeddingProvider")
    def test_get_provider_nomic(self, mock_nomic_provider, mock_get_config, mock_config):
        """Test getting Nomic embedding provider."""
        # Setup
        mock_get_config.return_value = mock_config["embeddings"]
        
        # Execute
        provider = get_embedding_provider()
        
        # Assert
        mock_get_config.assert_called_once_with("embeddings")
        mock_nomic_provider.assert_called_once_with(
            model="test-model",
            use_cache=True
        )
    
    @patch("src.features.llm_compare.embeddings.get_config")
    @patch("src.features.llm_compare.embeddings.OpenAIEmbeddingProvider")
    def test_get_provider_openai(self, mock_openai_provider, mock_get_config, mock_config):
        """Test getting OpenAI embedding provider."""
        # Setup
        config = mock_config.copy()
        config["embeddings"]["provider"] = "openai"
        mock_get_config.return_value = config["embeddings"]
        
        # Execute
        provider = get_embedding_provider()
        
        # Assert
        mock_get_config.assert_called_once_with("embeddings")
        mock_openai_provider.assert_called_once_with(
            model="test-model",
            use_cache=True
        )
    
    @patch("src.features.llm_compare.embeddings.get_config")
    def test_get_provider_unsupported(self, mock_get_config, mock_config):
        """Test getting unsupported embedding provider."""
        # Setup
        config = mock_config.copy()
        config["embeddings"]["provider"] = "unsupported"
        mock_get_config.return_value = config["embeddings"]
        
        # Execute and Assert
        with pytest.raises(ValueError, match="Unsupported embedding provider: unsupported"):
            provider = get_embedding_provider() 