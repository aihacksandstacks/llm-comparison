"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Tests for the database providers.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.shared.db_providers import PostgresVectorProvider, get_db_provider
from src.shared.config import DB_CONFIG

# Skip these tests if not running in CI or explicitly enabled
skip_db_tests = os.environ.get("ENABLE_DB_TESTS", "0").lower() not in ("1", "true", "yes")
skip_reason = "Database tests are disabled by default. Set ENABLE_DB_TESTS=1 to enable."

@pytest.fixture
def mock_engine():
    """Fixture for mocking SQLAlchemy engine."""
    with patch('src.shared.db_providers.create_engine') as mock_create_engine:
        # Create mock engine
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Mock connect method
        mock_conn = MagicMock()
        mock_engine.connect.return_value = mock_conn
        
        # Mock execute method
        mock_result = MagicMock()
        mock_conn.execute.return_value = mock_result
        
        yield mock_engine

class TestPostgresVectorProvider:
    """Tests for PostgresVectorProvider class."""
    
    def test_init(self):
        """Test initialization."""
        provider = PostgresVectorProvider()
        assert provider.config == DB_CONFIG
        assert provider.engine is None
        assert provider.connection_retries == 3
    
    @patch('src.shared.db_providers.create_engine')
    def test_connect_success(self, mock_create_engine):
        """Test successful connection."""
        # Set up mock
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Set up mock connection
        mock_conn = MagicMock()
        mock_engine.connect.return_value = mock_conn
        
        # Set up mock execution
        mock_result = MagicMock()
        mock_conn.execute.return_value = mock_result
        
        # Test connection
        provider = PostgresVectorProvider()
        result = provider.connect()
        
        assert result is True
        assert provider.engine is mock_engine
        mock_create_engine.assert_called_once()
        mock_engine.connect.assert_called()
    
    @patch('src.shared.db_providers.create_engine')
    def test_connect_failure(self, mock_create_engine):
        """Test failed connection."""
        # Set up mock to raise exception
        mock_create_engine.side_effect = Exception("Connection error")
        
        # Test connection
        provider = PostgresVectorProvider()
        result = provider.connect()
        
        assert result is False
        assert provider.engine is None
    
    def test_disconnect(self, mock_engine):
        """Test disconnection."""
        provider = PostgresVectorProvider()
        provider.engine = mock_engine
        
        provider.disconnect()
        
        mock_engine.dispose.assert_called_once()
    
    def test_store_embedding(self, mock_engine):
        """Test storing embedding."""
        # Set up provider
        provider = PostgresVectorProvider()
        provider.engine = mock_engine
        provider.embeddings_table = MagicMock()
        
        # Mock select/update/insert methods
        select_mock = MagicMock()
        provider.embeddings_table.select.return_value = select_mock
        select_mock.where.return_value = select_mock
        
        update_mock = MagicMock()
        provider.embeddings_table.update.return_value = update_mock
        update_mock.where.return_value = update_mock
        update_mock.values.return_value = update_mock
        
        insert_mock = MagicMock()
        provider.embeddings_table.insert.return_value = insert_mock
        insert_mock.values.return_value = insert_mock
        
        # Set up connection to return no results (new document)
        mock_conn = mock_engine.connect.return_value
        mock_conn.execute.return_value.fetchone.return_value = None
        
        # Test storing embedding
        result = provider.store_embedding(
            doc_id="test1",
            text="This is a test",
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"}
        )
        
        assert result is True
        mock_conn.execute.assert_called()  # Called multiple times
        provider.embeddings_table.insert.assert_called_once()
    
    def test_search_similar(self, mock_engine):
        """Test similarity search."""
        # Set up provider
        provider = PostgresVectorProvider()
        provider.engine = mock_engine
        
        # Set up mock rows for search results
        mock_row1 = ("doc1", "Text 1", 0.95, {"source": "test"})
        mock_row2 = ("doc2", "Text 2", 0.85, {"source": "test"})
        mock_conn = mock_engine.connect.return_value
        mock_conn.execute.return_value.fetchall.return_value = [mock_row1, mock_row2]
        
        # Test search
        results = provider.search_similar([0.1, 0.2, 0.3], top_k=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["similarity"] == 0.95
        assert results[1]["id"] == "doc2"
        mock_conn.execute.assert_called_once()
    
    def test_health_check(self, mock_engine):
        """Test health check."""
        # Set up provider
        provider = PostgresVectorProvider()
        provider.engine = mock_engine
        
        # Test health check
        result = provider.health_check()
        
        assert result is True
        mock_engine.connect.assert_called()
    
    def test_delete_embedding(self, mock_engine):
        """Test deleting embedding."""
        # Set up provider
        provider = PostgresVectorProvider()
        provider.engine = mock_engine
        provider.embeddings_table = MagicMock()
        
        # Mock delete method
        delete_mock = MagicMock()
        provider.embeddings_table.delete.return_value = delete_mock
        delete_mock.where.return_value = delete_mock
        
        # Set up connection to return success
        mock_conn = mock_engine.connect.return_value
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_conn.execute.return_value = mock_result
        
        # Test deleting embedding
        result = provider.delete_embedding("test1")
        
        assert result is True
        mock_conn.execute.assert_called_once()
        provider.embeddings_table.delete.assert_called_once()

@pytest.mark.skipif(skip_db_tests, reason=skip_reason)
class TestPostgresVectorProviderIntegration:
    """Integration tests for PostgresVectorProvider class."""
    
    def test_integration_connect(self):
        """Test connecting to the database."""
        provider = PostgresVectorProvider()
        result = provider.connect()
        
        assert result is True
        assert provider.engine is not None
        
        provider.disconnect()
    
    def test_integration_store_and_retrieve(self):
        """Test storing and retrieving embeddings."""
        provider = PostgresVectorProvider()
        provider.connect()
        
        # Generate random test vector
        test_vector = np.random.rand(768).tolist()
        
        # Store test vector
        result = provider.store_embedding(
            doc_id="integration_test_doc",
            text="This is an integration test document",
            embedding=test_vector,
            metadata={"source": "integration_test"}
        )
        
        assert result is True
        
        # Retrieve the vector
        results = provider.search_similar(test_vector, top_k=1)
        
        assert len(results) > 0
        assert results[0]["id"] == "integration_test_doc"
        assert results[0]["similarity"] > 0.9  # Should be very similar to itself
        
        # Clean up
        provider.delete_embedding("integration_test_doc")
        provider.disconnect()

def test_get_db_provider():
    """Test get_db_provider function."""
    # Test with postgres provider
    with patch.dict(os.environ, {"DB_PROVIDER": "postgres"}):
        provider = get_db_provider()
        assert isinstance(provider, PostgresVectorProvider)
    
    # Test with unknown provider (should default to postgres)
    with patch.dict(os.environ, {"DB_PROVIDER": "unknown"}):
        provider = get_db_provider()
        assert isinstance(provider, PostgresVectorProvider) 