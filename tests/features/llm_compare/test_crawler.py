"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Tests for the web crawler module.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.features.llm_compare.crawler import WebCrawler, get_web_crawler


@pytest.fixture
def mock_crawled_data_dir(tmp_path):
    """Create a temporary directory for crawled data."""
    crawled_dir = tmp_path / "data" / "crawled"
    crawled_dir.mkdir(parents=True, exist_ok=True)
    return crawled_dir

@pytest.fixture
def mock_web_config():
    """Mock web crawling configuration for testing."""
    return {
        "max_depth": 2,
        "max_pages": 50,
        "timeout": 15,
        "user_agent": "Test-Agent/1.0",
        "respect_robots_txt": False
    }

@pytest.fixture
def mock_crawl_result():
    """Mock crawl result data for testing."""
    return [
        {
            "url": "https://example.com",
            "title": "Example Domain",
            "text": "This domain is for use in illustrative examples in documents.",
            "timestamp": "2023-06-15T12:34:56",
            "content_type": "text/html",
            "status_code": 200
        },
        {
            "url": "https://example.com/about",
            "title": "About Example",
            "text": "More information about this example domain.",
            "timestamp": "2023-06-15T12:35:00",
            "content_type": "text/html",
            "status_code": 200
        }
    ]


class TestWebCrawler:
    """Tests for the WebCrawler class."""
    
    @patch("src.features.llm_compare.crawler.get_config")
    @patch("src.features.llm_compare.crawler.CRAWL4AI_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.crawler.DATA_DIR")
    def test_init(self, mock_data_dir, mock_get_config, mock_web_config, mock_crawled_data_dir):
        """Test initialization of WebCrawler."""
        # Setup
        mock_data_dir.return_value = str(mock_crawled_data_dir.parent)
        mock_get_config.return_value = mock_web_config
        
        # Execute
        crawler = WebCrawler()
        
        # Assert
        mock_get_config.assert_called_once_with("web_crawling")
        assert crawler.max_depth == 2
        assert crawler.max_pages == 50
        assert crawler.timeout == 15
        assert crawler.user_agent == "Test-Agent/1.0"
        assert crawler.respect_robots_txt is False
        assert crawler.api_key == "test_api_key"
    
    @patch("src.features.llm_compare.crawler.get_config")
    @patch("src.features.llm_compare.crawler.CRAWL4AI_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.crawler.DATA_DIR")
    def test_process_crawl_results(self, mock_data_dir, mock_get_config, 
                                 mock_web_config, mock_crawl_result):
        """Test processing of crawl results."""
        # Setup
        mock_data_dir.return_value = "data"
        mock_get_config.return_value = mock_web_config
        
        # Execute
        crawler = WebCrawler()
        processed_results = crawler._process_crawl_results(mock_crawl_result)
        
        # Assert
        assert len(processed_results) == 2
        assert processed_results[0]["url"] == "https://example.com"
        assert processed_results[0]["title"] == "Example Domain"
        assert processed_results[0]["content"] == "This domain is for use in illustrative examples in documents."
        assert processed_results[0]["metadata"]["status_code"] == 200
        
        assert processed_results[1]["url"] == "https://example.com/about"
        assert processed_results[1]["title"] == "About Example"
    
    @patch("src.features.llm_compare.crawler.get_config")
    @patch("src.features.llm_compare.crawler.CRAWL4AI_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.crawler.DATA_DIR")
    @patch("builtins.open", new_callable=mock_open, read_data='[{"url": "https://example.com"}]')
    def test_load_processed_crawl(self, mock_file, mock_data_dir, mock_get_config, 
                                mock_web_config, mock_crawled_data_dir):
        """Test loading processed crawl results."""
        # Setup
        mock_data_dir.return_value = str(mock_crawled_data_dir.parent)
        mock_get_config.return_value = mock_web_config
        
        # Create a mock processed file
        mock_processed_path = mock_crawled_data_dir / "example_processed.json"
        
        # Execute
        with patch.object(Path, "exists", return_value=True):
            crawler = WebCrawler()
            result = crawler.load_processed_crawl("example")
        
        # Assert
        mock_file.assert_called_once()
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com"
    
    @patch("src.features.llm_compare.crawler.get_config")
    @patch("src.features.llm_compare.crawler.CRAWL4AI_API_KEY", "test_api_key")
    @patch("src.features.llm_compare.crawler.DATA_DIR")
    @patch("crawl4ai.Client")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    def test_crawl_website(self, mock_time, mock_json_dump, mock_file, mock_client, 
                         mock_data_dir, mock_get_config, mock_web_config, 
                         mock_crawl_result, mock_crawled_data_dir):
        """Test crawling a website."""
        # Setup
        mock_data_dir.return_value = str(mock_crawled_data_dir.parent)
        mock_get_config.return_value = mock_web_config
        
        # Mock time.time to return consistent values
        mock_time.side_effect = [100.0, 105.0]  # Start time, end time
        
        # Mock the crawl4ai Client
        mock_client_instance = MagicMock()
        mock_job = MagicMock()
        mock_job.wait_until_complete.return_value = None
        mock_job.get_results.return_value = mock_crawl_result
        mock_client_instance.crawl.return_value = mock_job
        mock_client.return_value = mock_client_instance
        
        # Execute
        with patch("src.features.llm_compare.crawler.crawl4ai", create=True):
            crawler = WebCrawler()
            result = crawler.crawl_website("https://example.com", depth=2, max_pages=50)
        
        # Assert
        mock_client.assert_called_once_with(api_key="test_api_key")
        mock_client_instance.crawl.assert_called_once_with(
            urls=["https://example.com"],
            depth=2,
            max_pages=50,
            user_agent="Test-Agent/1.0",
            respect_robots=False
        )
        mock_job.wait_until_complete.assert_called_once()
        mock_job.get_results.assert_called_once()
        
        # Check that files were written
        assert mock_file.call_count == 2
        
        # Check the returned data
        assert "results" in result
        assert "metadata" in result
        assert len(result["results"]) == 2
        assert result["metadata"]["crawl_time"] == 5.0  # 105 - 100
        assert result["metadata"]["pages_crawled"] == 2

    @patch("src.features.llm_compare.crawler.get_web_crawler")
    def test_get_web_crawler(self, mock_get_web_crawler):
        """Test the get_web_crawler factory function."""
        # Setup
        mock_crawler = MagicMock()
        mock_get_web_crawler.return_value = mock_crawler
        
        # Execute
        result = get_web_crawler()
        
        # Assert
        assert result == mock_crawler 