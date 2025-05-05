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
    # Create a mock that mimics the properties of a Crawl4AI result
    mock_result = MagicMock()
    mock_result.url = "https://example.com"
    mock_result.title = "Example Domain"
    mock_result.markdown = MagicMock()
    mock_result.markdown.fit_markdown = "# Example Domain\n\nThis domain is for illustrative examples in documents."
    mock_result.html = "<html><body><h1>Example Domain</h1></body></html>"
    mock_result.text = "Example Domain This domain is for illustrative examples in documents."
    mock_result.metadata = MagicMock()
    mock_result.metadata.title = "Example Domain"
    mock_result.status_code = 200
    mock_result.success = True
    
    return mock_result


class TestWebCrawler:
    """Tests for the WebCrawler class."""
    
    @patch("src.features.llm_compare.crawler.get_config")
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
        assert hasattr(crawler, 'crawl4ai_available')
    
    @patch("src.features.llm_compare.crawler.get_config")
    @patch("src.features.llm_compare.crawler.DATA_DIR")
    def test_process_crawl4ai_result(self, mock_data_dir, mock_get_config, 
                                 mock_web_config, mock_crawl_result):
        """Test processing of a single Crawl4AI result."""
        # Setup
        mock_data_dir.return_value = "data"
        mock_get_config.return_value = mock_web_config
        
        # Execute
        crawler = WebCrawler()
        processed_result = crawler._process_crawl4ai_result(mock_crawl_result)
        
        # Assert
        assert processed_result["url"] == "https://example.com"
        assert processed_result["title"] == "Example Domain"
        assert "# Example Domain" in processed_result["content"]
        assert processed_result["metadata"]["status_code"] == 200
        assert processed_result["metadata"]["content_type"] == "markdown"
    
    @patch("src.features.llm_compare.crawler.get_config")
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
    @patch("src.features.llm_compare.crawler.DATA_DIR")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    @patch("crawl4ai.AsyncWebCrawler")
    @patch("asyncio.run")
    def test_crawl_website(self, mock_asyncio_run, mock_crawler_class, mock_time, mock_json_dump, 
                         mock_file, mock_data_dir, mock_get_config, mock_web_config, 
                         mock_crawl_result, mock_crawled_data_dir):
        """Test crawling a website with AsyncWebCrawler."""
        # Setup
        mock_data_dir.return_value = str(mock_crawled_data_dir.parent)
        mock_get_config.return_value = mock_web_config
        
        # Mock time.time to return consistent values
        mock_time.side_effect = [100.0, 105.0]  # Start time, end time
        
        # Create a mock for the async context manager
        mock_crawler = MagicMock()
        mock_crawler.__aenter__.return_value = mock_crawler
        mock_crawler.__aexit__.return_value = None
        
        # Set up the mock for AsyncWebCrawler
        mock_crawler_class.return_value = mock_crawler
        
        # Mock the arun method to return our mock result
        mock_crawler.arun.return_value = mock_crawl_result
        
        # Mock asyncio.run to simulate executing the async function
        async def simulate_crawl_website(*args, **kwargs):
            return {
                "results": [crawler._process_crawl4ai_result(mock_crawl_result)],
                "metadata": {
                    "url": "https://example.com",
                    "depth": 2,
                    "max_pages": 50,
                    "pages_crawled": 1,
                    "crawl_time": 5.0,
                    "output_name": "example_com",
                    "method": "crawl4ai"
                }
            }
        
        mock_asyncio_run.side_effect = simulate_crawl_website
        
        # Execute
        with patch("src.features.llm_compare.crawler.crawl4ai", create=True):
            crawler = WebCrawler()
            crawler.crawl4ai_available = True  # Force crawler to use Crawl4AI
            result = crawler.crawl_website("https://example.com", depth=1, max_pages=50)
        
        # Assert
        # Check that files were written
        assert mock_file.call_count >= 1
        
        # Check the returned data
        assert "results" in result
        assert "metadata" in result
        assert result["metadata"]["url"] == "https://example.com"

    @patch("src.features.llm_compare.crawler.WebCrawler")
    def test_get_web_crawler(self, mock_crawler_class):
        """Test the get_web_crawler factory function."""
        # Setup
        mock_crawler = MagicMock()
        mock_crawler_class.return_value = mock_crawler
        
        # Execute
        result = get_web_crawler()
        
        # Assert
        assert result == mock_crawler 