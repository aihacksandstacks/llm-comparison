"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Tests for the web crawler module's public API and integration with app.py.
These tests focus on the interface rather than implementation details to catch
import and signature errors early.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import importlib
import inspect
import asyncio

# Import the module under test
from src.features.llm_compare.crawler import (
    WebCrawler, 
    get_web_crawler, 
    crawl_website,
    simple_http_crawl,
    simple_http_crawl_sync
)


class TestCrawlerAPI:
    """Tests for the crawler module's public API."""
    
    def test_all_exported_functions_exist(self):
        """Test that all expected public functions exist and are exported."""
        import src.features.llm_compare.crawler as crawler_module
        
        essential_functions = [
            "get_web_crawler",
            "crawl_website",
            "simple_http_crawl",
            "simple_http_crawl_sync"
        ]
        
        for func_name in essential_functions:
            assert hasattr(crawler_module, func_name), f"Missing essential function: {func_name}"
            assert callable(getattr(crawler_module, func_name)), f"{func_name} is not callable"
    
    def test_function_signatures(self):
        """Test that function signatures match expected parameters."""
        # Check crawl_website parameters
        sig = inspect.signature(crawl_website)
        params = sig.parameters
        
        assert "url" in params
        assert "depth" in params
        assert "max_pages" in params
        assert "output_name" in params
        
        # Check simple_http_crawl_sync parameters
        sig = inspect.signature(simple_http_crawl_sync)
        params = sig.parameters
        
        assert "url" in params
        assert len(params) == 1  # Only url parameter
        
        # Check async simple_http_crawl parameters
        sig = inspect.signature(simple_http_crawl)
        params = sig.parameters
        
        assert "url" in params
        assert len(params) == 1  # Only url parameter
    
    def test_return_types(self):
        """Test that functions return expected types when mocked."""
        # Mock the WebCrawler._simple_http_crawl method
        with patch.object(WebCrawler, "_simple_http_crawl") as mock_simple_crawl:
            mock_simple_crawl.return_value = [{"url": "https://example.com"}]
            
            # Test simple_http_crawl
            result = asyncio.run(simple_http_crawl("https://example.com"))
            assert isinstance(result, list)
            assert len(result) > 0
            assert "url" in result[0]
        
        # Mock the simple_http_crawl function
        with patch("src.features.llm_compare.crawler.simple_http_crawl") as mock_simple_crawl:
            mock_simple_crawl.return_value = [{"url": "https://example.com"}]
            
            # Test simple_http_crawl_sync
            result = simple_http_crawl_sync("https://example.com")
            assert isinstance(result, list)
            assert len(result) > 0
            assert "url" in result[0]
        
        # Mock the crawl_website internal implementation
        with patch("src.features.llm_compare.crawler.asyncio.run") as mock_run:
            mock_run.return_value = {
                "results": [{"url": "https://example.com"}],
                "metadata": {"url": "https://example.com"}
            }
            
            # Test crawl_website
            result = crawl_website("https://example.com")
            assert isinstance(result, dict)
            assert "results" in result
            assert "metadata" in result


class TestAppIntegration:
    """Tests that verify crawler integration with app.py."""
    
    def test_app_imports(self):
        """Test that app.py can import all required functions from crawler.py."""
        # Use importlib to simulate the imports in app.py
        app_module_spec = importlib.util.find_spec("src.features.base_ui.app")
        
        # If app module is not found, this test is not applicable
        if app_module_spec is None:
            pytest.skip("App module not found")
        
        # Verify all crawler imports needed by app.py
        from src.features.llm_compare.crawler import get_web_crawler, crawl_website, simple_http_crawl_sync
        
        # Assert all imported objects are callable
        assert callable(get_web_crawler)
        assert callable(crawl_website)
        assert callable(simple_http_crawl_sync)
    
    def test_simple_http_mode(self):
        """Test the simple_http_crawl_sync function used by the app's simple mode."""
        with patch("src.features.llm_compare.crawler.asyncio.run") as mock_run:
            # Mock the async result
            mock_run.return_value = [{
                "url": "https://example.com",
                "title": "Example Domain",
                "content": "This is example content",
                "metadata": {
                    "crawl_timestamp": 123456789,
                    "status_code": 200,
                    "method": "simple_http_fallback"
                }
            }]
            
            # Call the sync wrapper
            result = simple_http_crawl_sync("https://example.com")
            
            # Verify the result
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["url"] == "https://example.com"
            assert "content" in result[0]
            assert result[0]["metadata"]["method"] == "simple_http_fallback"
    
    def test_error_handling(self):
        """Test that error handling in crawler functions works as expected."""
        # Test simple_http_crawl_sync error handling
        with patch("src.features.llm_compare.crawler.asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Test exception")
            
            # This should not raise an exception
            result = simple_http_crawl_sync("https://example.com")
            
            # Verify the error result format
            assert isinstance(result, list)
            assert len(result) == 1
            assert "Error fetching page" in result[0]["content"]
            assert result[0]["metadata"]["error"] == "Test exception"
        
        # Test crawl_website error handling
        with patch("src.features.llm_compare.crawler.asyncio.run") as mock_run:
            mock_run.side_effect = Exception("Test exception")
            
            # This should not raise an exception
            result = crawl_website("https://example.com")
            
            # Verify the error result format
            assert isinstance(result, dict)
            assert "error" in result
            assert "Test exception" in result["error"]
            assert "traceback" in result


class TestEdgeCases:
    """Tests for edge cases that could cause issues in production."""
    
    def test_empty_url(self):
        """Test behavior with empty URLs."""
        # simple_http_crawl_sync should handle empty URLs gracefully
        result = simple_http_crawl_sync("")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]["metadata"]
    
    def test_imports_with_missing_crawler(self):
        """Test imports when crawler is not available."""
        with patch.dict(sys.modules, {'crawl4ai': None}):
            # Force reload of the module to simulate missing crawler
            with pytest.raises(ImportError):
                import crawl4ai
            
            # The crawler module should still be importable
            from src.features.llm_compare.crawler import get_web_crawler, crawl_website, simple_http_crawl_sync
            
            # And the functions should be callable
            assert callable(get_web_crawler)
            assert callable(crawl_website)
            assert callable(simple_http_crawl_sync)
    
    def test_import_cycle_prevention(self):
        """Test that circular imports are prevented."""
        from src.features.llm_compare.crawler import get_web_crawler
        
        # The factory function should not reference the module it's defined in
        # to prevent circular import issues
        with patch("src.features.llm_compare.crawler.WebCrawler") as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_crawler_class.return_value = mock_crawler
            
            # This should not cause circular import issues
            result = get_web_crawler()
            assert result == mock_crawler


# Integration test for the whole crawler pipeline
@pytest.mark.integration
def test_crawler_full_pipeline():
    """
    Test the full crawler pipeline from URL to processed results.
    This is a slower test that actually makes network requests.
    """
    url = "https://example.com"
    
    # Skip if running in CI environment or if network is not available
    if os.environ.get("CI") == "true" or os.environ.get("SKIP_NETWORK_TESTS") == "true":
        pytest.skip("Skipping network tests in CI environment")
    
    try:
        import httpx
        # Check if network is available by making a HEAD request
        response = httpx.head(url, timeout=2)
        if response.status_code >= 400:
            pytest.skip(f"Network test skipped - server returned {response.status_code}")
    except Exception as e:
        pytest.skip(f"Network test skipped due to error: {str(e)}")
    
    # First test the synchronous wrapper for simple_http_crawl
    simple_result = simple_http_crawl_sync(url)
    assert isinstance(simple_result, list)
    assert len(simple_result) > 0
    assert simple_result[0]["url"] == url
    assert simple_result[0]["content"] != ""
    
    # Then test the full crawl_website function with minimal parameters
    result = crawl_website(url, depth=1, max_pages=1)
    assert isinstance(result, dict)
    
    # The result could contain an error if crawl4ai isn't installed
    # but it should still have the expected structure
    if "error" in result:
        assert "metadata" in result
        assert result["metadata"]["url"] == url
    else:
        assert "results" in result
        assert "metadata" in result
        assert len(result["results"]) > 0
        assert result["results"][0]["url"] == url 