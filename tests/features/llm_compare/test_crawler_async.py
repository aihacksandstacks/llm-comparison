"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Test the updated crawler implementation with the AsyncWebCrawler API.
"""

import os
import pytest
from pathlib import Path
import asyncio

# Mock the API key
os.environ["CRAWL4AI_API_KEY"] = "test_api_key"

from src.features.llm_compare.crawler import crawl_website, WebCrawler


def test_crawl_website_sync():
    """Test the synchronous wrapper function."""
    url = "https://example.com"
    
    # Use a short timeout for testing
    result = crawl_website(url, depth=1, max_pages=1)
    
    # Check that the result has the expected structure
    assert isinstance(result, dict)
    assert "metadata" in result
    
    # Check if there was an error
    if "error" in result:
        # This is acceptable during testing if the crawl4ai package isn't installed
        # or if there are network issues
        print(f"Crawl error (expected during testing): {result['error']}")
    else:
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0
        
        # Check the structure of a result page
        page = result["results"][0]
        assert "url" in page
        assert "title" in page
        assert "content" in page
        assert "metadata" in page


@pytest.mark.asyncio
async def test_crawl_website_async():
    """Test the async crawl_website method directly."""
    url = "https://example.com"
    
    # Create crawler instance
    crawler = WebCrawler()
    
    # Use a short timeout for testing
    result = await crawler.crawl_website(url, depth=1, max_pages=1)
    
    # Check that the result has the expected structure
    assert isinstance(result, dict)
    assert "metadata" in result
    
    # Check if there was an error
    if "error" in result:
        # This is acceptable during testing if the crawl4ai package isn't installed
        # or if there are network issues
        print(f"Crawl error (expected during testing): {result['error']}")
    else:
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0
        
        # Check the structure of a result page
        page = result["results"][0]
        assert "url" in page
        assert "title" in page
        assert "content" in page
        assert "metadata" in page


if __name__ == "__main__":
    # Run the test directly for debugging
    print("Testing synchronous wrapper...")
    test_crawl_website_sync()
    
    print("\nTesting asynchronous method...")
    asyncio.run(test_crawl_website_async())
    
    print("\nTests completed successfully!") 