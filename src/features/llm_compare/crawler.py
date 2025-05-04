"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Web crawling module for the LLM Comparison Tool.
Provides functionality for crawling websites and processing content for RAG.
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import time

from src.shared.config import DATA_DIR, CRAWL4AI_API_KEY, get_config

class WebCrawler:
    """Class for crawling websites and processing content for RAG."""
    
    def __init__(self):
        """Initialize the web crawler."""
        # Get configuration
        web_config = get_config("web_crawling")
        self.max_depth = web_config.get("max_depth", 3)
        self.max_pages = web_config.get("max_pages", 100)
        self.timeout = web_config.get("timeout", 30)
        self.user_agent = web_config.get("user_agent", "LLM-Comparison-Tool/0.1.0")
        self.respect_robots_txt = web_config.get("respect_robots_txt", True)
        
        # Set up API key for Crawl4AI
        self.api_key = CRAWL4AI_API_KEY
        
        if not self.api_key:
            print("Warning: Crawl4AI API key not found. Some functionality may be limited.")
        
        # Create data directory for crawled content
        self.crawled_data_dir = Path(DATA_DIR) / "crawled"
        self.crawled_data_dir.mkdir(parents=True, exist_ok=True)
    
    async def crawl_website(self, url: str, depth: Optional[int] = None, 
                     max_pages: Optional[int] = None, 
                     output_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Crawl a website and store the content.
        
        Args:
            url: URL to crawl.
            depth: Maximum depth to crawl (default from config).
            max_pages: Maximum pages to crawl (default from config).
            output_name: Name for the output files (defaults to domain name).
            
        Returns:
            Dictionary with crawl results and metadata.
        """
        depth = depth or self.max_depth
        max_pages = max_pages or self.max_pages
        
        # Import Crawl4AI here to avoid dependency issues if not using this feature
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        except ImportError:
            raise ImportError("Crawl4AI package not installed. Please install it with 'pip install crawl4ai'.")
        
        # Extract domain name for default output name
        if not output_name:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            output_name = domain.replace(".", "_")
        
        # Start the crawl job
        print(f"Starting crawl of {url} with depth {depth}, max pages {max_pages}")
        start_time = time.time()
        
        browser_config = BrowserConfig(
            headless=True,
            user_agent=self.user_agent
        )
        
        run_config = CrawlerRunConfig(
            respect_robots_txt=self.respect_robots_txt,
            wait_for_timeout=self.timeout * 1000  # Convert to milliseconds
        )
        
        try:
            # Use AsyncWebCrawler with modern API
            async with AsyncWebCrawler(config=browser_config) as crawler:
                if depth > 1:
                    # Use deep crawling for multiple pages
                    print(f"Performing deep crawl with depth {depth}, max pages {max_pages}")
                    crawl_results = []
                    
                    # Use deep_crawl method if available in this version
                    crawl_generator = await crawler.adeep_crawl(
                        start_url=url,
                        strategy="bfs",  # Breadth-first search
                        max_depth=depth,
                        max_pages=max_pages,
                        config=run_config
                    )
                    
                    # Process results as they come in
                    async for result in crawl_generator:
                        if result.success:
                            crawl_results.append(result)
                    
                    # Process the results
                    processed_results = []
                    for result in crawl_results:
                        processed_page = self._process_crawl_result(result)
                        processed_results.append(processed_page)
                        
                else:
                    # Single page crawl
                    result = await crawler.arun(url=url, config=run_config)
                    
                    if not result.success:
                        raise Exception(f"Crawl failed with error: {result.error_message}")
                    
                    # Process the single result
                    processed_results = [self._process_crawl_result(result)]
            
            # Save raw crawl results
            output_path = self.crawled_data_dir / f"{output_name}_raw.json"
            with open(output_path, "w") as f:
                # Need to convert CrawlResult objects to JSON-serializable dict
                serializable_results = [self._make_serializable(result) for result in processed_results]
                json.dump(serializable_results, f, indent=2)
            
            # Save processed results
            processed_path = self.crawled_data_dir / f"{output_name}_processed.json"
            with open(processed_path, "w") as f:
                json.dump(processed_results, f, indent=2)
            
            end_time = time.time()
            crawl_time = end_time - start_time
            
            # Generate metadata about the crawl
            metadata = {
                "url": url,
                "depth": depth,
                "max_pages": max_pages,
                "pages_crawled": len(processed_results),
                "crawl_time": crawl_time,
                "output_name": output_name,
                "raw_path": str(output_path),
                "processed_path": str(processed_path)
            }
            
            print(f"Crawl completed: {len(processed_results)} pages crawled in {crawl_time:.2f} seconds")
            
            return {
                "results": processed_results,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error during crawl: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "url": url,
                    "depth": depth,
                    "max_pages": max_pages
                }
            }
    
    def _make_serializable(self, result: Any) -> Dict[str, Any]:
        """Convert a CrawlResult to a JSON-serializable dictionary."""
        if hasattr(result, "__dict__"):
            return {k: v for k, v in result.__dict__.items() 
                   if not k.startswith("_") and not callable(v)}
        return result
    
    def _process_crawl_result(self, result: Any) -> Dict[str, Any]:
        """
        Process a single crawl result into a format suitable for RAG.
        
        Args:
            result: A CrawlResult from Crawl4AI.
            
        Returns:
            Processed page suitable for RAG.
        """
        # Extract content from the result
        content = ""
        title = ""
        
        if hasattr(result, "markdown") and result.markdown:
            # Prefer fit_markdown if available
            if hasattr(result.markdown, "fit_markdown") and result.markdown.fit_markdown:
                content = result.markdown.fit_markdown
            elif hasattr(result.markdown, "raw_markdown") and result.markdown.raw_markdown:
                content = result.markdown.raw_markdown
            # For older versions that might return markdown directly as a string
            elif isinstance(result.markdown, str):
                content = result.markdown
        
        # Try to get the title
        if hasattr(result, "metadata") and result.metadata:
            if hasattr(result.metadata, "title"):
                title = result.metadata.title
        
        # Create a structured document for the page
        processed_page = {
            "url": result.url if hasattr(result, "url") else "",
            "title": title,
            "content": content,
            "metadata": {
                "crawl_timestamp": time.time(),
                "status_code": result.status_code if hasattr(result, "status_code") else 0
            }
        }
        
        return processed_page
    
    def load_processed_crawl(self, output_name: str) -> List[Dict[str, Any]]:
        """
        Load previously processed crawl results.
        
        Args:
            output_name: Name of the output file.
            
        Returns:
            List of processed pages from the crawl.
        """
        processed_path = self.crawled_data_dir / f"{output_name}_processed.json"
        
        if not processed_path.exists():
            print(f"Processed crawl file not found: {processed_path}")
            return []
        
        try:
            with open(processed_path, "r") as f:
                processed_data = json.load(f)
            
            print(f"Loaded {len(processed_data)} pages from {processed_path}")
            return processed_data
        except Exception as e:
            print(f"Error loading processed crawl file {processed_path}: {e}")
            return []

def get_web_crawler() -> WebCrawler:
    """
    Factory function to get a WebCrawler instance.
    
    Returns:
        WebCrawler instance.
    """
    return WebCrawler()

# Helper function for synchronous interface
def crawl_website(url: str, depth: Optional[int] = None, 
                max_pages: Optional[int] = None, 
                output_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Crawl a website synchronously.
    
    Args:
        url: URL to crawl.
        depth: Maximum depth to crawl.
        max_pages: Maximum pages to crawl.
        output_name: Name for the output files.
        
    Returns:
        Dictionary with crawl results and metadata.
    """
    crawler = WebCrawler()
    
    # Configure event loop for Windows compatibility
    try:
        if os.name == 'nt':  # Windows
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass  # Ignore if this fails
        
    # Run crawl asynchronously but provide a synchronous interface
    return asyncio.run(crawler.crawl_website(url, depth, max_pages, output_name)) 