"""
Web crawling module for the LLM Comparison Tool.
Provides functionality for crawling websites and processing content for RAG.
"""

import os
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
    
    def crawl_website(self, url: str, depth: Optional[int] = None, 
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
            import crawl4ai
        except ImportError:
            raise ImportError("Crawl4AI package not installed. Please install it with 'pip install crawl4ai'.")
        
        # Set up the crawler client
        client = crawl4ai.Client(api_key=self.api_key)
        
        # Extract domain name for default output name
        if not output_name:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            output_name = domain.replace(".", "_")
        
        # Start the crawl job
        print(f"Starting crawl of {url} with depth {depth}, max pages {max_pages}")
        start_time = time.time()
        
        crawl_config = {
            "url": url,
            "depth": depth,
            "max_pages": max_pages,
            "user_agent": self.user_agent,
            "respect_robots_txt": self.respect_robots_txt,
            "timeout": self.timeout
        }
        
        try:
            # Start the crawl job
            job = client.crawl(
                urls=[url],
                depth=depth,
                max_pages=max_pages,
                user_agent=self.user_agent,
                respect_robots=self.respect_robots_txt
            )
            
            # Wait for the job to complete
            job.wait_until_complete()
            
            # Get the results
            results = job.get_results()
            
            # Save raw crawl results
            output_path = self.crawled_data_dir / f"{output_name}_raw.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            
            # Process the results into a format suitable for RAG
            processed_results = self._process_crawl_results(results)
            
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
                "pages_crawled": len(results),
                "crawl_time": crawl_time,
                "output_name": output_name,
                "raw_path": str(output_path),
                "processed_path": str(processed_path)
            }
            
            print(f"Crawl completed: {len(results)} pages crawled in {crawl_time:.2f} seconds")
            
            return {
                "results": processed_results,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error during crawl of {url}: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "url": url,
                    "depth": depth,
                    "max_pages": max_pages
                }
            }
    
    def _process_crawl_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw crawl results into a format suitable for RAG.
        
        Args:
            results: Raw crawl results from Crawl4AI.
            
        Returns:
            Processed results suitable for RAG.
        """
        processed_data = []
        
        for page in results:
            # Extract main content from HTML if available
            content = page.get("text", "")
            
            # Create a structured document for each page
            processed_page = {
                "url": page.get("url", ""),
                "title": page.get("title", ""),
                "content": content,
                "metadata": {
                    "crawl_timestamp": page.get("timestamp", ""),
                    "content_type": page.get("content_type", ""),
                    "status_code": page.get("status_code", 0)
                }
            }
            
            processed_data.append(processed_page)
        
        return processed_data
    
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