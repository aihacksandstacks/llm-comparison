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
import traceback
import logging  # Add logging module

from src.shared.config import DATA_DIR, get_config

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level

# Create console handler for more detailed output
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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
        
        # Create data directory for crawled content
        self.crawled_data_dir = Path(DATA_DIR) / "crawled"
        self.crawled_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if crawl4ai is installed
        try:
            # Test importing crawl4ai at initialization time to provide better feedback
            import crawl4ai
            self.crawl4ai_available = True
            logger.info("Crawl4AI is available.")
        except ImportError as e:
            self.crawl4ai_available = False
            logger.warning(f"Warning: Crawl4AI is not installed: {e}")
            logger.warning("Using fallback HTTP crawling method.")
            logger.warning("To enable full crawling capabilities, install crawl4ai with: pip install crawl4ai")
    
    def warmup(self):
        """Optional warm-up step to initialize the browser for faster first-time crawling."""
        if self.crawl4ai_available:
            try:
                from crawl4ai import WebCrawler as Crawl4AIWebCrawler
                # Initialize the crawler but don't actually crawl anything
                crawler = Crawl4AIWebCrawler()
                crawler.warmup()
                logger.info("Browser warmed up successfully")
            except Exception as e:
                logger.warning(f"Warning: Failed to warm up browser: {e}")
                logger.debug(traceback.format_exc())
    
    async def crawl_website(self, url: str, depth: Optional[int] = 1, 
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
        
        # Extract domain name for default output name
        if not output_name:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            output_name = domain.replace(".", "_")
        
        # Start the crawl job
        logger.info(f"Starting crawl of {url} with depth {depth}, max pages {max_pages}")
        start_time = time.time()
        
        try:
            # Skip Crawl4AI import attempt if we already know it's not available
            if not self.crawl4ai_available:
                logger.info("Crawl4AI is not available. Using fallback HTTP crawling method.")
                processed_results = await self._simple_http_crawl(url)
            else:
                try:
                    logger.debug("Attempting to import crawl4ai modules...")
                    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
                    from crawl4ai.content_filter_strategy import PruningContentFilter
                    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

                    # Create configs for crawl4ai
                    logger.debug("Creating browser config...")
                    browser_config = BrowserConfig(
                        headless=True,
                        user_agent=self.user_agent,
                        verbose=True
                    )
                    
                    # Configure the markdown generator with pruning filter for better content quality
                    logger.debug("Setting up content filter and markdown generator...")
                    pruning_filter = PruningContentFilter(
                        threshold=0.5,  # Balance between keeping important content and removing noise
                        threshold_type="fixed",
                        min_word_threshold=10  # Minimum word count to consider a block
                    )
                    
                    md_generator = DefaultMarkdownGenerator(content_filter=pruning_filter)
                    
                    # Create crawler run configuration with compatible parameters
                    # Note: The API might have changed, so we're using the most compatible parameters
                    logger.debug("Creating crawler run config...")
                    run_config = CrawlerRunConfig(
                        cache_mode=CacheMode.ENABLED,  # Enable caching for better performance
                        markdown_generator=md_generator,
                        word_count_threshold=10,  # Minimum words per content block
                        exclude_external_links=True,  # Only include internal links in markdown
                        remove_overlay_elements=True,  # Remove popups/modals that might interfere
                        process_iframes=True,  # Process content inside iframes
                        wait_until="networkidle",  # Wait for network to be idle (helps with dynamic content)
                        delay_before_return_html=0.5  # Wait a bit after page load
                    )
                    
                    logger.info("Starting crawl with AsyncWebCrawler...")
                    # Log all config parameters
                    logger.debug(f"BrowserConfig: {browser_config.__dict__ if hasattr(browser_config, '__dict__') else browser_config}")
                    logger.debug(f"CrawlerRunConfig: {run_config.__dict__ if hasattr(run_config, '__dict__') else run_config}")
                    
                    if depth <= 1:
                        # For simple single-page crawl
                        logger.debug(f"Single page crawl for {url}")
                        async with AsyncWebCrawler(config=browser_config) as crawler:
                            result = await crawler.arun(url=url, config=run_config)
                            
                            logger.debug(f"Crawl result type: {type(result)}")
                            logger.debug(f"Crawl result attributes: {dir(result) if hasattr(result, '__dir__') else 'No attributes'}")
                            
                            if hasattr(result, "success") and not result.success:
                                error_msg = getattr(result, 'error_message', 'Unknown error')
                                logger.error(f"Crawl failed: {error_msg}")
                                raise Exception(f"Crawl failed: {error_msg}")
                            
                            processed_results = [self._process_crawl4ai_result(result)]
                    else:
                        # For deep crawling, we need to implement our own crawling logic
                        # since the deep_crawl method isn't supported in the current API
                        logger.debug(f"Deep crawl for {url} with depth {depth}")
                        
                        # We'll implement a simple BFS crawling algorithm
                        processed_results = []
                        visited_urls = set()
                        to_visit = [(url, 1)]  # (url, current_depth)
                        
                        # Store base domain for filtering internal links
                        from urllib.parse import urlparse
                        base_domain = urlparse(url).netloc
                        
                        async with AsyncWebCrawler(config=browser_config) as crawler:
                            while to_visit and len(processed_results) < max_pages:
                                current_url, current_depth = to_visit.pop(0)
                                
                                if current_url in visited_urls:
                                    continue
                                
                                visited_urls.add(current_url)
                                logger.debug(f"Crawling {current_url} at depth {current_depth}")
                                
                                try:
                                    result = await crawler.arun(url=current_url, config=run_config)
                                    
                                    if hasattr(result, "success") and result.success:
                                        processed_page = self._process_crawl4ai_result(result)
                                        processed_results.append(processed_page)
                                        logger.debug(f"Successfully crawled {current_url}")
                                        
                                        # If we're not at max depth yet, add links to the queue
                                        if current_depth < depth:
                                            # Extract links from the result
                                            links = []
                                            
                                            # Handle links based on their format
                                            if hasattr(result, "links"):
                                                if isinstance(result.links, dict):
                                                    # Handle dictionary format links (common in Crawl4AI)
                                                    for link_type, link_list in result.links.items():
                                                        if isinstance(link_list, list):
                                                            links.extend(link_list)
                                                        else:
                                                            links.append(link_list)
                                                elif isinstance(result.links, list):
                                                    # Handle list format links
                                                    links = result.links
                                            
                                            from urllib.parse import urljoin, urlparse
                                            
                                            # Process extracted links
                                            logger.debug(f"Processing {len(links)} links from {current_url}")
                                            found_valid_links = 0
                                            
                                            for link in links:
                                                link_url = None
                                                
                                                # Handle different link formats
                                                if isinstance(link, str):
                                                    link_url = link
                                                elif isinstance(link, dict) and 'href' in link:
                                                    # Handle dict with href field (common format in Crawl4AI)
                                                    link_url = link['href']
                                                elif hasattr(link, "url"):
                                                    link_url = link.url
                                                elif hasattr(link, "href"):
                                                    link_url = link.href
                                                
                                                if not link_url:
                                                    continue
                                                
                                                # Ensure it's an absolute URL
                                                if not link_url.startswith(('http://', 'https://')):
                                                    link_url = urljoin(current_url, link_url)
                                                
                                                # Check if it's an internal link (same domain)
                                                link_domain = urlparse(link_url).netloc
                                                if link_domain == base_domain and link_url not in visited_urls:
                                                    to_visit.append((link_url, current_depth + 1))
                                                    found_valid_links += 1
                                            
                                            logger.debug(f"Found {found_valid_links} valid new links to visit")
                                    else:
                                        error_msg = getattr(result, 'error_message', 'Unknown error')
                                        logger.warning(f"Failed to crawl {current_url}: {error_msg}")
                                except Exception as e:
                                    logger.warning(f"Error crawling {current_url}: {e}")
                            
                            logger.info(f"Deep crawl completed. Found {len(processed_results)} pages.")
                
                except Exception as e:
                    logger.error(f"Error with AsyncWebCrawler: {e}")
                    logger.debug(traceback.format_exc())
                    # Fall back to simple HTTP request if crawl4ai fails
                    logger.info("Falling back to simple HTTP crawling method...")
                    processed_results = await self._simple_http_crawl(url)
            
            # Save raw crawl results
            output_path = self.crawled_data_dir / f"{output_name}_raw.json"
            logger.debug(f"Saving raw results to {output_path}")
            with open(output_path, "w") as f:
                # Need to convert CrawlResult objects to JSON-serializable dict
                serializable_results = [self._make_serializable(result) for result in processed_results]
                json.dump(serializable_results, f, indent=2)
            
            # Save processed results
            processed_path = self.crawled_data_dir / f"{output_name}_processed.json"
            logger.debug(f"Saving processed results to {processed_path}")
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
                "processed_path": str(processed_path),
                "method": "crawl4ai" if self.crawl4ai_available else "simple_http"
            }
            
            logger.info(f"Crawl completed in {crawl_time:.2f}s with {len(processed_results)} pages")
            
            # Extra validation to ensure the structure is valid and expected for UI
            result = {
                "results": processed_results,
                "metadata": metadata
            }
            
            # Validate the structure (add a check in case of serialization problems)
            try:
                # Test if we can serialize to JSON and back
                json_str = json.dumps(result)
                json.loads(json_str)
                logger.debug("Result validated - can be serialized to JSON")
            except Exception as e:
                logger.error(f"Result validation failed, cannot be serialized to JSON: {e}")
                # Add error information to be caught by the UI
                return {
                    "error": f"Result validation failed: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "metadata": metadata
                }
            
            # Return results and metadata
            return result
        
        except Exception as e:
            logger.error(f"Error during crawl: {e}")
            logger.debug(traceback.format_exc())
            
            # Return error information
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metadata": {
                    "url": url,
                    "depth": depth,
                    "max_pages": max_pages,
                    "attempt_time": time.time() - start_time,
                    "output_name": output_name
                }
            }
    
    async def _simple_http_crawl(self, url: str) -> List[Dict[str, Any]]:
        """Simple fallback method that uses httpx to crawl a single page."""
        try:
            logger.info(f"Using simple HTTP fallback for {url}")
            import httpx
            
            async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout) as client:
                logger.debug(f"Making HTTP request to {url}")
                response = await client.get(url, headers={"User-Agent": self.user_agent})
                response.raise_for_status()
                
                html_content = response.text
                
                # Try to extract title
                title = ""
                try:
                    import re
                    title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
                    if title_match:
                        title = title_match.group(1).strip()
                except Exception as e:
                    logger.warning(f"Error extracting title: {e}")
                
                # Create a simple result
                result = {
                    "url": url,
                    "title": title,
                    "content": html_content,
                    "metadata": {
                        "crawl_timestamp": time.time(),
                        "status_code": response.status_code,
                        "method": "simple_http_fallback"
                    }
                }
                
                logger.debug(f"HTTP crawl successful for {url} (status {response.status_code})")
                return [result]
        except Exception as e:
            logger.error(f"Error in simple HTTP crawl: {e}")
            logger.debug(traceback.format_exc())
            # Return a minimal result with the error
            return [{
                "url": url,
                "title": "Error fetching page",
                "content": f"Error fetching page: {str(e)}",
                "metadata": {
                    "crawl_timestamp": time.time(),
                    "status_code": 0,
                    "error": str(e),
                    "method": "simple_http_fallback"
                }
            }]
    
    def _make_serializable(self, result: Any) -> Dict[str, Any]:
        """Convert a CrawlResult to a JSON-serializable dictionary."""
        logger.debug(f"Making serializable: {type(result)}")
        if hasattr(result, "__dict__"):
            return {k: v for k, v in result.__dict__.items() 
                   if not k.startswith("_") and not callable(v)}
        return result
    
    def _process_crawl4ai_result(self, result: Any) -> Dict[str, Any]:
        """
        Process a single crawl result from Crawl4AI into a format suitable for RAG.
        
        Args:
            result: A CrawlResult from Crawl4AI.
            
        Returns:
            Processed page suitable for RAG.
        """
        try:
            url = getattr(result, "url", "unknown")
            logger.debug(f"Processing crawl result for URL: {url}")
            
            # Get content from markdown if available (preferred format for RAG)
            content = ""
            if hasattr(result, "markdown"):
                logger.debug("Markdown content available")
                if hasattr(result.markdown, "fit_markdown") and result.markdown.fit_markdown:
                    content = result.markdown.fit_markdown
                    logger.debug("Using fit_markdown")
                elif hasattr(result.markdown, "raw_markdown") and result.markdown.raw_markdown:
                    content = result.markdown.raw_markdown
                    logger.debug("Using raw_markdown")
                elif isinstance(result.markdown, str):
                    content = result.markdown
                    logger.debug("Using markdown string directly")
            
            # Fallback to HTML or text if markdown not available
            if not content and hasattr(result, "html") and result.html:
                # Use just a summary of HTML for logging purposes
                logger.debug(f"Using HTML content for {url} (markdown not available)")
                content = result.html
            elif not content and hasattr(result, "text") and result.text:
                logger.debug(f"Using text content for {url} (markdown not available)")
                content = result.text
            
            # Get title
            title = ""
            if hasattr(result, "metadata") and result.metadata:
                if hasattr(result.metadata, "title"):
                    title = result.metadata.title
                    logger.debug(f"Got title from metadata: {title}")
            elif hasattr(result, "title"):
                title = result.title
                logger.debug(f"Got title directly: {title}")
            
            # Get status code
            status_code = 0
            if hasattr(result, "status_code"):
                status_code = result.status_code
                logger.debug(f"Status code: {status_code}")
            
            # Create structured document
            processed_page = {
                "url": url,
                "title": title,
                "content": content,
                "metadata": {
                    "crawl_timestamp": time.time(),
                    "status_code": status_code,
                    "content_type": "markdown" if hasattr(result, "markdown") else "html"
                }
            }
            
            logger.info(f"Processed page for URL {url} with {len(content)} chars of content")
            return processed_page
        except Exception as e:
            logger.error(f"Error processing crawl result: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return a minimal processed page rather than failing
            return {
                "url": getattr(result, "url", "unknown"),
                "title": "Error processing result",
                "content": f"Error: {str(e)}",
                "metadata": {
                    "crawl_timestamp": time.time(),
                    "status_code": 0,
                    "error": str(e)
                }
            }
    
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
            logger.warning(f"Processed crawl file not found: {processed_path}")
            return []
        
        try:
            with open(processed_path, "r") as f:
                processed_data = json.load(f)
            
            logger.info(f"Loaded {len(processed_data)} pages from {processed_path}")
            return processed_data
        except Exception as e:
            logger.error(f"Error loading processed crawl file {processed_path}: {e}")
            return []

def get_web_crawler() -> WebCrawler:
    """
    Factory function to get a WebCrawler instance.
    
    Returns:
        WebCrawler instance.
    """
    return WebCrawler()

# Simple HTTP crawl function for external use
async def simple_http_crawl(url: str) -> List[Dict[str, Any]]:
    """
    Simple HTTP crawler that can be called directly.
    
    Args:
        url: URL to crawl.
        
    Returns:
        List with the crawled page result.
    """
    crawler = WebCrawler()
    return await crawler._simple_http_crawl(url)

# Synchronous version of simple HTTP crawl
def simple_http_crawl_sync(url: str) -> List[Dict[str, Any]]:
    """
    Simple HTTP crawler that can be called synchronously.
    
    Args:
        url: URL to crawl.
        
    Returns:
        List with the crawled page result.
    """
    # Configure event loop for Windows compatibility
    try:
        if os.name == 'nt':  # Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        logger.warning(f"Warning: Could not set event loop policy: {e}")
    
    # Run crawl asynchronously but provide a synchronous interface
    try:
        return asyncio.run(simple_http_crawl(url))
    except Exception as e:
        logger.error(f"Error during simple HTTP crawl: {str(e)}")
        logger.debug(traceback.format_exc())
        return [{
            "url": url,
            "title": "Error fetching page",
            "content": f"Error fetching page: {str(e)}",
            "metadata": {
                "crawl_timestamp": time.time(),
                "status_code": 0,
                "error": str(e),
                "method": "simple_http_fallback"
            }
        }]

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
            # Use the already imported asyncio module
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        logger.warning(f"Warning: Could not set event loop policy: {e}")
        
    # Run crawl asynchronously but provide a synchronous interface
    try:
        logger.info(f"Starting synchronous crawl of {url}")
        
        # Ensure the result is a dictionary
        result = asyncio.run(crawler.crawl_website(url, depth, max_pages, output_name))
        
        # Verify result structure
        if not isinstance(result, dict):
            error_message = f"Invalid crawler result: expected dict, got {type(result)}"
            logger.error(error_message)
            return {
                "error": error_message,
                "traceback": "",
                "metadata": {
                    "url": url,
                    "depth": depth,
                    "max_pages": max_pages,
                    "output_name": output_name,
                    "method": "crawl4ai" if crawler.crawl4ai_available else "simple_http"
                }
            }
        
        # Ensure the result has expected keys (error or metadata+results)
        if "error" not in result and ("metadata" not in result or "results" not in result):
            error_message = f"Invalid crawler result structure: missing required keys. Available keys: {list(result.keys())}"
            logger.error(error_message)
            return {
                "error": error_message,
                "traceback": "",
                "metadata": result.get("metadata", {
                    "url": url,
                    "depth": depth,
                    "max_pages": max_pages,
                    "output_name": output_name,
                    "method": "crawl4ai" if crawler.crawl4ai_available else "simple_http"
                })
            }
            
        return result
    except Exception as e:
        logger.error(f"Error during crawler execution: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "metadata": {
                "url": url,
                "depth": depth,
                "max_pages": max_pages,
                "output_name": output_name,
                "method": "unknown"
            }
        } 