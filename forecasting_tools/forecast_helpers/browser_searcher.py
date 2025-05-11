from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from forecasting_tools.forecast_helpers.browser_manager import BrowserManager, BrowserManagerError

logger = logging.getLogger(__name__)


class BrowserSearcher:
    """
    Secure web content extraction using browser automation.
    
    This class provides an interface for extracting information from web pages
    that cannot be accessed through standard APIs, with strong security protections.
    """
    
    def __init__(self, max_browsers: int = 2, page_timeout_ms: int = 30000):
        """
        Initialize the browser searcher.
        
        Args:
            max_browsers: Maximum number of concurrent browser instances
            page_timeout_ms: Default timeout for page operations in milliseconds
        """
        self.max_browsers = max_browsers
        self.page_timeout_ms = page_timeout_ms
        self.browser_manager = None
    
    async def setup(self) -> None:
        """Initialize the browser manager."""
        self.browser_manager = await BrowserManager.get_instance(
            max_browsers=self.max_browsers,
            page_timeout_ms=self.page_timeout_ms
        )
    
    async def close(self) -> None:
        """Release resources."""
        if self.browser_manager:
            await self.browser_manager.close()
            self.browser_manager = None
    
    async def extract_page_content(
        self, 
        url: str, 
        content_selector: str = "body", 
        wait_for_selector: Optional[str] = None,
        timeout_ms: Optional[int] = None
    ) -> str:
        """
        Extract content from a web page using browser automation.
        
        Args:
            url: The URL to navigate to
            content_selector: CSS selector for the content to extract
            wait_for_selector: Optional selector to wait for before extraction
            timeout_ms: Optional custom timeout in milliseconds
            
        Returns:
            Extracted and sanitized content as string
        """
        if not self.browser_manager:
            await self.setup()
        
        try:
            content = await self.browser_manager.extract_content(
                url=url,
                selector=content_selector,
                wait_for=wait_for_selector,
                timeout_ms=timeout_ms
            )
            return content
        except BrowserManagerError as e:
            logger.error(f"Browser extraction error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in browser extraction: {e}")
            return ""
    
    async def get_formatted_search_results(
        self, 
        query: str,
        search_url: Optional[str] = None,
        content_selector: str = ".search-results", 
        result_item_selector: str = ".result-item",
        search_box_selector: str = "input[type='search']",
        search_button_selector: str = "button[type='submit']"
    ) -> str:
        """
        Extract search results using browser automation.
        
        Args:
            query: Search query
            search_url: URL of the search engine (or None to use default)
            content_selector: CSS selector for the search results container
            result_item_selector: CSS selector for individual result items
            search_box_selector: CSS selector for the search input field
            search_button_selector: CSS selector for the search button
            
        Returns:
            Formatted search results
        """
        if not self.browser_manager:
            await self.setup()
        
        # Default to metaforecast if no search URL is provided
        if not search_url:
            search_url = f"https://metaforecast.org/search?query={query}"
            
        try:
            # Extract content from the search page
            content = await self.browser_manager.extract_content(
                url=search_url,
                selector=content_selector,
                timeout_ms=45000  # Longer timeout for search results
            )
            
            if not content:
                return "No search results found."
            
            # Format the results
            formatted_results = f"## Browser Search Results for: {query}\n\n"
            formatted_results += content
            
            # Add timestamp for freshness tracking
            formatted_results += f"\n\n*Results fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error extracting search results: {e}")
            return f"Error retrieving search results: {str(e)}"
    
    def get_formatted_results_sync(self, query: str) -> str:
        """Synchronous wrapper for get_formatted_search_results."""
        try:
            return asyncio.run(self.get_formatted_search_results(query))
        finally:
            # Make sure to clean up resources in synchronous context
            if self.browser_manager:
                asyncio.run(self.close())
                
    @staticmethod
    def is_available() -> bool:
        """Check if browser automation is available in this environment."""
        try:
            import playwright.async_api
            return True
        except ImportError:
            return False 