from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import aiohttp

logger = logging.getLogger(__name__)


class Crawl4AISearcher:
    """
    Client for Crawl4AI API integration.
    
    This class provides an interface to the Crawl4AI service for deep web crawling
    and information extraction, optimized for forecasting tasks.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Crawl4AI searcher.
        
        Args:
            api_key: Optional API key. If not provided, will try to load from 
                    CRAWL4AI_API_KEY environment variable or Streamlit secrets.
        """
        # Try to get the API key from the provided parameter
        self.api_key = api_key
        
        # If no API key was provided, try to get it from environment variables
        if self.api_key is None:
            self.api_key = os.getenv("CRAWL4AI_API_KEY")
            
        # If still no API key, try to get it from Streamlit secrets
        if self.api_key is None:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "crawl4ai" in st.secrets and "api_key" in st.secrets.crawl4ai:
                    self.api_key = st.secrets.crawl4ai.api_key
                    logger.info("Using Crawl4AI API key from Streamlit secrets")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Error accessing Streamlit secrets: {e}")
        
        self.base_url = "https://api.crawl4ai.com/v1"
        self.session = None
    
    async def setup(self):
        """Lazy initialization of resources."""
        if not self.session and self.api_key:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
    
    async def close(self):
        """Release resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search(
        self, 
        query: str, 
        depth: int = 2, 
        domains: Optional[List[str]] = None,
        max_results: int = 10,
        include_content: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a search with Crawl4AI.
        
        Args:
            query: The search query
            depth: How deep to crawl (1-3)
            domains: Optional list of domains to restrict the search to
            max_results: Maximum number of results to return
            include_content: Whether to include the full content in results
            
        Returns:
            Dictionary containing search results
        """
        if not self.api_key:
            logger.warning("CRAWL4AI_API_KEY not set, skipping search")
            return {"results": [], "message": "API key not configured"}
        
        await self.setup()
        
        payload = {
            "query": query,
            "depth": depth,
            "max_results": max_results,
            "include_content": include_content
        }
        
        if domains:
            payload["domains"] = domains
        
        try:
            async with self.session.post(
                f"{self.base_url}/search", json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Crawl4AI search failed: {error_text}")
                    return {
                        "results": [],
                        "error": f"API returned status code {response.status}"
                    }
                
                data = await response.json()
                return data
        except Exception as e:
            logger.exception(f"Error during Crawl4AI search: {e}")
            return {"results": [], "error": str(e)}
    
    async def get_formatted_search_results(
        self, 
        query: str, 
        depth: int = 2,
        domains: Optional[List[str]] = None
    ) -> str:
        """
        Get formatted search results for the given query.
        
        Args:
            query: The search query
            depth: How deep to crawl (1-3)
            domains: Optional list of domains to restrict the search to
            
        Returns:
            Formatted string containing search results
        """
        results = await self.search(query, depth, domains)
        
        if "error" in results:
            return f"Error from Crawl4AI: {results['error']}"
        
        if not results.get("results", []):
            return "No results found from Crawl4AI."
        
        formatted_results = "## Crawl4AI Search Results\n\n"
        
        for i, result in enumerate(results.get("results", []), 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            date = result.get("date", "")
            
            formatted_results += f"### {i}. {title}\n"
            formatted_results += f"**Source:** [{url}]({url})\n"
            if date:
                formatted_results += f"**Date:** {date}\n"
            formatted_results += f"\n{snippet}\n\n"
            
            # Add separator between results except for the last one
            if i < len(results.get("results", [])):
                formatted_results += "---\n\n"
        
        return formatted_results
        
    def get_formatted_results_sync(self, query: str, depth: int = 2, domains: Optional[List[str]] = None) -> str:
        """Synchronous wrapper for get_formatted_search_results."""
        return asyncio.run(self.get_formatted_search_results(query, depth, domains)) 