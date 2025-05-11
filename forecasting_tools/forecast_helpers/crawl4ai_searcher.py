from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)

class Crawl4AISearcher:
    """
    Client for Crawl4AI API integration.
    
    Provides deep web crawling capabilities optimized for high-quality research
    and forecasting data collection.
    """
    
    # Default configuration
    DEFAULT_MAX_DEPTH = 2
    DEFAULT_MAX_RESULTS = 10
    BASE_URL = "https://api.crawl4ai.io/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Crawl4AI client.
        
        Args:
            api_key: Optional API key. If not provided, will look for CRAWL4AI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("CRAWL4AI_API_KEY")
        self.client = None
        
        # Check for Streamlit secrets if API key not found
        if not self.api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "crawl4ai" in st.secrets:
                    self.api_key = st.secrets.crawl4ai.api_key
                    logger.info("Using Crawl4AI API key from Streamlit secrets")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Error accessing Streamlit secrets: {e}")
    
    async def setup(self):
        """Lazy initialization of HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=60.0)
            # Add default headers
            self.client.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            })
    
    async def search(self, 
                    query: str, 
                    depth: int = DEFAULT_MAX_DEPTH,
                    domains: Optional[List[str]] = None,
                    max_results: int = DEFAULT_MAX_RESULTS,
                    time_range: Optional[str] = None,
                    language: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a Crawl4AI search with the specified parameters.
        
        Args:
            query: The search query to execute
            depth: How deep to crawl (1-3)
            domains: Optional list of domains to focus on
            max_results: Maximum number of results to return
            time_range: Optional time range (e.g., "1d", "1w", "1m", "1y")
            language: Optional language code (e.g., "en", "es", "fr")
            
        Returns:
            A dictionary containing the search results
        """
        if not self.api_key:
            logger.warning("No Crawl4AI API key found. Skipping search.")
            return {"results": [], "error": "No API key configured"}
        
        await self.setup()
        
        payload = {
            "query": query,
            "depth": min(depth, 3),  # Cap at 3 to prevent excessive resource usage
            "max_results": max_results
        }
        
        # Add optional parameters
        if domains:
            payload["domains"] = domains
        if time_range:
            payload["time_range"] = time_range
        if language:
            payload["language"] = language
            
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/search",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during Crawl4AI search: {e.response.status_code} - {e.response.text}")
            return {"results": [], "error": f"API error: {e.response.status_code}"}
        except Exception as e:
            logger.exception(f"Error during Crawl4AI search: {e}")
            return {"results": [], "error": f"Error: {str(e)}"}
    
    def get_formatted_search_results(self, query: str, depth: int = DEFAULT_MAX_DEPTH) -> str:
        """
        Get formatted search results as a Markdown string.
        
        Args:
            query: The search query
            depth: Search depth (1-3)
            
        Returns:
            Formatted Markdown string with search results
        """
        return asyncio.run(self.get_formatted_search_results_async(query, depth))
    
    async def get_formatted_search_results_async(self, 
                                              query: str, 
                                              depth: int = DEFAULT_MAX_DEPTH,
                                              domains: Optional[List[str]] = None) -> str:
        """
        Get formatted search results as a Markdown string (async version).
        
        Args:
            query: The search query
            depth: Search depth (1-3)
            domains: Optional list of domains to focus on
            
        Returns:
            Formatted Markdown string with search results
        """
        search_results = await self.search(query, depth, domains)
        
        if "error" in search_results and search_results.get("results", []) == []:
            return f"No results found via Crawl4AI. Error: {search_results['error']}"
        
        results = search_results.get("results", [])
        if not results:
            return "No results found via Crawl4AI."
        
        formatted_output = "## Crawl4AI Deep Search Results\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "#")
            summary = result.get("summary", "No summary available")
            source = result.get("source", "Unknown source")
            date = result.get("date")
            
            # Format the date if present
            date_str = ""
            if date:
                try:
                    # Convert ISO date to readable format
                    date_obj = datetime.fromisoformat(date.replace("Z", "+00:00"))
                    date_str = f"Published: {date_obj.strftime('%B %d, %Y')}\n"
                except (ValueError, TypeError):
                    date_str = f"Published: {date}\n"
            
            formatted_output += f"### {i}. {title}\n"
            formatted_output += f"**Source**: {source}\n"
            formatted_output += date_str
            formatted_output += f"**URL**: {url}\n\n"
            formatted_output += f"{summary}\n\n"
            formatted_output += "---\n\n"
        
        return formatted_output
    
    async def close(self):
        """Close the HTTP client and release resources."""
        if self.client:
            await self.client.aclose()
            self.client = None 