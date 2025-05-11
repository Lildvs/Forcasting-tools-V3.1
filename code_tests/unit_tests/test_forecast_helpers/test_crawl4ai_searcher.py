from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from forecasting_tools.forecast_helpers.crawl4ai_searcher import Crawl4AISearcher


@pytest.mark.asyncio
async def test_crawl4ai_searcher_no_api_key():
    """Test that Crawl4AI searcher handles missing API key gracefully."""
    searcher = Crawl4AISearcher(api_key=None)
    results = await searcher.search("test query")
    
    assert results["results"] == []
    assert "message" in results


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post")
async def test_crawl4ai_searcher_search(mock_post):
    """Test that Crawl4AI searcher correctly formats requests and handles responses."""
    # Mock the HTTP response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com/test",
                "snippet": "This is a test result",
                "date": "2023-05-01"
            }
        ]
    })
    
    # Configure the mock to return our response
    mock_post.return_value.__aenter__.return_value = mock_response
    
    # Create a searcher with a dummy API key
    searcher = Crawl4AISearcher(api_key="test_key")
    
    # Call the search method
    results = await searcher.search("test query", depth=2)
    
    # Verify the request was made correctly
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "search" in args[0]
    assert kwargs["json"]["query"] == "test query"
    assert kwargs["json"]["depth"] == 2
    
    # Verify the results were processed correctly
    assert len(results["results"]) == 1
    assert results["results"][0]["title"] == "Test Result"


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post")
async def test_crawl4ai_searcher_formatted_results(mock_post):
    """Test that Crawl4AI searcher formats results correctly."""
    # Mock the HTTP response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/test1",
                "snippet": "This is test result 1",
                "date": "2023-05-01"
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/test2",
                "snippet": "This is test result 2",
                "date": "2023-06-01"
            }
        ]
    })
    
    # Configure the mock
    mock_post.return_value.__aenter__.return_value = mock_response
    
    # Create a searcher with a dummy API key
    searcher = Crawl4AISearcher(api_key="test_key")
    
    # Get formatted results
    formatted = await searcher.get_formatted_search_results("test query")
    
    # Verify formatting
    assert "Test Result 1" in formatted
    assert "Test Result 2" in formatted
    assert "https://example.com/test1" in formatted
    assert "https://example.com/test2" in formatted
    assert "2023-05-01" in formatted
    assert "2023-06-01" in formatted 