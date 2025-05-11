from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from forecasting_tools.forecast_helpers.browser_manager import BrowserManager, BrowserManagerError, BrowserPool


@pytest.mark.asyncio
@patch("forecasting_tools.forecast_helpers.browser_manager.BrowserManager._initialize")
async def test_browser_manager_instance(mock_initialize):
    """Test that BrowserManager singleton pattern works correctly."""
    # Mock initialize to avoid actual initialization
    mock_initialize.return_value = None
    
    # Get multiple instances
    instance1 = await BrowserManager.get_instance()
    instance2 = await BrowserManager.get_instance()
    
    # Verify singleton pattern
    assert instance1 is instance2
    assert mock_initialize.call_count == 1  # Should only be called once


@pytest.mark.asyncio
@patch("playwright.async_api.async_playwright")
async def test_browser_manager_initialization(mock_playwright):
    """Test BrowserManager initialization with mocked Playwright."""
    # Setup mocks
    mock_playwright_instance = AsyncMock()
    mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
    
    # Create manager and initialize
    manager = BrowserManager()
    await manager._initialize()
    
    # Verify initialization
    assert manager._initialized is True
    assert manager.playwright is mock_playwright_instance
    assert isinstance(manager.browser_pool, BrowserPool)


@pytest.mark.asyncio
@patch("forecasting_tools.forecast_helpers.browser_manager.BrowserManager._initialize")
@patch("forecasting_tools.forecast_helpers.browser_manager.BrowserManager._setup_security_protections")
@patch("forecasting_tools.forecast_helpers.browser_manager.BrowserManager.log_browser_activity")
async def test_extract_content(mock_log, mock_security, mock_initialize):
    """Test content extraction with mocked browser."""
    # Setup manager with mocked components
    manager = BrowserManager()
    manager._initialized = True
    
    # Mock browser pool
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = AsyncMock()
    
    # Setup response
    mock_response = AsyncMock()
    mock_response.status = 200
    
    # Setup page behavior
    mock_page.goto = AsyncMock(return_value=mock_response)
    mock_page.wait_for_selector = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="Test content")
    mock_page.url = "https://example.com"
    
    # Setup context behavior
    mock_context.new_page = AsyncMock(return_value=mock_page)
    
    # Setup browser pool behavior
    manager.browser_pool = MagicMock()
    manager.browser_pool.get_browser_and_context = AsyncMock(return_value=(mock_browser, mock_context))
    manager.browser_pool.release_browser_and_context = AsyncMock()
    
    # Mock additional methods
    manager._additional_sanitization = MagicMock(return_value="Sanitized test content")
    
    # Call extract_content
    content = await manager.extract_content(
        url="https://example.com",
        selector=".content",
        wait_for="#loaded"
    )
    
    # Verify expectations
    assert content == "Sanitized test content"
    mock_page.goto.assert_called_once()
    mock_page.wait_for_selector.assert_called_with(".content", timeout=30000)
    mock_page.evaluate.assert_called_once()
    manager.browser_pool.release_browser_and_context.assert_called_once()


@pytest.mark.asyncio
async def test_security_protections():
    """Test security protections using static analysis."""
    # These are security properties that should be present in the code
    # We don't execute them, but verify they exist
    
    manager = BrowserManager()
    
    # Security functions that should exist
    assert hasattr(manager, '_sanitize_log_entry')
    assert hasattr(manager, '_additional_sanitization')
    assert hasattr(manager, '_setup_security_protections')
    assert hasattr(manager, '_is_trusted_source')
    
    # Check that trusted domains are limited
    assert hasattr(manager, '_TRUSTED_DOMAINS')
    assert isinstance(manager._TRUSTED_DOMAINS, list)
    assert len(manager._TRUSTED_DOMAINS) > 0
    
    # Check that user agents are randomized
    assert hasattr(manager, '_USER_AGENTS')
    assert isinstance(manager._USER_AGENTS, list)
    assert len(manager._USER_AGENTS) > 1 