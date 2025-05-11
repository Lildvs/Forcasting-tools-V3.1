from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Define custom exceptions
class BrowserManagerError(Exception):
    """Base exception for browser manager errors."""
    pass

class ResourceExhaustedError(BrowserManagerError):
    """Exception raised when browser resources are exhausted."""
    pass

class BrowserManager:
    """
    Secure browser automation manager using Playwright.
    
    This class provides a secure interface for browser automation tasks,
    with built-in protections against common security vulnerabilities:
    - XSS protection
    - Resource exhaustion prevention
    - Secure credential management
    - Content sanitization
    - Browser fingerprinting protection
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    # Common user agents for fingerprinting protection
    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
    ]
    
    # Trusted domains that are allowed to execute scripts
    _TRUSTED_DOMAINS = [
        "metaculus.com",
        "metaforecast.org"
    ]
    
    def __init__(self, max_browsers: int = 3, page_timeout_ms: int = 30000):
        """
        Initialize the browser manager.
        
        Args:
            max_browsers: Maximum number of concurrent browser instances
            page_timeout_ms: Default timeout for page operations in milliseconds
        """
        self.max_browsers = max_browsers
        self.page_timeout_ms = page_timeout_ms
        self.active_browsers = []
        self.playwright = None
        self.browser_pool = None
        self._initialized = False
    
    @classmethod
    async def get_instance(cls, max_browsers: int = 3, page_timeout_ms: int = 30000) -> BrowserManager:
        """
        Get or create the singleton instance of BrowserManager.
        
        This ensures only one browser manager is active at a time.
        
        Args:
            max_browsers: Maximum number of concurrent browser instances
            page_timeout_ms: Default timeout for page operations in milliseconds
            
        Returns:
            The BrowserManager instance
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = BrowserManager(max_browsers, page_timeout_ms)
            
            if not cls._instance._initialized:
                await cls._instance._initialize()
                
            return cls._instance
    
    async def _initialize(self) -> None:
        """Initialize the browser manager and create the browser pool."""
        try:
            # Import at runtime to avoid dependency issues
            try:
                from playwright.async_api import async_playwright
                self.playwright = await async_playwright().start()
                self._initialized = True
                self.browser_pool = BrowserPool(self.playwright, self.max_browsers)
                logger.info("Browser manager initialized successfully")
            except ImportError:
                logger.warning("Playwright not installed. Browser automation will not be available.")
                logger.warning("Install with: pip install playwright && python -m playwright install")
                self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize browser manager: {e}")
            self._initialized = False
            raise
    
    async def close(self) -> None:
        """Close all browser instances and cleanup resources."""
        if self.browser_pool:
            await self.browser_pool.close_all()
        
        if self.playwright:
            await self.playwright.stop()
            
        self._initialized = False
        BrowserManager._instance = None
        logger.info("Browser manager closed")
    
    async def __aenter__(self) -> BrowserManager:
        """Async context manager entry."""
        if not self._initialized:
            await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent to avoid fingerprinting."""
        return random.choice(self._USER_AGENTS)
    
    def _is_trusted_source(self, url: str) -> bool:
        """Check if a URL is from a trusted source."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return any(trusted in domain for trusted in self._TRUSTED_DOMAINS)
        except Exception:
            return False
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID for tracking."""
        return f"session_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def _sanitize_log_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from log entries."""
        sanitized = entry.copy()
        # Remove any potentially sensitive parameters in URL
        if "url" in sanitized:
            try:
                from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
                parsed = urlparse(sanitized["url"])
                query_params = parse_qs(parsed.query)
                
                # Remove sensitive parameters
                for param in ["key", "token", "api_key", "password", "secret"]:
                    if param in query_params:
                        query_params[param] = ["[REDACTED]"]
                
                # Rebuild URL
                sanitized_query = urlencode(query_params, doseq=True)
                sanitized_url = urlunparse(
                    (parsed.scheme, parsed.netloc, parsed.path, 
                     parsed.params, sanitized_query, parsed.fragment)
                )
                sanitized["url"] = sanitized_url
            except Exception as e:
                logger.warning(f"Failed to sanitize URL: {e}")
        
        return sanitized
    
    def _additional_sanitization(self, content: str) -> str:
        """Additional sanitization for extracted content."""
        if not content:
            return ""
        
        # Remove any potential script tags or dangerous content
        import re
        sanitized = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', content)
        sanitized = re.sub(r'javascript:', '', sanitized)
        sanitized = re.sub(r'on\w+\s*=', '', sanitized)
        
        return sanitized
    
    async def _check_navigation(self, frame: Any) -> None:
        """Monitor for suspicious redirects."""
        url = frame.url
        if not self._is_trusted_source(url):
            logger.warning(f"Navigation to untrusted source: {url}")
    
    async def _capture_error_screenshot(self, page: Any) -> Optional[str]:
        """Capture a screenshot when an error occurs for debugging."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_screenshot_{timestamp}.png"
            filepath = os.path.join(os.getcwd(), "logs", filename)
            
            # Ensure log directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            await page.screenshot(path=filepath)
            logger.info(f"Error screenshot saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to capture error screenshot: {e}")
            return None
    
    async def get_browser_and_context(self) -> Tuple[Any, Any]:
        """
        Get a browser and context with secure settings.
        
        Returns:
            A tuple of (browser, context)
        """
        if not self._initialized:
            raise BrowserManagerError("Browser manager not initialized")
        
        browser, context = await self.browser_pool.get_browser_and_context()
        return browser, context
    
    async def extract_content(self, url: str, selector: str, wait_for: Optional[str] = None, 
                             timeout_ms: Optional[int] = None) -> str:
        """
        Securely extract content from a webpage.
        
        Args:
            url: URL to navigate to
            selector: CSS selector for content to extract
            wait_for: Optional selector to wait for before extraction
            timeout_ms: Optional custom timeout in milliseconds
            
        Returns:
            Extracted and sanitized content as string
        """
        if not self._initialized:
            raise BrowserManagerError("Browser manager not initialized")
        
        timeout = timeout_ms or self.page_timeout_ms
        browser, context = None, None
        
        try:
            browser, context = await self.browser_pool.get_browser_and_context()
            page = await context.new_page()
            
            # Set up security protections
            await self._setup_security_protections(page)
            
            # Navigate to the URL
            await self.log_browser_activity(page, "navigating", {"url": url})
            response = await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            
            if not response:
                raise BrowserManagerError(f"Failed to navigate to {url}: No response")
            
            if response.status >= 400:
                raise BrowserManagerError(f"Failed to navigate to {url}: Status {response.status}")
            
            # Wait for specific element if requested
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout)
            
            # Wait for the selector
            await page.wait_for_selector(selector, timeout=timeout)
            
            # Extract content securely
            content = await page.evaluate("""(selector) => {
                const el = document.querySelector(selector);
                if (!el) return '';
                
                return el.innerText || el.textContent || '';
            }""", selector)
            
            sanitized_content = self._additional_sanitization(content)
            await self.log_browser_activity(page, "extracted_content", 
                                          {"selector": selector, "length": len(sanitized_content)})
            
            return sanitized_content
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            if 'page' in locals():
                await self._capture_error_screenshot(page)
            return ""
        finally:
            if context and browser:
                await self.browser_pool.release_browser_and_context(browser, context)
    
    async def _setup_security_protections(self, page: Any) -> None:
        """Set up security protections for a page."""
        # Add security headers
        await page.set_extra_http_headers({
            "X-XSS-Protection": "1; mode=block",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY"
        })
        
        # Block potentially dangerous resources
        await page.route("**/*.{html,js,mjs}", 
                       lambda route: route.abort() if not self._is_trusted_source(route.request.url) 
                       else route.continue_())
        
        # Monitor for suspicious redirects
        page.on("framenavigated", lambda frame: self._check_navigation(frame))
        
        # Prevent domain relaxation
        await page.add_init_script("""
            Object.defineProperty(document, 'domain', {
                get: function() { return this.location.hostname; },
                set: function() { /* prevent domain relaxing */ }
            });
        """)
        
        # Disable alert/confirm/prompt dialogs
        page.on("dialog", lambda dialog: dialog.dismiss())
    
    async def log_browser_activity(self, page: Any, action: str, details: Optional[Dict] = None) -> None:
        """
        Log browser activity with security in mind.
        
        Args:
            page: The page being used
            action: The action being performed
            details: Optional details about the action
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "url": page.url,
            "details": details or {},
            "session_id": self._generate_session_id()
        }
        
        sanitized_entry = self._sanitize_log_entry(entry)
        logger.info(f"Browser activity: {json.dumps(sanitized_entry)}")
    
    @staticmethod
    def _get_auth_credentials() -> Optional[Dict[str, str]]:
        """
        Get authentication credentials from secure sources.
        
        Returns:
            Dictionary with username and password or None if not configured
        """
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "playwright" in st.secrets:
                return {
                    "username": st.secrets.playwright.username,
                    "password": st.secrets.playwright.password
                }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error accessing Streamlit secrets: {e}")
        
        # Fallback to environment variables
        username = os.getenv("PLAYWRIGHT_USERNAME")
        password = os.getenv("PLAYWRIGHT_PASSWORD")
        
        if not username or not password:
            logger.warning("Playwright credentials not configured")
            return None
            
        return {"username": username, "password": password}


class BrowserPool:
    """
    Pool of browser instances to manage resource usage.
    
    This class handles browser lifecycle and reuse to prevent resource exhaustion.
    """
    
    def __init__(self, playwright: Any, max_browsers: int = 3):
        """
        Initialize the browser pool.
        
        Args:
            playwright: The playwright instance
            max_browsers: Maximum number of concurrent browser instances
        """
        self.playwright = playwright
        self.max_browsers = max_browsers
        self.browsers = []
        self.contexts = {}  # Map browsers to their contexts
        self.lock = asyncio.Lock()
    
    async def get_browser_and_context(self) -> Tuple[Any, Any]:
        """
        Get a browser and context from the pool or create new ones.
        
        Returns:
            Tuple of (browser, context)
        """
        async with self.lock:
            # Check if we have any available browsers
            for browser in self.browsers:
                contexts = self.contexts.get(browser, [])
                if len(contexts) < 5:  # Max 5 contexts per browser
                    # Create a new context for this browser
                    context = await self._create_secure_context(browser)
                    self.contexts[browser].append(context)
                    return browser, context
            
            # No available browsers with room for contexts, create a new one if under limit
            if len(self.browsers) < self.max_browsers:
                browser = await self._create_secure_browser()
                self.browsers.append(browser)
                context = await self._create_secure_context(browser)
                self.contexts[browser] = [context]
                return browser, context
            
            # We've hit the limit
            raise ResourceExhaustedError("Maximum number of browsers reached")
    
    async def _create_secure_browser(self) -> Any:
        """
        Create a new browser with secure settings.
        
        Returns:
            A new browser instance
        """
        browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-features=Site-per-process",
                "--disable-webrtc"
            ]
        )
        logger.info("Created new secure browser instance")
        return browser
    
    async def _create_secure_context(self, browser: Any) -> Any:
        """
        Create a new browser context with secure settings.
        
        Args:
            browser: The browser to create a context for
            
        Returns:
            A new browser context
        """
        # Get a random user agent
        user_agent = random.choice(BrowserManager._USER_AGENTS)
        
        context = await browser.new_context(
            java_script_enabled=True,
            bypass_csp=False,
            viewport={"width": 1280, "height": 800},
            user_agent=user_agent,
            locale="en-US",
            timezone_id="UTC",
            permissions=[]
        )
        
        # Clear storage between runs
        await context.clear_cookies()
        
        logger.info("Created new secure browser context")
        return context
    
    async def release_browser_and_context(self, browser: Any, context: Any) -> None:
        """
        Release a browser and context back to the pool.
        
        Args:
            browser: The browser to release
            context: The context to release
        """
        async with self.lock:
            try:
                # Close the context
                await context.close()
                
                # Remove from contexts map
                if browser in self.contexts and context in self.contexts[browser]:
                    self.contexts[browser].remove(context)
                
                logger.info("Released browser context")
            except Exception as e:
                logger.error(f"Error releasing browser context: {e}")
    
    async def close_all(self) -> None:
        """Close all browsers and contexts in the pool."""
        async with self.lock:
            try:
                for browser in self.browsers:
                    try:
                        await browser.close()
                    except Exception as e:
                        logger.error(f"Error closing browser: {e}")
                
                self.browsers = []
                self.contexts = {}
                logger.info("Closed all browsers")
            except Exception as e:
                logger.error(f"Error closing all browsers: {e}") 