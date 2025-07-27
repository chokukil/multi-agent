"""
Playwright Validation Test - Quick diagnostic test
"""

import pytest
import asyncio
from playwright.async_api import async_playwright, Page


@pytest.mark.asyncio
async def test_playwright_browser_launch():
    """Test that Playwright can launch a browser successfully"""
    async with async_playwright() as p:
        # Test Chromium launch
        browser = await p.chromium.launch()
        assert browser is not None, "Chromium browser should launch successfully"
        
        # Test page creation
        page = await browser.new_page()
        assert page is not None, "Page should be created successfully"
        
        # Test navigation to a simple page
        await page.goto("data:text/html,<h1>Playwright Test</h1>")
        
        # Test element interaction
        element = await page.query_selector("h1")
        assert element is not None, "H1 element should be found"
        
        text_content = await element.text_content()
        assert text_content == "Playwright Test", "Text content should match"
        
        await browser.close()


@pytest.mark.asyncio
async def test_playwright_configuration():
    """Test Playwright configuration and capabilities"""
    async with async_playwright() as p:
        # Test browser contexts
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Cherry AI E2E Test"
        )
        
        page = await context.new_page()
        
        # Test viewport
        viewport_size = page.viewport_size
        assert viewport_size["width"] == 1920, "Viewport width should be 1920"
        assert viewport_size["height"] == 1080, "Viewport height should be 1080"
        
        # Test JavaScript execution
        result = await page.evaluate("() => { return 2 + 2; }")
        assert result == 4, "JavaScript evaluation should work"
        
        await browser.close()


@pytest.mark.asyncio
async def test_playwright_wait_conditions():
    """Test Playwright wait conditions"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Create a page with delayed content
        html_content = """
        <html>
        <body>
            <div id="initial">Loading...</div>
            <script>
                setTimeout(() => {
                    document.getElementById('initial').textContent = 'Content Loaded';
                    document.getElementById('initial').setAttribute('data-testid', 'loaded-content');
                }, 1000);
            </script>
        </body>
        </html>
        """
        
        await page.goto(f"data:text/html,{html_content}")
        
        # Test wait for selector with timeout
        loaded_element = await page.wait_for_selector("[data-testid='loaded-content']", timeout=5000)
        assert loaded_element is not None, "Should wait for content to load"
        
        text = await loaded_element.text_content()
        assert text == "Content Loaded", "Content should be updated after delay"
        
        await browser.close()


@pytest.mark.asyncio
async def test_playwright_error_handling():
    """Test Playwright error handling"""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Test timeout error handling
        with pytest.raises(Exception):  # Should be TimeoutError but catching general for compatibility
            await page.wait_for_selector("#non-existent-element", timeout=1000)
        
        # Test navigation error handling
        try:
            await page.goto("http://invalid-url-that-does-not-exist.com")
        except Exception as e:
            # Navigation errors are expected for invalid URLs
            assert "net::" in str(e) or "Navigation" in str(e) or "timeout" in str(e).lower()
        
        await browser.close()


if __name__ == "__main__":
    # Run the validation tests
    pytest.main([__file__, "-v"])