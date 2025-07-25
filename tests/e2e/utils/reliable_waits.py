"""
Reliable Wait Conditions for E2E Tests

Replaces asyncio.sleep() with intelligent wait conditions based on:
- DOM element visibility/state changes
- Network request completion
- Agent status updates
- UI state transitions
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from playwright.async_api import Page, Locator, TimeoutError as PlaywrightTimeoutError
import httpx

logger = logging.getLogger(__name__)


class ReliableWaits:
    """
    Intelligent wait conditions that replace sleep() calls
    """
    
    def __init__(self, page: Page, default_timeout: int = 30000):
        self.page = page
        self.default_timeout = default_timeout
    
    async def wait_for_file_upload_complete(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for file upload to complete
        Replaces: await asyncio.sleep(3) after file upload
        """
        timeout = timeout or self.default_timeout
        
        try:
            # Wait for upload progress to appear and disappear
            await self.page.wait_for_selector("[data-testid='upload-progress']", timeout=5000)
            await self.page.wait_for_selector("[data-testid='upload-progress']", state="hidden", timeout=timeout)
            
            # Wait for upload success indicator
            await self.page.wait_for_selector("[data-testid='upload-complete'], .upload-success", timeout=5000)
            
            logger.debug("File upload completed successfully")
            return True
            
        except PlaywrightTimeoutError:
            # Fallback: check for file in file list
            try:
                await self.page.wait_for_selector("[data-testid='uploaded-file-item']", timeout=5000)
                return True
            except PlaywrightTimeoutError:
                logger.warning("File upload completion could not be confirmed")
                return False
    
    async def wait_for_agent_response(self, agent_type: Optional[str] = None, timeout: Optional[int] = None) -> bool:
        """
        Wait for agent to respond to request
        Replaces: await asyncio.sleep(5) after sending message
        """
        timeout = timeout or self.default_timeout
        
        try:
            # Wait for typing indicator to appear
            typing_selector = "[data-testid='agent-typing'], .typing-indicator"
            await self.page.wait_for_selector(typing_selector, timeout=3000)
            
            # Wait for typing indicator to disappear (response complete)
            await self.page.wait_for_selector(typing_selector, state="hidden", timeout=timeout)
            
            # Wait for new message to appear
            await self.page.wait_for_selector("[data-testid='ai-message']:last-child", timeout=5000)
            
            # If specific agent type specified, verify agent badge
            if agent_type:
                agent_badge_selector = f"[data-testid='agent-badge'][data-agent='{agent_type}']"
                await self.page.wait_for_selector(agent_badge_selector, timeout=5000)
            
            logger.debug(f"Agent response received {f'from {agent_type}' if agent_type else ''}")
            return True
            
        except PlaywrightTimeoutError:
            logger.warning(f"Agent response timeout {f'for {agent_type}' if agent_type else ''}")
            return False
    
    async def wait_for_analysis_complete(self, timeout: Optional[int] = None) -> bool:
        """
        Wait for data analysis to complete
        Replaces: await asyncio.sleep(10) during analysis
        """
        timeout = timeout or 60000  # Analysis can take longer
        
        try:
            # Wait for analysis to start
            analysis_selectors = [
                "[data-testid='analysis-progress']",
                "[data-testid='analysis-running']",
                ".analysis-in-progress"
            ]
            
            analysis_started = False
            for selector in analysis_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=5000)
                    analysis_started = True
                    break
                except PlaywrightTimeoutError:
                    continue
            
            if analysis_started:
                # Wait for analysis to complete
                await self.page.wait_for_selector(
                    "[data-testid='analysis-complete'], [data-testid='analysis-results']", 
                    timeout=timeout
                )
            
            # Verify results are displayed
            results_visible = await self.page.is_visible("[data-testid='analysis-results'], .analysis-output")
            
            logger.debug("Analysis completed and results displayed")
            return results_visible
            
        except PlaywrightTimeoutError:
            logger.warning("Analysis completion timeout")
            return False
    
    async def wait_for_chart_render(self, chart_type: Optional[str] = None, timeout: Optional[int] = None) -> bool:
        """
        Wait for chart/visualization to render
        Replaces: await asyncio.sleep(3) after chart generation
        """
        timeout = timeout or self.default_timeout
        
        try:
            # Chart containers that indicate rendering
            chart_selectors = [
                "canvas",  # Plotly/Chart.js canvases
                "svg",     # D3/SVG charts
                "[data-testid='chart-container']",
                ".stPlotlyChart",  # Streamlit Plotly
                ".stPyplotGlobalChart"  # Streamlit Pyplot
            ]
            
            if chart_type:
                chart_selectors.insert(0, f"[data-chart-type='{chart_type}']")
            
            # Wait for any chart element to appear
            for selector in chart_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=5000)
                    
                    # Additional wait for chart to be fully rendered
                    await self.page.wait_for_function(
                        f"document.querySelector('{selector}').complete !== false",
                        timeout=timeout
                    )
                    
                    logger.debug(f"Chart rendered: {selector}")
                    return True
                    
                except PlaywrightTimeoutError:
                    continue
            
            logger.warning("No chart elements found within timeout")
            return False
            
        except PlaywrightTimeoutError:
            logger.warning("Chart rendering timeout")
            return False
    
    async def wait_for_agent_health_check(self, agent_ports: List[int], timeout: Optional[int] = None) -> Dict[int, bool]:
        """
        Wait for agents to be healthy
        Replaces: await asyncio.sleep(5) before agent calls
        """
        timeout = timeout or 30000
        start_time = time.time()
        health_status = {}
        
        async with httpx.AsyncClient() as client:
            while (time.time() - start_time) * 1000 < timeout:
                all_healthy = True
                
                for port in agent_ports:
                    if port in health_status and health_status[port]:
                        continue  # Already confirmed healthy
                    
                    try:
                        response = await client.get(
                            f"http://localhost:{port}/health",
                            timeout=5.0
                        )
                        health_status[port] = response.status_code == 200
                        if not health_status[port]:
                            all_healthy = False
                        
                    except Exception:
                        health_status[port] = False
                        all_healthy = False
                
                if all_healthy:
                    logger.debug(f"All agents healthy: {agent_ports}")
                    return health_status
                
                await asyncio.sleep(1)  # Brief sleep between health checks
        
        logger.warning(f"Agent health check timeout. Status: {health_status}")
        return health_status
    
    async def wait_for_ui_state_change(self, 
                                     element_selector: str, 
                                     expected_state: str = "visible",
                                     timeout: Optional[int] = None) -> bool:
        """
        Wait for UI element state to change
        Replaces: await asyncio.sleep(1) after UI interactions
        """
        timeout = timeout or self.default_timeout
        
        try:
            if expected_state == "visible":
                await self.page.wait_for_selector(element_selector, state="visible", timeout=timeout)
            elif expected_state == "hidden":
                await self.page.wait_for_selector(element_selector, state="hidden", timeout=timeout)
            elif expected_state == "enabled":
                await self.page.wait_for_selector(f"{element_selector}:not([disabled])", timeout=timeout)
            elif expected_state == "disabled":
                await self.page.wait_for_selector(f"{element_selector}[disabled]", timeout=timeout)
            
            logger.debug(f"UI state change confirmed: {element_selector} -> {expected_state}")
            return True
            
        except PlaywrightTimeoutError:
            logger.warning(f"UI state change timeout: {element_selector} -> {expected_state}")
            return False
    
    async def wait_for_network_idle(self, timeout: Optional[int] = None, idle_time: int = 500) -> bool:
        """
        Wait for network activity to settle
        Replaces: await asyncio.sleep(2) after triggering network requests
        """
        timeout = timeout or self.default_timeout
        
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
            
            # Additional wait for any async operations
            await asyncio.sleep(idle_time / 1000)
            
            logger.debug("Network activity settled")
            return True
            
        except PlaywrightTimeoutError:
            logger.warning("Network idle timeout")
            return False
    
    async def wait_for_text_content(self, 
                                  element_selector: str, 
                                  expected_text: str = None,
                                  timeout: Optional[int] = None) -> bool:
        """
        Wait for element to contain specific text
        Replaces: await asyncio.sleep(1) waiting for text updates
        """
        timeout = timeout or self.default_timeout
        
        try:
            if expected_text:
                await self.page.wait_for_function(
                    f"document.querySelector('{element_selector}')?.textContent?.includes('{expected_text}')",
                    timeout=timeout
                )
            else:
                # Just wait for element to have any text content
                await self.page.wait_for_function(
                    f"document.querySelector('{element_selector}')?.textContent?.trim().length > 0",
                    timeout=timeout
                )
            
            logger.debug(f"Text content confirmed: {element_selector}")
            return True
            
        except PlaywrightTimeoutError:
            logger.warning(f"Text content timeout: {element_selector}")
            return False
    
    async def wait_for_multiple_agents_response(self, 
                                              expected_agent_count: int,
                                              timeout: Optional[int] = None) -> bool:
        """
        Wait for multiple agents to complete their responses
        Replaces: await asyncio.sleep(10) in multi-agent scenarios
        """
        timeout = timeout or 60000  # Multi-agent operations take longer
        
        try:
            # Wait for all agent responses to appear
            await self.page.wait_for_function(
                f"document.querySelectorAll('[data-testid=\"agent-response\"]').length >= {expected_agent_count}",
                timeout=timeout
            )
            
            # Wait for all typing indicators to disappear
            await self.page.wait_for_selector("[data-testid='agent-typing']", state="hidden", timeout=10000)
            
            logger.debug(f"All {expected_agent_count} agents responded")
            return True
            
        except PlaywrightTimeoutError:
            logger.warning(f"Multi-agent response timeout (expected {expected_agent_count})")
            return False
    
    async def smart_wait_with_fallback(self, 
                                     primary_condition: Callable,
                                     fallback_sleep: float = 1.0,
                                     max_attempts: int = 3) -> bool:
        """
        Attempt smart wait with fallback to sleep if needed
        Use when migrating from sleep() incrementally
        """
        for attempt in range(max_attempts):
            try:
                result = await primary_condition()
                if result:
                    return True
            except Exception as e:
                logger.debug(f"Smart wait attempt {attempt + 1} failed: {e}")
            
            if attempt < max_attempts - 1:
                await asyncio.sleep(fallback_sleep)
        
        logger.warning("Smart wait failed, falling back to final sleep")
        await asyncio.sleep(fallback_sleep)
        return False


class TestWaitConditions:
    """
    Specialized wait conditions for common test scenarios
    """
    
    @staticmethod
    async def wait_for_file_processing(page: Page, file_name: str, timeout: int = 30000) -> bool:
        """Wait for file processing to complete"""
        waits = ReliableWaits(page, timeout)
        
        # File upload completion
        if not await waits.wait_for_file_upload_complete():
            return False
        
        # Data processing
        if not await waits.wait_for_analysis_complete():
            return False
        
        # Results display
        return await waits.wait_for_text_content("[data-testid='file-analysis-results']")
    
    @staticmethod
    async def wait_for_agent_collaboration(page: Page, 
                                         agent_types: List[str], 
                                         timeout: int = 60000) -> bool:
        """Wait for multi-agent collaboration to complete"""
        waits = ReliableWaits(page, timeout)
        
        # Wait for each agent to respond in sequence
        for agent_type in agent_types:
            if not await waits.wait_for_agent_response(agent_type, timeout=timeout//len(agent_types)):
                return False
        
        # Final integration wait
        return await waits.wait_for_multiple_agents_response(len(agent_types))
    
    @staticmethod
    async def wait_for_error_recovery(page: Page, timeout: int = 15000) -> bool:
        """Wait for error recovery mechanisms to complete"""
        waits = ReliableWaits(page, timeout)
        
        # Wait for error message to appear
        await waits.wait_for_ui_state_change("[data-testid='error-message']", "visible", 5000)
        
        # Wait for recovery attempt
        await waits.wait_for_ui_state_change("[data-testid='retry-button'], [data-testid='fallback-message']", "visible", timeout)
        
        return True