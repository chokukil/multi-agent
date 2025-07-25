"""
Helper functions for E2E testing of Cherry AI Streamlit Platform
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
import pandas as pd
from playwright.async_api import Page, Download
import logging

logger = logging.getLogger(__name__)


class TestHelpers:
    """Collection of helper functions for E2E tests."""
    
    @staticmethod
    async def measure_performance(func, *args, **kwargs) -> tuple:
        """Measure execution time of an async function."""
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    
    @staticmethod
    async def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
        """Retry a function on failure with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: int = 30, check_interval: float = 0.5) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(check_interval)
        return False
    
    @staticmethod
    def generate_test_report(test_results: Dict[str, Any], output_path: str):
        """Generate a comprehensive test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(test_results),
                "passed": sum(1 for r in test_results.values() if r.get("status") == "passed"),
                "failed": sum(1 for r in test_results.values() if r.get("status") == "failed"),
                "skipped": sum(1 for r in test_results.values() if r.get("status") == "skipped")
            },
            "test_results": test_results,
            "environment": {
                "platform": "Cherry AI Streamlit Platform",
                "test_type": "E2E",
                "browser": "Chromium"
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report generated: {output_path}")
        return report
    
    @staticmethod
    async def verify_file_download(download: Download, expected_format: str) -> bool:
        """Verify that a file was downloaded correctly."""
        # Save the download
        download_path = f"tests/e2e/downloads/{download.suggested_filename}"
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        await download.save_as(download_path)
        
        # Verify file exists and has content
        if not os.path.exists(download_path):
            return False
        
        file_size = os.path.getsize(download_path)
        if file_size == 0:
            return False
        
        # Verify format
        if expected_format.lower() in download.suggested_filename.lower():
            return True
        
        return False
    
    @staticmethod
    async def simulate_user_typing(page: Page, selector: str, text: str, typing_speed: float = 0.1):
        """Simulate realistic user typing."""
        input_element = page.locator(selector)
        await input_element.click()
        
        for char in text:
            await input_element.type(char)
            await asyncio.sleep(typing_speed)
    
    @staticmethod
    async def check_accessibility(page: Page) -> Dict[str, Any]:
        """Basic accessibility checks."""
        results = {
            "has_alt_text": True,
            "has_aria_labels": True,
            "keyboard_navigable": True,
            "color_contrast": True
        }
        
        # Check for images without alt text
        images = await page.locator('img:not([alt])').count()
        if images > 0:
            results["has_alt_text"] = False
        
        # Check for buttons without aria-label
        buttons = await page.locator('button:not([aria-label])').count()
        if buttons > 0:
            results["has_aria_labels"] = False
        
        # TODO: Add more comprehensive accessibility checks
        
        return results
    
    @staticmethod
    async def capture_network_activity(page: Page) -> List[Dict[str, Any]]:
        """Capture network activity during test."""
        network_logs = []
        
        def log_request(request):
            network_logs.append({
                "timestamp": datetime.now().isoformat(),
                "type": "request",
                "url": request.url,
                "method": request.method,
                "headers": dict(request.headers)
            })
        
        def log_response(response):
            network_logs.append({
                "timestamp": datetime.now().isoformat(),
                "type": "response",
                "url": response.url,
                "status": response.status,
                "headers": dict(response.headers)
            })
        
        page.on("request", log_request)
        page.on("response", log_response)
        
        return network_logs
    
    @staticmethod
    async def simulate_slow_network(page: Page, download_speed: int = 50, upload_speed: int = 20):
        """Simulate slow network conditions."""
        # Create Chrome DevTools Protocol session
        client = await page.context.new_cdp_session(page)
        
        # Set network conditions
        await client.send('Network.emulateNetworkConditions', {
            'offline': False,
            'downloadThroughput': download_speed * 1024,  # Convert to bytes/sec
            'uploadThroughput': upload_speed * 1024,
            'latency': 100  # 100ms latency
        })
    
    @staticmethod
    async def check_memory_usage(page: Page) -> Dict[str, float]:
        """Check memory usage of the page."""
        metrics = await page.evaluate('''() => {
            if (performance.memory) {
                return {
                    usedJSHeapSize: performance.memory.usedJSHeapSize / 1048576,  // Convert to MB
                    totalJSHeapSize: performance.memory.totalJSHeapSize / 1048576,
                    jsHeapSizeLimit: performance.memory.jsHeapSizeLimit / 1048576
                };
            }
            return null;
        }''')
        
        return metrics or {}
    
    @staticmethod
    def validate_test_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate test data quality."""
        validation_results = {
            "is_valid": True,
            "issues": [],
            "stats": {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": df.isnull().sum().sum(),
                "duplicate_rows": df.duplicated().sum()
            }
        }
        
        # Check for empty dataframe
        if df.empty:
            validation_results["is_valid"] = False
            validation_results["issues"].append("DataFrame is empty")
        
        # Check for too many missing values
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_percentage > 50:
            validation_results["is_valid"] = False
            validation_results["issues"].append(f"Too many missing values: {missing_percentage:.2f}%")
        
        return validation_results
    
    @staticmethod
    async def wait_for_streamlit_rerun(page: Page, timeout: int = 5000):
        """Wait for Streamlit to complete a rerun."""
        # Streamlit adds a specific class during rerun
        await page.wait_for_function(
            "document.querySelector('.stApp').classList.contains('st-emotion-cache-running') === false",
            timeout=timeout
        )
    
    @staticmethod
    async def get_console_errors(page: Page) -> List[str]:
        """Get console errors from the page."""
        console_errors = []
        
        def handle_console_message(msg):
            if msg.type == "error":
                console_errors.append({
                    "text": msg.text,
                    "location": msg.location
                })
        
        page.on("console", handle_console_message)
        return console_errors


class SecurityTestHelpers:
    """Helper functions for security testing."""
    
    @staticmethod
    def generate_xss_payloads() -> List[str]:
        """Generate common XSS test payloads."""
        return [
            '<script>alert("XSS")</script>',
            '<img src=x onerror=alert("XSS")>',
            'javascript:alert("XSS")',
            '<svg onload=alert("XSS")>',
            '"><script>alert("XSS")</script>',
            '<iframe src="javascript:alert(\'XSS\')"></iframe>'
        ]
    
    @staticmethod
    def generate_sql_injection_payloads() -> List[str]:
        """Generate common SQL injection test payloads."""
        return [
            "' OR '1'='1",
            "1; DROP TABLE users--",
            "admin'--",
            "1' UNION SELECT NULL--",
            "' OR 1=1--",
            "'; DELETE FROM data WHERE '1'='1"
        ]
    
    @staticmethod
    def generate_malicious_filenames() -> List[str]:
        """Generate malicious filename test cases."""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file.csv.exe",
            "file\x00.csv",
            "file|command.csv",
            "file;rm -rf /.csv"
        ]
    
    @staticmethod
    async def check_csrf_protection(page: Page) -> bool:
        """Check if CSRF protection is implemented."""
        # Look for CSRF tokens in forms
        csrf_tokens = await page.locator('input[name*="csrf"], meta[name*="csrf"]').count()
        return csrf_tokens > 0
    
    @staticmethod
    async def check_secure_headers(page: Page) -> Dict[str, bool]:
        """Check for security headers."""
        response = await page.goto(page.url)
        headers = response.headers
        
        security_checks = {
            "x-frame-options": "x-frame-options" in headers,
            "x-content-type-options": "x-content-type-options" in headers,
            "strict-transport-security": "strict-transport-security" in headers,
            "content-security-policy": "content-security-policy" in headers,
            "x-xss-protection": "x-xss-protection" in headers
        }
        
        return security_checks


class PerformanceTestHelpers:
    """Helper functions for performance testing."""
    
    @staticmethod
    async def measure_page_load_time(page: Page) -> Dict[str, float]:
        """Measure various page load metrics."""
        metrics = await page.evaluate('''() => {
            const timing = performance.timing;
            return {
                domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                loadComplete: timing.loadEventEnd - timing.navigationStart,
                firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0,
                firstContentfulPaint: performance.getEntriesByType('paint')[1]?.startTime || 0
            };
        }''')
        
        return metrics
    
    @staticmethod
    async def simulate_concurrent_users(base_url: str, num_users: int, test_scenario):
        """Simulate multiple concurrent users."""
        tasks = []
        
        async def user_session(user_id: int):
            # Each user gets their own browser context
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                try:
                    await page.goto(base_url)
                    result = await test_scenario(page, user_id)
                    return {"user_id": user_id, "status": "success", "result": result}
                except Exception as e:
                    return {"user_id": user_id, "status": "error", "error": str(e)}
                finally:
                    await browser.close()
        
        # Launch concurrent user sessions
        for i in range(num_users):
            tasks.append(user_session(i))
        
        results = await asyncio.gather(*tasks)
        return results
    
    @staticmethod
    def calculate_performance_score(metrics: Dict[str, float]) -> float:
        """Calculate overall performance score based on metrics."""
        # Define weights and thresholds
        criteria = {
            "page_load": {"weight": 0.3, "threshold": 3000, "value": metrics.get("page_load", 0)},
            "file_upload": {"weight": 0.2, "threshold": 10000, "value": metrics.get("file_upload", 0)},
            "analysis_time": {"weight": 0.3, "threshold": 30000, "value": metrics.get("analysis_time", 0)},
            "memory_usage": {"weight": 0.2, "threshold": 500, "value": metrics.get("memory_usage", 0)}
        }
        
        score = 0
        for criterion, data in criteria.items():
            if data["value"] <= data["threshold"]:
                score += data["weight"] * 100
            else:
                # Linear degradation after threshold
                degradation = min((data["value"] / data["threshold"] - 1) * 50, 50)
                score += data["weight"] * (100 - degradation)
        
        return round(score, 2)