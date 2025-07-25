"""
E2E Test Configuration and Fixtures for Cherry AI Streamlit Platform

This module provides test fixtures, configuration, and utilities for E2E testing.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from typing import Generator, Dict, Any, List
from pathlib import Path
import pandas as pd
import numpy as np
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import httpx
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "base_url": "http://localhost:8501",
    "agents": {
        "data_cleaning": {"port": 8306, "name": "Data Cleaning Agent"},
        "data_loader": {"port": 8307, "name": "Data Loader Agent"},
        "visualization": {"port": 8308, "name": "Data Visualization Agent"},
        "wrangling": {"port": 8309, "name": "Data Wrangling Agent"},
        "feature_engineering": {"port": 8310, "name": "Feature Engineering Agent"},
        "sql_database": {"port": 8311, "name": "SQL Database Agent"},
        "eda_tools": {"port": 8312, "name": "EDA Tools Agent"},
        "h2o_ml": {"port": 8313, "name": "H2O ML Agent"},
        "mlflow": {"port": 8314, "name": "MLflow Tools Agent"},
        "pandas_hub": {"port": 8315, "name": "Pandas Hub Agent"}
    },
    "timeouts": {
        "page_load": 30000,  # 30 seconds
        "file_upload": 60000,  # 60 seconds
        "analysis": 120000,  # 2 minutes
        "agent_response": 30000  # 30 seconds
    },
    "test_data_dir": "tests/e2e/test_data",
    "screenshots_dir": "tests/e2e/screenshots",
    "videos_dir": "tests/e2e/videos"
}

# Ensure directories exist
for dir_path in [TEST_CONFIG["test_data_dir"], TEST_CONFIG["screenshots_dir"], TEST_CONFIG["videos_dir"]]:
    os.makedirs(dir_path, exist_ok=True)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def browser():
    """Launch browser for testing."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # Set to True for CI/CD
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        yield browser
        await browser.close()


@pytest.fixture(scope="function")
async def context(browser: Browser):
    """Create a new browser context for each test."""
    context = await browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        record_video_dir=TEST_CONFIG["videos_dir"],
        locale='en-US',
        timezone_id='Asia/Seoul'
    )
    yield context
    await context.close()


@pytest.fixture(scope="function")
async def page(context: BrowserContext):
    """Create a new page for each test."""
    page = await context.new_page()
    page.set_default_timeout(TEST_CONFIG["timeouts"]["page_load"])
    yield page
    
    # Take screenshot on test completion
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    await page.screenshot(path=f"{TEST_CONFIG['screenshots_dir']}/test_{timestamp}.png")
    
    await page.close()


@pytest.fixture(scope="session")
def test_data_generator():
    """Generate various test datasets."""
    class TestDataGenerator:
        @staticmethod
        def create_simple_csv(rows: int = 100, columns: List[str] = None) -> pd.DataFrame:
            """Create a simple CSV dataset."""
            if columns is None:
                columns = ['id', 'name', 'age', 'city', 'salary']
            
            data = {
                'id': range(1, rows + 1),
                'name': [f'Person_{i}' for i in range(1, rows + 1)],
                'age': np.random.randint(20, 65, size=rows),
                'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], size=rows),
                'salary': np.random.randint(30000, 120000, size=rows)
            }
            
            return pd.DataFrame(data)
        
        @staticmethod
        def create_timeseries_data(days: int = 30) -> pd.DataFrame:
            """Create time series data."""
            dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
            data = {
                'date': dates,
                'sales': np.random.randint(100, 1000, size=days),
                'customers': np.random.randint(50, 500, size=days),
                'revenue': np.random.uniform(5000, 50000, size=days)
            }
            return pd.DataFrame(data)
        
        @staticmethod
        def create_multi_dataset_scenario() -> Dict[str, pd.DataFrame]:
            """Create multiple related datasets."""
            # Customer data
            customers = pd.DataFrame({
                'customer_id': range(1, 101),
                'name': [f'Customer_{i}' for i in range(1, 101)],
                'age': np.random.randint(18, 70, size=100),
                'region': np.random.choice(['North', 'South', 'East', 'West'], size=100)
            })
            
            # Order data
            orders = pd.DataFrame({
                'order_id': range(1, 501),
                'customer_id': np.random.randint(1, 101, size=500),
                'product_id': np.random.randint(1, 51, size=500),
                'quantity': np.random.randint(1, 10, size=500),
                'order_date': pd.date_range(start='2024-01-01', periods=500, freq='H')
            })
            
            # Product data
            products = pd.DataFrame({
                'product_id': range(1, 51),
                'product_name': [f'Product_{i}' for i in range(1, 51)],
                'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], size=50),
                'price': np.random.uniform(10, 1000, size=50)
            })
            
            return {
                'customers.csv': customers,
                'orders.csv': orders,
                'products.csv': products
            }
        
        @staticmethod
        def create_malformed_csv() -> str:
            """Create a malformed CSV for error testing."""
            return """id,name,age,city
1,John,25,Seoul
2,Jane,30
3,Bob,35,Busan,Extra
4,Alice,forty,Incheon
"""
        
        @staticmethod
        def save_test_files(output_dir: str = TEST_CONFIG["test_data_dir"]):
            """Save all test files to disk."""
            # Simple dataset
            simple_df = TestDataGenerator.create_simple_csv()
            simple_df.to_csv(f"{output_dir}/simple_data.csv", index=False)
            
            # Time series
            ts_df = TestDataGenerator.create_timeseries_data()
            ts_df.to_csv(f"{output_dir}/timeseries_data.csv", index=False)
            
            # Multi-dataset
            multi_data = TestDataGenerator.create_multi_dataset_scenario()
            for filename, df in multi_data.items():
                df.to_csv(f"{output_dir}/{filename}", index=False)
            
            # Malformed data
            with open(f"{output_dir}/malformed_data.csv", 'w') as f:
                f.write(TestDataGenerator.create_malformed_csv())
            
            # Large dataset (5MB)
            large_df = TestDataGenerator.create_simple_csv(rows=100000)
            large_df.to_csv(f"{output_dir}/large_data.csv", index=False)
            
            logger.info(f"Test data files created in {output_dir}")
    
    return TestDataGenerator


@pytest.fixture(scope="session")
def agent_health_checker():
    """Check health of A2A agents."""
    class AgentHealthChecker:
        @staticmethod
        async def check_agent_health(port: int) -> bool:
            """Check if an agent is healthy."""
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://localhost:{port}/.well-known/agent.json",
                        timeout=5.0
                    )
                    return response.status_code == 200
            except Exception as e:
                logger.error(f"Agent on port {port} is not healthy: {e}")
                return False
        
        @staticmethod
        async def check_all_agents() -> Dict[str, bool]:
            """Check health of all agents."""
            health_status = {}
            for agent_name, agent_info in TEST_CONFIG["agents"].items():
                is_healthy = await AgentHealthChecker.check_agent_health(agent_info["port"])
                health_status[agent_name] = is_healthy
                logger.info(f"{agent_info['name']} (port {agent_info['port']}): {'✓' if is_healthy else '✗'}")
            return health_status
        
        @staticmethod
        async def wait_for_agents(timeout: int = 60) -> bool:
            """Wait for all agents to be ready."""
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                health_status = await AgentHealthChecker.check_all_agents()
                if all(health_status.values()):
                    logger.info("All agents are ready!")
                    return True
                await asyncio.sleep(5)
            
            logger.error("Timeout waiting for agents to be ready")
            return False
    
    return AgentHealthChecker


@pytest.fixture
async def streamlit_app(page: Page):
    """Navigate to Streamlit app and wait for it to load."""
    await page.goto(TEST_CONFIG["base_url"])
    
    # Wait for Streamlit to fully load
    await page.wait_for_selector('div[data-testid="stApp"]', timeout=TEST_CONFIG["timeouts"]["page_load"])
    
    # Wait for specific Cherry AI elements
    await page.wait_for_selector('text=Cherry AI Data Science Platform', timeout=10000)
    
    return page


@pytest.fixture
def screenshot_on_failure(request, page: Page):
    """Take screenshot on test failure."""
    yield
    if request.node.rep_call.failed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = request.node.nodeid.replace("/", "_").replace("::", "_")
        screenshot_path = f"{TEST_CONFIG['screenshots_dir']}/failure_{test_name}_{timestamp}.png"
        page.screenshot(path=screenshot_path)
        logger.error(f"Test failed. Screenshot saved to: {screenshot_path}")


@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = []
        
        def record_metric(self, metric_name: str, value: float, unit: str = "ms"):
            """Record a performance metric."""
            self.metrics.append({
                "timestamp": datetime.now().isoformat(),
                "metric": metric_name,
                "value": value,
                "unit": unit
            })
        
        def get_summary(self) -> Dict[str, Any]:
            """Get performance summary."""
            if not self.metrics:
                return {}
            
            summary = {}
            metric_names = set(m["metric"] for m in self.metrics)
            
            for metric_name in metric_names:
                values = [m["value"] for m in self.metrics if m["metric"] == metric_name]
                summary[metric_name] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "count": len(values)
                }
            
            return summary
        
        def save_report(self, filepath: str):
            """Save performance report."""
            report = {
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics,
                "summary": self.get_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
    
    return PerformanceMonitor()


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "ui: UI-specific tests")
    config.addinivalue_line("markers", "agent: Agent collaboration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test results available to fixtures."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)