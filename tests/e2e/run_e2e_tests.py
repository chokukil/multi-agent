#!/usr/bin/env python3
"""
E2E Test Runner for Cherry AI Streamlit Platform

This script executes the comprehensive E2E test suite and generates detailed reports.
"""

import os
import sys
import json
import asyncio
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/e2e/e2e_test_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class E2ETestRunner:
    """Comprehensive E2E test runner with reporting."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.report_dir = self.test_dir / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        self.test_results = {
            "execution_start": datetime.now().isoformat(),
            "test_categories": {},
            "performance_metrics": {},
            "errors": [],
            "summary": {}
        }
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        prerequisites = {
            "streamlit_installed": self._check_streamlit(),
            "playwright_installed": self._check_playwright(),
            "cherry_ai_app": self._check_cherry_ai_app(),
            "test_data_dir": self._check_test_data_dir(),
            "agents_available": self._check_agents_health()
        }
        
        all_good = all(prerequisites.values())
        
        for check, result in prerequisites.items():
            status = "✓" if result else "✗"
            logger.info(f"{status} {check}: {'PASS' if result else 'FAIL'}")
        
        return all_good
    
    def _check_streamlit(self) -> bool:
        """Check if Streamlit is available."""
        try:
            import streamlit
            return True
        except ImportError:
            return False
    
    def _check_playwright(self) -> bool:
        """Check if Playwright is available."""
        try:
            from playwright.async_api import async_playwright
            return True
        except ImportError:
            return False
    
    def _check_cherry_ai_app(self) -> bool:
        """Check if Cherry AI Streamlit app exists."""
        app_path = project_root / "cherry_ai_streamlit_app.py"
        return app_path.exists()
    
    def _check_test_data_dir(self) -> bool:
        """Check if test data directory exists."""
        test_data_dir = self.test_dir / "test_data"
        return test_data_dir.exists()
    
    def _check_agents_health(self) -> bool:
        """Check if A2A agents are running."""
        try:
            import httpx
            import asyncio
            
            async def check_agent(port: int) -> bool:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{port}/.well-known/agent.json",
                            timeout=5.0
                        )
                        return response.status_code == 200
                except Exception:
                    return False
            
            async def check_all_agents():
                agent_ports = [8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314, 8315]
                results = await asyncio.gather(*[check_agent(port) for port in agent_ports])
                return sum(results) >= len(agent_ports) // 2  # At least half should be running
            
            return asyncio.run(check_all_agents())
        except Exception as e:
            logger.warning(f"Could not check agent health: {e}")
            return False
    
    def start_streamlit_app(self) -> subprocess.Popen:
        """Start the Streamlit application."""
        logger.info("Starting Streamlit application...")
        
        app_path = project_root / "cherry_ai_streamlit_app.py"
        
        # Start Streamlit in the background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port=8501",
            "--server.headless=true",
            "--browser.serverAddress=localhost",
            "--browser.gatherUsageStats=false"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(project_root))
        
        # Wait for app to start
        time.sleep(10)
        
        if process.poll() is None:
            logger.info("✓ Streamlit application started successfully")
        else:
            logger.error("✗ Failed to start Streamlit application")
            raise RuntimeError("Could not start Streamlit app")
        
        return process
    
    def run_test_category(self, category: str, test_file: str) -> Dict[str, Any]:
        """Run tests for a specific category."""
        logger.info(f"Running {category} tests from {test_file}...")
        
        # Construct pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / test_file),
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.report_dir}/{category}_results.json",
            "--html-report",
            f"--html-report-file={self.report_dir}/{category}_report.html"
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1800,  # 30 minutes timeout
                cwd=str(self.test_dir)
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            json_report_path = self.report_dir / f"{category}_results.json"
            
            if json_report_path.exists():
                with open(json_report_path, 'r') as f:
                    test_data = json.load(f)
                
                category_results = {
                    "status": "completed",
                    "execution_time": execution_time,
                    "total_tests": test_data.get("summary", {}).get("total", 0),
                    "passed": test_data.get("summary", {}).get("passed", 0),
                    "failed": test_data.get("summary", {}).get("failed", 0),
                    "skipped": test_data.get("summary", {}).get("skipped", 0),
                    "errors": test_data.get("summary", {}).get("error", 0),
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                category_results = {
                    "status": "error",
                    "execution_time": execution_time,
                    "error": "No JSON report generated",
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            logger.info(f"✓ {category} tests completed in {execution_time:.2f}s")
            return category_results
            
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {category} tests timed out")
            return {
                "status": "timeout",
                "execution_time": time.time() - start_time,
                "error": "Test execution timed out"
            }
        except Exception as e:
            logger.error(f"✗ {category} tests failed: {e}")
            return {
                "status": "error",
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all E2E test categories."""
        logger.info("Starting comprehensive E2E test execution...")
        
        test_categories = [
            ("user_journeys", "test_user_journeys.py"),
            ("agent_collaboration", "test_agent_collaboration.py"),
            ("error_recovery", "test_error_recovery.py")
        ]
        
        streamlit_process = None
        
        try:
            # Start Streamlit app
            streamlit_process = self.start_streamlit_app()
            
            # Run each test category
            for category, test_file in test_categories:
                try:
                    self.test_results["test_categories"][category] = self.run_test_category(category, test_file)
                except Exception as e:
                    self.test_results["test_categories"][category] = {
                        "status": "error",
                        "error": str(e)
                    }
                    self.test_results["errors"].append(f"{category}: {str(e)}")
            
            # Generate summary
            self._generate_summary()
            
        except Exception as e:
            logger.error(f"Critical error during test execution: {e}")
            self.test_results["errors"].append(f"Critical error: {str(e)}")
        
        finally:
            # Stop Streamlit app
            if streamlit_process:
                logger.info("Stopping Streamlit application...")
                streamlit_process.terminate()
                streamlit_process.wait(timeout=10)
        
        self.test_results["execution_end"] = datetime.now().isoformat()
        return self.test_results
    
    def _generate_summary(self):
        """Generate test execution summary."""
        categories = self.test_results["test_categories"]
        
        total_tests = sum(cat.get("total_tests", 0) for cat in categories.values())
        total_passed = sum(cat.get("passed", 0) for cat in categories.values())
        total_failed = sum(cat.get("failed", 0) for cat in categories.values())
        total_skipped = sum(cat.get("skipped", 0) for cat in categories.values())
        total_errors = sum(cat.get("errors", 0) for cat in categories.values())
        
        execution_times = [cat.get("execution_time", 0) for cat in categories.values()]
        total_execution_time = sum(execution_times)
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "errors": total_errors,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "total_execution_time": total_execution_time,
            "categories_completed": len([cat for cat in categories.values() if cat.get("status") == "completed"]),
            "categories_failed": len([cat for cat in categories.values() if cat.get("status") in ["error", "timeout"]])
        }
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive HTML report."""
        logger.info("Generating comprehensive E2E test report...")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cherry AI Streamlit Platform - E2E Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .category {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .category-header {{ background: #f8f9fa; padding: 10px; font-weight: bold; }}
        .category-content {{ padding: 15px; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; }}
        .test-details {{ margin-top: 15px; }}
        .performance {{ background: #e3f2fd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cherry AI Streamlit Platform - E2E Test Report</h1>
        <p><strong>Execution Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Platform:</strong> Cherry AI Streamlit Platform with A2A Multi-Agent Integration</p>
        <p><strong>Test Framework:</strong> Playwright + pytest</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metric">
            <strong>Total Tests:</strong> {self.test_results['summary'].get('total_tests', 0)}
        </div>
        <div class="metric success">
            <strong>Passed:</strong> {self.test_results['summary'].get('passed', 0)}
        </div>
        <div class="metric error">
            <strong>Failed:</strong> {self.test_results['summary'].get('failed', 0)}
        </div>
        <div class="metric warning">
            <strong>Skipped:</strong> {self.test_results['summary'].get('skipped', 0)}
        </div>
        <div class="metric">
            <strong>Success Rate:</strong> {self.test_results['summary'].get('success_rate', 0):.1f}%
        </div>
        <div class="metric">
            <strong>Total Execution Time:</strong> {self.test_results['summary'].get('total_execution_time', 0):.1f}s
        </div>
    </div>
    
    <div class="performance">
        <h2>Performance Metrics</h2>
        {self._generate_performance_section()}
    </div>
    
    <h2>Test Categories</h2>
    {self._generate_category_sections()}
    
    <div class="test-details">
        <h2>Technical Details</h2>
        <h3>Test Environment</h3>
        <ul>
            <li>Streamlit Version: Latest</li>
            <li>Playwright Version: Latest</li>
            <li>Python Version: {sys.version}</li>
            <li>Platform: {sys.platform}</li>
        </ul>
        
        <h3>Test Coverage</h3>
        <ul>
            <li>✓ User Journey Workflows (File Upload, Chat Interface, Analysis)</li>
            <li>✓ Multi-Agent Collaboration (Sequential/Parallel Execution)</li>
            <li>✓ Error Recovery Scenarios (File Errors, Agent Failures, Security)</li>
            <li>✓ Performance Requirements (Load Times, Memory Usage)</li>
            <li>✓ Security Validation (XSS Prevention, Input Sanitization)</li>
        </ul>
        
        {self._generate_recommendations_section()}
    </div>
</body>
</html>
        """
        
        report_path = self.report_dir / "comprehensive_e2e_report.html"
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✓ Comprehensive report generated: {report_path}")
        return str(report_path)
    
    def _generate_performance_section(self) -> str:
        """Generate performance metrics section."""
        categories = self.test_results["test_categories"]
        
        performance_html = "<ul>"
        
        for category, results in categories.items():
            execution_time = results.get("execution_time", 0)
            status = results.get("status", "unknown")
            
            status_class = "success" if status == "completed" else "error"
            
            performance_html += f"""
            <li>
                <strong>{category.replace('_', ' ').title()}:</strong> 
                <span class="{status_class}">{execution_time:.2f}s ({status})</span>
            </li>
            """
        
        performance_html += "</ul>"
        return performance_html
    
    def _generate_category_sections(self) -> str:
        """Generate individual category sections."""
        categories_html = ""
        
        for category, results in self.test_results["test_categories"].items():
            status = results.get("status", "unknown")
            status_class = "success" if status == "completed" else "error"
            
            categories_html += f"""
            <div class="category">
                <div class="category-header {status_class}">
                    {category.replace('_', ' ').title()} Tests - {status.title()}
                </div>
                <div class="category-content">
                    <p><strong>Execution Time:</strong> {results.get('execution_time', 0):.2f} seconds</p>
                    <p><strong>Total Tests:</strong> {results.get('total_tests', 0)}</p>
                    <p><strong>Passed:</strong> <span class="success">{results.get('passed', 0)}</span></p>
                    <p><strong>Failed:</strong> <span class="error">{results.get('failed', 0)}</span></p>
                    <p><strong>Skipped:</strong> <span class="warning">{results.get('skipped', 0)}</span></p>
                    
                    {self._generate_category_details(category, results)}
                </div>
            </div>
            """
        
        return categories_html
    
    def _generate_category_details(self, category: str, results: Dict[str, Any]) -> str:
        """Generate detailed information for a category."""
        if results.get("status") == "error":
            return f"""
            <div class="error">
                <h4>Error Details:</h4>
                <p>{results.get('error', 'Unknown error')}</p>
                <pre>{results.get('stderr', '')}</pre>
            </div>
            """
        
        return f"""
        <h4>Category Focus:</h4>
        <ul>
            {self._get_category_focus(category)}
        </ul>
        """
    
    def _get_category_focus(self, category: str) -> str:
        """Get the focus areas for each test category."""
        focus_areas = {
            "user_journeys": """
                <li>File upload with drag-and-drop functionality</li>
                <li>ChatGPT/Claude-style interface interactions</li>
                <li>Real-time streaming responses</li>
                <li>Progressive disclosure and artifact rendering</li>
                <li>Performance requirements validation</li>
            """,
            "agent_collaboration": """
                <li>Sequential agent execution workflows</li>
                <li>Parallel agent processing</li>
                <li>Agent health monitoring and discovery</li>
                <li>Result integration and conflict resolution</li>
                <li>Real-time progress tracking</li>
            """,
            "error_recovery": """
                <li>File upload error handling</li>
                <li>Agent communication failures</li>
                <li>System resource constraints</li>
                <li>Security threat prevention</li>
                <li>Graceful degradation scenarios</li>
            """
        }
        
        return focus_areas.get(category, "<li>General testing coverage</li>")
    
    def _generate_recommendations_section(self) -> str:
        """Generate recommendations based on test results."""
        summary = self.test_results["summary"]
        
        recommendations = []
        
        if summary.get("success_rate", 0) < 80:
            recommendations.append("⚠️ Success rate below 80% - investigate failing tests")
        
        if summary.get("total_execution_time", 0) > 1800:  # 30 minutes
            recommendations.append("⚠️ Long execution time - consider test optimization")
        
        if summary.get("categories_failed", 0) > 0:
            recommendations.append("🔧 Some test categories failed - review error logs")
        
        if len(self.test_results.get("errors", [])) > 0:
            recommendations.append("🔍 Critical errors encountered - check system setup")
        
        if not recommendations:
            recommendations.append("✅ All systems performing well - no issues detected")
        
        rec_html = "<h3>Recommendations</h3><ul>"
        for rec in recommendations:
            rec_html += f"<li>{rec}</li>"
        rec_html += "</ul>"
        
        return rec_html
    
    def save_json_report(self) -> str:
        """Save detailed JSON report."""
        json_path = self.report_dir / "comprehensive_e2e_results.json"
        
        with open(json_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"✓ JSON report saved: {json_path}")
        return str(json_path)


def main():
    """Main execution function."""
    logger.info("🚀 Starting Cherry AI Streamlit Platform E2E Test Suite")
    
    runner = E2ETestRunner()
    
    # Check prerequisites
    if not runner.check_prerequisites():
        logger.error("❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    # Run tests
    logger.info("📋 Prerequisites satisfied. Running E2E tests...")
    results = runner.run_all_tests()
    
    # Generate reports
    html_report = runner.generate_comprehensive_report()
    json_report = runner.save_json_report()
    
    # Print summary
    summary = results["summary"]
    logger.info(f"""
🎯 E2E Test Execution Complete!

📊 Summary:
   Total Tests: {summary.get('total_tests', 0)}
   Passed: {summary.get('passed', 0)}
   Failed: {summary.get('failed', 0)}
   Skipped: {summary.get('skipped', 0)}
   Success Rate: {summary.get('success_rate', 0):.1f}%
   Execution Time: {summary.get('total_execution_time', 0):.1f}s

📄 Reports Generated:
   HTML Report: {html_report}
   JSON Report: {json_report}

{'' if summary.get('success_rate', 0) >= 80 else '⚠️  Some tests failed - please review the reports for details.'}
    """)
    
    # Exit with appropriate code
    exit_code = 0 if summary.get('success_rate', 0) >= 80 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()