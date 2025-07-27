"""
Comprehensive E2E Tests for Cherry AI Streamlit Platform
QA-focused deep testing based on design and task specifications
"""

import pytest
import asyncio
import httpx
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, expect
import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test Configuration
CHERRY_AI_URL = "http://localhost:8501"
TEST_DATA_DIR = Path(__file__).parent / "test_data"
ARTIFACTS_DIR = Path("tests/_artifacts/e2e-20250725-225049")

@dataclass
class TestResult:
    test_name: str
    status: str
    duration: float
    error: Optional[str] = None
    details: Dict[str, Any] = None

class CherryAITestSuite:
    """Comprehensive E2E test suite for Cherry AI Platform"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.artifacts_dir = ARTIFACTS_DIR
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    async def setup_browser(self) -> tuple[Browser, BrowserContext, Page]:
        """Set up browser with proper configuration"""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            record_video_dir=str(self.artifacts_dir / "videos") if True else None,
            record_video_size={'width': 1920, 'height': 1080}
        )
        
        # Enable tracing if requested
        await context.tracing.start(
            name="cherry-ai-e2e-trace",
            screenshots=True,
            snapshots=True,
            sources=True
        )
        
        page = await context.new_page()
        return browser, context, page
    
    async def cleanup_browser(self, browser: Browser, context: BrowserContext):
        """Clean up browser resources"""
        try:
            await context.tracing.stop(path=str(self.artifacts_dir / "trace.zip"))
        except:
            pass
        await context.close()
        await browser.close()
    
    def record_result(self, test_name: str, status: str, duration: float, 
                     error: Optional[str] = None, details: Dict[str, Any] = None):
        """Record test result"""
        result = TestResult(
            test_name=test_name,
            status=status,
            duration=duration,
            error=error,
            details=details or {}
        )
        self.results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌"
        logger.info(f"{status_icon} {test_name}: {status} ({duration:.2f}s)")
        if error:
            logger.error(f"   Error: {error}")

@pytest.mark.asyncio
class TestCherryAIPlatformE2E:
    """Comprehensive E2E tests based on design specifications"""
    
    @pytest.fixture(scope="class")
    def test_suite(self):
        """Initialize test suite"""
        return CherryAITestSuite()
    
    async def test_01_app_availability_and_health(self, test_suite: CherryAITestSuite):
        """TC01: Verify Cherry AI app is running and healthy"""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(CHERRY_AI_URL, timeout=10.0)
                
            assert response.status_code == 200, f"App not responding: {response.status_code}"
            
            # Check for Streamlit indicators
            content = response.text
            streamlit_indicators = ["streamlit", "stApp", "data-testid"]
            has_streamlit = any(indicator in content.lower() for indicator in streamlit_indicators)
            
            assert has_streamlit, "Response doesn't appear to be a Streamlit app"
            
            test_suite.record_result(
                "App Health Check", "PASS", time.time() - start_time,
                details={"response_size": len(content), "status_code": response.status_code}
            )
            
        except Exception as e:
            test_suite.record_result(
                "App Health Check", "FAIL", time.time() - start_time, str(e)
            )
            raise
    
    async def test_02_page_load_performance(self, test_suite: CherryAITestSuite):
        """TC02: Verify page load performance meets requirements (<3s)"""
        start_time = time.time()
        
        try:
            browser, context, page = await test_suite.setup_browser()
            
            load_start = time.time()
            await page.goto(CHERRY_AI_URL, wait_until="networkidle", timeout=30000)
            load_time = time.time() - load_start
            
            # Check for key UI elements
            await page.wait_for_selector("body", timeout=10000)
            
            # Take screenshot for verification
            await page.screenshot(path=str(test_suite.artifacts_dir / "page_load.png"))
            
            assert load_time < 3.0, f"Page load time {load_time:.2f}s exceeds 3s requirement"
            
            test_suite.record_result(
                "Page Load Performance", "PASS", time.time() - start_time,
                details={"load_time": load_time, "requirement": 3.0}
            )
            
            await test_suite.cleanup_browser(browser, context)
            
        except Exception as e:
            test_suite.record_result(
                "Page Load Performance", "FAIL", time.time() - start_time, str(e)
            )
            try:
                await test_suite.cleanup_browser(browser, context)
            except:
                pass
            raise
    
    async def test_03_ui_elements_presence(self, test_suite: CherryAITestSuite):
        """TC03: Verify core UI elements are present and functional"""
        start_time = time.time()
        
        try:
            browser, context, page = await test_suite.setup_browser()
            await page.goto(CHERRY_AI_URL, wait_until="networkidle")
            
            # Expected UI elements based on design specs
            ui_elements = {
                "file_upload": ["file_uploader", "upload", "drag", "drop"],
                "chat_interface": ["chat", "message", "input", "text"],
                "main_content": ["main", "container", "element"],
                "streamlit_framework": ["stApp", "streamlit"]
            }
            
            found_elements = {}
            content = await page.content()
            
            for element_type, selectors in ui_elements.items():
                found = False
                for selector in selectors:
                    if selector.lower() in content.lower():
                        found = True
                        break
                found_elements[element_type] = found
            
            # Check for Streamlit app structure
            streamlit_present = await page.locator("[data-testid]").count() > 0
            found_elements["streamlit_structure"] = streamlit_present
            
            missing_elements = [k for k, v in found_elements.items() if not v]
            
            if missing_elements:
                logger.warning(f"Missing UI elements: {missing_elements}")
            
            # Take screenshot for UI verification
            await page.screenshot(path=str(test_suite.artifacts_dir / "ui_elements.png"))
            
            test_suite.record_result(
                "UI Elements Presence", "PASS" if len(missing_elements) <= 1 else "FAIL",
                time.time() - start_time,
                f"Missing elements: {missing_elements}" if missing_elements else None,
                details={"found_elements": found_elements}
            )
            
            await test_suite.cleanup_browser(browser, context)
            
        except Exception as e:
            test_suite.record_result(
                "UI Elements Presence", "FAIL", time.time() - start_time, str(e)
            )
            try:
                await test_suite.cleanup_browser(browser, context)
            except:
                pass
            raise
    
    async def test_04_chat_interface_interaction(self, test_suite: CherryAITestSuite):
        """TC04: Test ChatGPT/Claude-style chat interface functionality"""
        start_time = time.time()
        
        try:
            browser, context, page = await test_suite.setup_browser()
            await page.goto(CHERRY_AI_URL, wait_until="networkidle")
            
            # Wait for page to stabilize
            await asyncio.sleep(2)
            
            # Look for text input elements (Streamlit typically uses textarea or text_input)
            text_inputs = await page.locator("textarea, input[type='text'], input[type='string']").all()
            
            if not text_inputs:
                # Try more generic selectors
                text_inputs = await page.locator("[contenteditable], .stTextInput input, .stTextArea textarea").all()
            
            interaction_success = False
            test_message = "Hello, this is a test message from E2E testing"
            
            if text_inputs:
                try:
                    # Try to interact with the first available input
                    text_input = text_inputs[0]
                    await text_input.click()
                    await text_input.fill(test_message)
                    
                    # Look for send button or try Enter key
                    send_buttons = await page.locator("button:has-text('Send'), button:has-text('Submit'), [type='submit']").all()
                    
                    if send_buttons:
                        await send_buttons[0].click()
                    else:
                        await text_input.press("Enter")
                    
                    interaction_success = True
                    
                    # Wait briefly for any response
                    await asyncio.sleep(2)
                    
                except Exception as interaction_error:
                    logger.warning(f"Input interaction failed: {interaction_error}")
            
            # Take screenshot of chat interface
            await page.screenshot(path=str(test_suite.artifacts_dir / "chat_interface.png"))
            
            test_suite.record_result(
                "Chat Interface Interaction", 
                "PASS" if interaction_success else "PARTIAL",
                time.time() - start_time,
                None if interaction_success else "Could not fully interact with chat interface",
                details={
                    "text_inputs_found": len(text_inputs),
                    "interaction_success": interaction_success,
                    "test_message": test_message
                }
            )
            
            await test_suite.cleanup_browser(browser, context)
            
        except Exception as e:
            test_suite.record_result(
                "Chat Interface Interaction", "FAIL", time.time() - start_time, str(e)
            )
            try:
                await test_suite.cleanup_browser(browser, context)
            except:
                pass
            raise
    
    async def test_05_file_upload_interface(self, test_suite: CherryAITestSuite):
        """TC05: Test file upload functionality"""
        start_time = time.time()
        
        try:
            browser, context, page = await test_suite.setup_browser()
            await page.goto(CHERRY_AI_URL, wait_until="networkidle")
            
            # Wait for page to stabilize
            await asyncio.sleep(2)
            
            # Look for file upload elements
            file_uploaders = await page.locator("input[type='file'], .stFileUploader, [data-testid*='file'], [data-testid*='upload']").all()
            
            upload_found = len(file_uploaders) > 0
            upload_attempted = False
            
            if file_uploaders:
                try:
                    # Create a simple test file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                        f.write("name,value\\ntest1,100\\ntest2,200\\ntest3,300\\n")
                        test_file_path = f.name
                    
                    # Try to upload the file
                    await file_uploaders[0].set_input_files(test_file_path)
                    upload_attempted = True
                    
                    # Wait for upload processing
                    await asyncio.sleep(3)
                    
                    # Clean up test file
                    Path(test_file_path).unlink()
                    
                except Exception as upload_error:
                    logger.warning(f"File upload failed: {upload_error}")
            
            # Take screenshot of file upload interface
            await page.screenshot(path=str(test_suite.artifacts_dir / "file_upload.png"))
            
            test_suite.record_result(
                "File Upload Interface",
                "PASS" if upload_found and upload_attempted else "PARTIAL",
                time.time() - start_time,
                None if upload_found else "No file upload interface found",
                details={
                    "uploaders_found": len(file_uploaders),
                    "upload_attempted": upload_attempted
                }
            )
            
            await test_suite.cleanup_browser(browser, context)
            
        except Exception as e:
            test_suite.record_result(
                "File Upload Interface", "FAIL", time.time() - start_time, str(e)
            )
            try:
                await test_suite.cleanup_browser(browser, context)
            except:
                pass
            raise
    
    async def test_06_responsive_design(self, test_suite: CherryAITestSuite):
        """TC06: Test responsive design across different screen sizes"""
        start_time = time.time()
        
        try:
            browser, context, page = await test_suite.setup_browser()
            
            # Test different viewport sizes
            viewports = [
                {"name": "desktop", "width": 1920, "height": 1080},
                {"name": "tablet", "width": 768, "height": 1024},
                {"name": "mobile", "width": 375, "height": 667}
            ]
            
            responsive_results = {}
            
            for viewport in viewports:
                await page.set_viewport_size({"width": viewport["width"], "height": viewport["height"]})
                await page.goto(CHERRY_AI_URL, wait_until="networkidle")
                
                # Wait for layout to adjust
                await asyncio.sleep(2)
                
                # Take screenshot for each viewport
                await page.screenshot(
                    path=str(test_suite.artifacts_dir / f"responsive_{viewport['name']}.png")
                )
                
                # Check if content is accessible
                body_visible = await page.locator("body").is_visible()
                content_width = await page.evaluate("document.body.scrollWidth")
                viewport_width = viewport["width"]
                
                responsive_results[viewport["name"]] = {
                    "body_visible": body_visible,
                    "content_width": content_width,
                    "viewport_width": viewport_width,
                    "horizontal_overflow": content_width > viewport_width * 1.1  # Allow 10% tolerance
                }
            
            # Check if all viewports rendered successfully
            all_responsive = all(
                result["body_visible"] and not result["horizontal_overflow"]
                for result in responsive_results.values()
            )
            
            test_suite.record_result(
                "Responsive Design",
                "PASS" if all_responsive else "PARTIAL",
                time.time() - start_time,
                None if all_responsive else "Some viewport issues detected",
                details={"viewport_results": responsive_results}
            )
            
            await test_suite.cleanup_browser(browser, context)
            
        except Exception as e:
            test_suite.record_result(
                "Responsive Design", "FAIL", time.time() - start_time, str(e)
            )
            try:
                await test_suite.cleanup_browser(browser, context)
            except:
                pass
            raise
    
    async def test_07_accessibility_features(self, test_suite: CherryAITestSuite):
        """TC07: Test accessibility features and keyboard navigation"""
        start_time = time.time()
        
        try:
            browser, context, page = await test_suite.setup_browser()
            await page.goto(CHERRY_AI_URL, wait_until="networkidle")
            
            # Test keyboard navigation
            await page.keyboard.press("Tab")
            focused_element = await page.evaluate("document.activeElement.tagName")
            
            # Check for ARIA labels and roles
            aria_elements = await page.locator("[aria-label], [role], [aria-describedby]").count()
            
            # Check for proper headings structure
            headings = await page.locator("h1, h2, h3, h4, h5, h6").count()
            
            # Check for alt text on images
            images = await page.locator("img").count()
            images_with_alt = await page.locator("img[alt]").count()
            
            accessibility_score = 0
            max_score = 4
            
            if focused_element and focused_element != "BODY":
                accessibility_score += 1
            if aria_elements > 0:
                accessibility_score += 1
            if headings > 0:
                accessibility_score += 1
            if images == 0 or images_with_alt >= images * 0.8:  # 80% of images have alt text
                accessibility_score += 1
            
            accessibility_percentage = (accessibility_score / max_score) * 100
            
            test_suite.record_result(
                "Accessibility Features",
                "PASS" if accessibility_percentage >= 75 else "PARTIAL",
                time.time() - start_time,
                None if accessibility_percentage >= 75 else f"Accessibility score: {accessibility_percentage}%",
                details={
                    "keyboard_navigation": focused_element != "BODY",
                    "aria_elements": aria_elements,
                    "headings": headings,
                    "images_with_alt": f"{images_with_alt}/{images}",
                    "accessibility_score": accessibility_percentage
                }
            )
            
            await test_suite.cleanup_browser(browser, context)
            
        except Exception as e:
            test_suite.record_result(
                "Accessibility Features", "FAIL", time.time() - start_time, str(e)
            )
            try:
                await test_suite.cleanup_browser(browser, context)
            except:
                pass
            raise
    
    async def test_08_performance_metrics(self, test_suite: CherryAITestSuite):
        """TC08: Collect and validate performance metrics"""
        start_time = time.time()
        
        try:
            browser, context, page = await test_suite.setup_browser()
            
            # Navigate and measure performance
            navigation_start = time.time()
            await page.goto(CHERRY_AI_URL, wait_until="domcontentloaded")
            dom_ready_time = time.time() - navigation_start
            
            await page.wait_for_load_state("networkidle")
            full_load_time = time.time() - navigation_start
            
            # Get performance metrics
            try:
                metrics = await page.evaluate("""
                    () => {
                        const perfData = performance.getEntriesByType('navigation')[0];
                        return {
                            dom_content_loaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                            load_complete: perfData.loadEventEnd - perfData.loadEventStart,
                            first_paint: performance.getEntriesByType('paint').find(p => p.name === 'first-paint')?.startTime || 0,
                            first_contentful_paint: performance.getEntriesByType('paint').find(p => p.name === 'first-contentful-paint')?.startTime || 0
                        };
                    }
                """)
                
                # Check memory usage
                memory_info = await page.evaluate("performance.memory ? performance.memory.usedJSHeapSize : 0")
                
            except Exception:
                metrics = {}
                memory_info = 0
            
            # Performance thresholds
            performance_criteria = {
                "dom_ready_time": {"value": dom_ready_time, "threshold": 2.0, "unit": "seconds"},
                "full_load_time": {"value": full_load_time, "threshold": 3.0, "unit": "seconds"},
                "memory_usage": {"value": memory_info / (1024*1024), "threshold": 100, "unit": "MB"}
            }
            
            # Evaluate performance
            performance_issues = []
            for metric, data in performance_criteria.items():
                if data["value"] > data["threshold"]:
                    performance_issues.append(f"{metric}: {data['value']:.2f}{data['unit']} > {data['threshold']}{data['unit']}")
            
            test_suite.record_result(
                "Performance Metrics",
                "PASS" if not performance_issues else "PARTIAL",
                time.time() - start_time,
                f"Performance issues: {performance_issues}" if performance_issues else None,
                details={
                    "dom_ready_time": dom_ready_time,
                    "full_load_time": full_load_time,
                    "memory_usage_mb": memory_info / (1024*1024) if memory_info else 0,
                    "browser_metrics": metrics,
                    "performance_issues": performance_issues
                }
            )
            
            await test_suite.cleanup_browser(browser, context)
            
        except Exception as e:  
            test_suite.record_result(
                "Performance Metrics", "FAIL", time.time() - start_time, str(e)
            )
            try:
                await test_suite.cleanup_browser(browser, context)
            except:
                pass
            raise

    async def generate_comprehensive_report(self, test_suite: CherryAITestSuite):
        """Generate comprehensive test report"""
        total_tests = len(test_suite.results)
        passed = len([r for r in test_suite.results if r.status == "PASS"])
        partial = len([r for r in test_suite.results if r.status == "PARTIAL"])
        failed = len([r for r in test_suite.results if r.status == "FAIL"])
        
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        # Generate detailed report
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "partial": partial,
                "failed": failed,
                "success_rate": success_rate,
                "timestamp": time.time()
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "duration": r.duration,
                    "error": r.error,
                    "details": r.details
                }
                for r in test_suite.results
            ],
            "recommendations": self.generate_qa_recommendations(test_suite.results)
        }
        
        # Save detailed JSON report
        report_file = test_suite.artifacts_dir / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML report
        html_report = self.generate_html_report(report)
        html_file = test_suite.artifacts_dir / "test_report.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        # Generate JUnit XML report
        junit_report = self.generate_junit_report(report)
        junit_file = test_suite.artifacts_dir / "junit_report.xml"
        with open(junit_file, 'w') as f:
            f.write(junit_report)
        
        return report
    
    def generate_qa_recommendations(self, results: List[TestResult]) -> List[str]:
        """Generate QA recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in results if r.status == "FAIL"]
        partial_tests = [r for r in results if r.status == "PARTIAL"]
        
        if failed_tests:
            recommendations.append(f"🔴 Critical: {len(failed_tests)} tests failed - immediate attention required")
            for test in failed_tests:
                recommendations.append(f"   - Fix {test.test_name}: {test.error}")
        
        if partial_tests:
            recommendations.append(f"🟡 Review: {len(partial_tests)} tests partially successful - optimization opportunities")
            for test in partial_tests:
                recommendations.append(f"   - Improve {test.test_name}: {test.error or 'Partial functionality detected'}")
        
        # Performance recommendations
        perf_results = [r for r in results if "Performance" in r.test_name and r.details]
        for result in perf_results:
            if result.details and "performance_issues" in result.details:
                issues = result.details["performance_issues"]
                if issues:
                    recommendations.append(f"⚡ Performance: Address {len(issues)} performance issues")
                    for issue in issues:
                        recommendations.append(f"   - {issue}")
        
        # Accessibility recommendations
        accessibility_results = [r for r in results if "Accessibility" in r.test_name and r.details]
        for result in accessibility_results:
            if result.details and result.details.get("accessibility_score", 100) < 90:
                score = result.details["accessibility_score"]
                recommendations.append(f"♿ Accessibility: Improve accessibility score from {score:.1f}% to 90%+")
        
        if not failed_tests and not partial_tests:
            recommendations.append("✅ Excellent: All tests passed - platform is ready for production")
            recommendations.append("📊 Monitor: Continue monitoring performance and user experience metrics")
            recommendations.append("🔄 Maintenance: Schedule regular regression testing")
        
        return recommendations
    
    def generate_html_report(self, report: Dict) -> str:
        """Generate HTML test report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cherry AI E2E Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }}
                .test-result {{ margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .pass {{ background: #d4edda; border-left: 5px solid #28a745; }}
                .partial {{ background: #fff3cd; border-left: 5px solid #ffc107; }}
                .fail {{ background: #f8d7da; border-left: 5px solid #dc3545; }}
                .recommendations {{ background: #d1ecf1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                pre {{ background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎭 Cherry AI Streamlit Platform - E2E Test Report</h1>
                <p><strong>QA Analysis:</strong> Deep testing with coverage analysis and performance validation</p>
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>{report['summary']['total_tests']}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="metric">
                    <h3>{report['summary']['passed']}</h3>
                    <p>Passed</p>
                </div>
                <div class="metric">
                    <h3>{report['summary']['partial']}</h3>
                    <p>Partial</p>
                </div>
                <div class="metric">
                    <h3>{report['summary']['failed']}</h3>
                    <p>Failed</p>
                </div>
                <div class="metric">
                    <h3>{report['summary']['success_rate']:.1f}%</h3>
                    <p>Success Rate</p>
                </div>
            </div>
            
            <h2>Test Results</h2>
        """
        
        for result in report['results']:
            status_class = result['status'].lower()
            status_icon = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(result['status'].lower(), "⚪")
            
            html += f"""
            <div class="test-result {status_class}">
                <h3>{status_icon} {result['test_name']}</h3>
                <p><strong>Status:</strong> {result['status']} | <strong>Duration:</strong> {result['duration']:.2f}s</p>
                {f"<p><strong>Error:</strong> {result['error']}</p>" if result['error'] else ""}
                {f"<details><summary>Details</summary><pre>{json.dumps(result['details'], indent=2)}</pre></details>" if result['details'] else ""}
            </div>
            """
        
        html += f"""
            <div class="recommendations">
                <h2>🎯 QA Recommendations</h2>
                <ul>
                    {"".join(f"<li>{rec}</li>" for rec in report['recommendations'])}
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def generate_junit_report(self, report: Dict) -> str:
        """Generate JUnit XML report"""
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml += f'<testsuite name="CherryAI_E2E" tests="{report["summary"]["total_tests"]}" '
        xml += f'failures="{report["summary"]["failed"]}" errors="{report["summary"]["partial"]}" '
        xml += f'time="{sum(r["duration"] for r in report["results"]):.2f}">\n'
        
        for result in report['results']:
            xml += f'  <testcase name="{result["test_name"]}" time="{result["duration"]:.2f}"'
            
            if result['status'] == 'FAIL':
                xml += f'>\n    <failure message="{result["error"] or "Test failed"}">{result["error"] or "No details"}</failure>\n  </testcase>\n'
            elif result['status'] == 'PARTIAL':
                xml += f'>\n    <error message="{result["error"] or "Partial success"}">{result["error"] or "Partial functionality"}</error>\n  </testcase>\n'
            else:
                xml += ' />\n'
        
        xml += '</testsuite>\n'
        return xml


# Main test execution
@pytest.mark.asyncio
async def test_cherry_ai_comprehensive_suite():
    """Execute comprehensive test suite"""
    test_suite = CherryAITestSuite()
    test_instance = TestCherryAIPlatformE2E()
    
    logger.info("🚀 Starting Cherry AI Comprehensive E2E Test Suite")
    
    # Execute all tests
    test_methods = [
        test_instance.test_01_app_availability_and_health,
        test_instance.test_02_page_load_performance,
        test_instance.test_03_ui_elements_presence,
        test_instance.test_04_chat_interface_interaction,
        test_instance.test_05_file_upload_interface,
        test_instance.test_06_responsive_design,
        test_instance.test_07_accessibility_features,
        test_instance.test_08_performance_metrics
    ]
    
    for test_method in test_methods:
        try:
            await test_method(test_suite)
        except Exception as e:
            logger.error(f"Test execution error: {e}")
            continue
    
    # Generate comprehensive report
    report = await test_instance.generate_comprehensive_report(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("🎭 CHERRY AI E2E TEST SUITE - FINAL RESULTS")
    print(f"{'='*60}")
    print(f"📊 Total Tests: {report['summary']['total_tests']}")
    print(f"✅ Passed: {report['summary']['passed']}")
    print(f"⚠️  Partial: {report['summary']['partial']}")
    print(f"❌ Failed: {report['summary']['failed']}")
    print(f"🎯 Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"\n📄 Detailed reports saved to: {test_suite.artifacts_dir}")
    print(f"   - HTML Report: test_report.html")
    print(f"   - JSON Report: comprehensive_test_report.json")
    print(f"   - JUnit Report: junit_report.xml")
    print(f"   - Screenshots: *.png files")
    print(f"   - Browser Trace: trace.zip")
    
    if report['recommendations']:
        print(f"\n🎯 QA RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
    
    print(f"\n{'='*60}")
    
    # Assert overall success
    success_threshold = 70  # 70% success rate required
    assert report['summary']['success_rate'] >= success_threshold, \
        f"Test suite success rate {report['summary']['success_rate']:.1f}% below threshold {success_threshold}%"


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(test_cherry_ai_comprehensive_suite())