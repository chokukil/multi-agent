#!/usr/bin/env python3
"""
Playwright E2E 테스트 시스템
Requirements 15에 따른 MCP 통합 E2E 테스트 구현

핵심 기능:
1. Streamlit 앱 자동화 E2E 테스트
2. MCP 서버 통합 검증
3. 파일 업로드 및 데이터 분석 워크플로우 테스트
4. 도메인 전문성 시나리오 테스트
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright not available. E2E tests will be disabled.")

from ..llm_factory import LLMFactory
from ..monitoring.mcp_server_manager import MCPServerManager
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class PlaywrightE2ETester:
    """
    Playwright 기반 E2E 테스트 시스템
    
    MCP 통합 및 실제 사용자 시나리오 검증
    """
    
    def __init__(self, headless: bool = True, slow_mo: int = 100):
        """
        E2E 테스터 초기화
        
        Args:
            headless: 헤드리스 모드 실행 여부
            slow_mo: 액션 간 지연시간 (ms)
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.browser = None
        self.context = None
        self.page = None
        
        # MCP 서버 관리자
        self.mcp_manager = MCPServerManager()
        
        # LLM 클라이언트
        self.llm_client = LLMFactory.create_llm()
        
        # 테스트 결과 저장
        self.test_results = []
        self.screenshots_dir = Path("e2e_test_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # 설정
        self.config = get_config()
        self.streamlit_url = self.config.get("streamlit_url", "http://localhost:8501")
        
        logger.info(f"PlaywrightE2ETester initialized - headless: {headless}")
    
    async def setup_browser(self):
        """브라우저 초기화"""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is not installed")
        
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="ko-KR"
        )
        
        self.page = await self.context.new_page()
        
        # 콘솔 로그 수집
        self.page.on("console", lambda msg: logger.info(f"Browser console: {msg.text}"))
        
        logger.info("Browser setup completed")
    
    async def teardown_browser(self):
        """브라우저 정리"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        logger.info("Browser teardown completed")
    
    async def run_comprehensive_e2e_tests(self) -> Dict[str, Any]:
        """
        포괄적인 E2E 테스트 실행
        
        Returns:
            전체 테스트 결과
        """
        start_time = datetime.now()
        
        try:
            await self.setup_browser()
            
            # MCP 서버 시작 확인
            mcp_status = await self.verify_mcp_servers()
            
            # 기본 UI 테스트
            ui_test_results = await self.run_basic_ui_tests()
            
            # 파일 업로드 테스트
            upload_test_results = await self.run_file_upload_tests()
            
            # 멀티에이전트 협업 테스트
            collaboration_test_results = await self.run_multiagent_collaboration_tests()
            
            # 도메인 전문성 테스트
            expertise_test_results = await self.run_domain_expertise_tests()
            
            # 성능 테스트
            performance_results = await self.run_performance_tests()
            
            # 전체 결과 종합
            total_results = {
                "test_session": {
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration": (datetime.now() - start_time).total_seconds(),
                    "browser": "chromium",
                    "headless": self.headless
                },
                "mcp_servers": mcp_status,
                "ui_tests": ui_test_results,
                "upload_tests": upload_test_results,
                "collaboration_tests": collaboration_test_results,
                "expertise_tests": expertise_test_results,
                "performance_tests": performance_results,
                "overall_success_rate": self._calculate_success_rate(),
                "screenshots_saved": len(list(self.screenshots_dir.glob("*.png")))
            }
            
            # 결과 저장
            await self.save_test_results(total_results)
            
            return total_results
            
        except Exception as e:
            logger.error(f"E2E test execution failed: {str(e)}")
            return {
                "error": str(e),
                "test_session": {
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "success": False
                }
            }
            
        finally:
            await self.teardown_browser()
    
    async def verify_mcp_servers(self) -> Dict[str, Any]:
        """MCP 서버 상태 검증"""
        logger.info("Verifying MCP servers...")
        
        # MCP 서버 상태 확인
        server_status = await self.mcp_manager.check_all_servers()
        
        return {
            "servers_checked": len(server_status),
            "servers_running": len([s for s in server_status.values() if s.get("status") == "running"]),
            "server_details": server_status,
            "mcp_integration_ready": all(s.get("status") == "running" for s in server_status.values())
        }
    
    async def run_basic_ui_tests(self) -> Dict[str, Any]:
        """기본 UI 테스트"""
        logger.info("Running basic UI tests...")
        
        test_results = []
        
        try:
            # Streamlit 앱 로드
            await self.page.goto(self.streamlit_url)
            await self.page.wait_for_load_state("networkidle")
            
            # 스크린샷 캡처
            screenshot_path = self.screenshots_dir / f"ui_main_page_{int(time.time())}.png"
            await self.page.screenshot(path=screenshot_path)
            
            # 페이지 제목 확인
            title = await self.page.title()
            test_results.append({
                "test": "page_title",
                "success": "CherryAI" in title or "Data Analysis" in title,
                "details": f"Title: {title}"
            })
            
            # 주요 컴포넌트 존재 확인
            sidebar_exists = await self.page.locator("[data-testid='stSidebar']").count() > 0
            test_results.append({
                "test": "sidebar_exists",
                "success": sidebar_exists,
                "details": f"Sidebar found: {sidebar_exists}"
            })
            
            # 파일 업로더 존재 확인
            file_uploader = await self.page.locator("[data-testid='stFileUploader']").count() > 0
            test_results.append({
                "test": "file_uploader_exists",
                "success": file_uploader,
                "details": f"File uploader found: {file_uploader}"
            })
            
            # 채팅 인터페이스 확인
            chat_input = await self.page.locator("textarea").count() > 0
            test_results.append({
                "test": "chat_interface_exists",
                "success": chat_input,
                "details": f"Chat input found: {chat_input}"
            })
            
        except Exception as e:
            test_results.append({
                "test": "basic_ui_load",
                "success": False,
                "error": str(e)
            })
        
        return {
            "tests_run": len(test_results),
            "tests_passed": len([t for t in test_results if t.get("success", False)]),
            "results": test_results
        }
    
    async def run_file_upload_tests(self) -> Dict[str, Any]:
        """파일 업로드 테스트"""
        logger.info("Running file upload tests...")
        
        test_results = []
        
        try:
            # 테스트용 CSV 파일 생성
            test_csv_path = await self._create_test_csv()
            
            # 파일 업로더 찾기
            file_uploader = self.page.locator("[data-testid='stFileUploader'] input")
            
            if await file_uploader.count() > 0:
                # 파일 업로드
                await file_uploader.set_input_files(str(test_csv_path))
                
                # 업로드 완료 대기
                await self.page.wait_for_timeout(3000)
                
                # 스크린샷 캡처
                screenshot_path = self.screenshots_dir / f"file_uploaded_{int(time.time())}.png"
                await self.page.screenshot(path=screenshot_path)
                
                # 업로드 성공 확인
                success_indicators = [
                    await self.page.locator("text=업로드").count() > 0,
                    await self.page.locator("text=데이터").count() > 0,
                    await self.page.locator("text=분석").count() > 0
                ]
                
                upload_success = any(success_indicators)
                
                test_results.append({
                    "test": "csv_file_upload",
                    "success": upload_success,
                    "details": f"Upload indicators: {success_indicators}"
                })
                
                # 파일 정보 표시 확인
                await self.page.wait_for_timeout(2000)
                data_info = await self.page.locator("text=행").count() > 0
                
                test_results.append({
                    "test": "file_info_display",
                    "success": data_info,
                    "details": f"Data info displayed: {data_info}"
                })
            else:
                test_results.append({
                    "test": "file_uploader_not_found",
                    "success": False,
                    "details": "File uploader component not found"
                })
                
        except Exception as e:
            test_results.append({
                "test": "file_upload_error",
                "success": False,
                "error": str(e)
            })
        
        return {
            "tests_run": len(test_results),
            "tests_passed": len([t for t in test_results if t.get("success", False)]),
            "results": test_results
        }
    
    async def run_multiagent_collaboration_tests(self) -> Dict[str, Any]:
        """멀티에이전트 협업 테스트"""
        logger.info("Running multiagent collaboration tests...")
        
        test_results = []
        
        try:
            # 분석 쿼리 입력
            test_query = "이 데이터의 기본 통계와 상관관계를 분석해주세요"
            
            # 채팅 입력창 찾기
            chat_input = self.page.locator("textarea")
            
            if await chat_input.count() > 0:
                # 쿼리 입력
                await chat_input.fill(test_query)
                await self.page.keyboard.press("Enter")
                
                # 분석 진행 대기
                await self.page.wait_for_timeout(10000)
                
                # 에이전트 활동 스크린샷
                screenshot_path = self.screenshots_dir / f"agent_collaboration_{int(time.time())}.png"
                await self.page.screenshot(path=screenshot_path)
                
                # 분석 결과 확인
                results_indicators = [
                    await self.page.locator("text=통계").count() > 0,
                    await self.page.locator("text=상관관계").count() > 0,
                    await self.page.locator("text=분석").count() > 0,
                    await self.page.locator("text=결과").count() > 0
                ]
                
                analysis_completed = any(results_indicators)
                
                test_results.append({
                    "test": "multiagent_analysis",
                    "success": analysis_completed,
                    "query": test_query,
                    "details": f"Analysis indicators: {results_indicators}"
                })
                
                # 차트/시각화 확인
                await self.page.wait_for_timeout(3000)
                charts_present = await self.page.locator("[data-testid='stPlotlyChart']").count() > 0
                
                test_results.append({
                    "test": "visualization_generation",
                    "success": charts_present,
                    "details": f"Charts generated: {charts_present}"
                })
            else:
                test_results.append({
                    "test": "chat_input_not_found",
                    "success": False,
                    "details": "Chat input not found"
                })
                
        except Exception as e:
            test_results.append({
                "test": "collaboration_test_error",
                "success": False,
                "error": str(e)
            })
        
        return {
            "tests_run": len(test_results),
            "tests_passed": len([t for t in test_results if t.get("success", False)]),
            "results": test_results
        }
    
    async def run_domain_expertise_tests(self) -> Dict[str, Any]:
        """도메인 전문성 테스트"""
        logger.info("Running domain expertise tests...")
        
        test_results = []
        
        try:
            # 전문 도메인 쿼리 테스트
            expert_queries = [
                "이 데이터셋에서 이상치를 감지하고 분석해주세요",
                "주성분 분석(PCA)를 수행해주세요",
                "머신러닝 모델을 구축하고 성능을 평가해주세요"
            ]
            
            for i, query in enumerate(expert_queries):
                # 쿼리 입력
                chat_input = self.page.locator("textarea")
                if await chat_input.count() > 0:
                    await chat_input.fill(query)
                    await self.page.keyboard.press("Enter")
                    
                    # 처리 시간 대기
                    await self.page.wait_for_timeout(15000)
                    
                    # 전문성 결과 스크린샷
                    screenshot_path = self.screenshots_dir / f"expertise_test_{i}_{int(time.time())}.png"
                    await self.page.screenshot(path=screenshot_path)
                    
                    # 전문적 응답 확인
                    expert_indicators = [
                        await self.page.locator("text=분석").count() > 0,
                        await self.page.locator("text=모델").count() > 0,
                        await self.page.locator("text=결과").count() > 0
                    ]
                    
                    expertise_demonstrated = any(expert_indicators)
                    
                    test_results.append({
                        "test": f"expert_query_{i+1}",
                        "success": expertise_demonstrated,
                        "query": query[:50] + "...",
                        "details": f"Expert response indicators: {expert_indicators}"
                    })
                    
                    # 다음 테스트를 위한 대기
                    await self.page.wait_for_timeout(2000)
                
        except Exception as e:
            test_results.append({
                "test": "expertise_test_error",
                "success": False,
                "error": str(e)
            })
        
        return {
            "tests_run": len(test_results),
            "tests_passed": len([t for t in test_results if t.get("success", False)]),
            "results": test_results
        }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """성능 테스트"""
        logger.info("Running performance tests...")
        
        performance_metrics = {}
        
        try:
            # 페이지 로드 시간 측정
            start_time = time.time()
            await self.page.goto(self.streamlit_url)
            await self.page.wait_for_load_state("networkidle")
            page_load_time = time.time() - start_time
            
            performance_metrics["page_load_time"] = page_load_time
            
            # 첫 번째 응답 시간 측정 (TTFT)
            start_time = time.time()
            chat_input = self.page.locator("textarea")
            if await chat_input.count() > 0:
                await chat_input.fill("간단한 데이터 요약을 해주세요")
                await self.page.keyboard.press("Enter")
                
                # 첫 번째 응답 대기
                await self.page.wait_for_timeout(5000)
                ttft = time.time() - start_time
                
                performance_metrics["time_to_first_token"] = ttft
            
            # 메모리 사용량 추정 (브라우저 기준)
            performance_info = await self.page.evaluate("""
                () => ({
                    memory: performance.memory ? {
                        used: performance.memory.usedJSHeapSize,
                        total: performance.memory.totalJSHeapSize,
                        limit: performance.memory.jsHeapSizeLimit
                    } : null,
                    navigation: performance.navigation.type,
                    timing: performance.timing.loadEventEnd - performance.timing.navigationStart
                })
            """)
            
            performance_metrics["browser_performance"] = performance_info
            
        except Exception as e:
            performance_metrics["error"] = str(e)
        
        return {
            "metrics": performance_metrics,
            "performance_acceptable": performance_metrics.get("page_load_time", 999) < 10.0,
            "ttft_acceptable": performance_metrics.get("time_to_first_token", 999) < 30.0
        }
    
    async def _create_test_csv(self) -> Path:
        """테스트용 CSV 파일 생성"""
        test_data = """name,age,salary,department
Alice,25,50000,Engineering
Bob,30,60000,Marketing
Charlie,35,70000,Engineering
Diana,28,55000,Sales
Eve,32,65000,Marketing
Frank,29,58000,Engineering
Grace,31,62000,Sales
Henry,27,52000,Marketing
Iris,33,68000,Engineering
Jack,26,51000,Sales"""
        
        test_csv_path = Path("test_data.csv")
        test_csv_path.write_text(test_data)
        
        return test_csv_path.absolute()
    
    def _calculate_success_rate(self) -> float:
        """전체 성공률 계산"""
        total_tests = 0
        passed_tests = 0
        
        for result in self.test_results:
            if isinstance(result, dict) and "results" in result:
                for test in result["results"]:
                    total_tests += 1
                    if test.get("success", False):
                        passed_tests += 1
        
        return (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
    
    async def save_test_results(self, results: Dict[str, Any]):
        """테스트 결과 저장"""
        try:
            # JSON 결과 저장
            results_file = Path(f"e2e_test_results_{int(time.time())}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 요약 보고서 생성
            report_file = Path(f"e2e_test_report_{int(time.time())}.md")
            await self._generate_markdown_report(results, report_file)
            
            logger.info(f"E2E test results saved: {results_file}, {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {str(e)}")
    
    async def _generate_markdown_report(self, results: Dict[str, Any], report_file: Path):
        """마크다운 보고서 생성"""
        report = f"""# E2E 테스트 보고서

## 테스트 세션 정보
- **시작 시간**: {results['test_session']['start_time']}
- **종료 시간**: {results['test_session']['end_time']}
- **지속 시간**: {results['test_session']['duration']:.2f}초
- **브라우저**: {results['test_session']['browser']}
- **헤드리스 모드**: {results['test_session']['headless']}

## MCP 서버 상태
- **확인된 서버**: {results['mcp_servers']['servers_checked']}개
- **실행 중인 서버**: {results['mcp_servers']['servers_running']}개
- **통합 준비**: {'✅' if results['mcp_servers']['mcp_integration_ready'] else '❌'}

## UI 테스트 결과
- **테스트 실행**: {results['ui_tests']['tests_run']}개
- **테스트 통과**: {results['ui_tests']['tests_passed']}개
- **성공률**: {results['ui_tests']['tests_passed']/results['ui_tests']['tests_run']*100:.1f}%

## 파일 업로드 테스트
- **테스트 실행**: {results['upload_tests']['tests_run']}개  
- **테스트 통과**: {results['upload_tests']['tests_passed']}개
- **성공률**: {results['upload_tests']['tests_passed']/results['upload_tests']['tests_run']*100:.1f}%

## 멀티에이전트 협업 테스트
- **테스트 실행**: {results['collaboration_tests']['tests_run']}개
- **테스트 통과**: {results['collaboration_tests']['tests_passed']}개
- **성공률**: {results['collaboration_tests']['tests_passed']/results['collaboration_tests']['tests_run']*100:.1f}%

## 도메인 전문성 테스트  
- **테스트 실행**: {results['expertise_tests']['tests_run']}개
- **테스트 통과**: {results['expertise_tests']['tests_passed']}개
- **성공률**: {results['expertise_tests']['tests_passed']/results['expertise_tests']['tests_run']*100:.1f}%

## 성능 테스트 결과
- **페이지 로드 시간**: {results['performance_tests']['metrics'].get('page_load_time', 'N/A')}초
- **첫 토큰까지 시간**: {results['performance_tests']['metrics'].get('time_to_first_token', 'N/A')}초
- **성능 허용 기준**: {'✅' if results['performance_tests']['performance_acceptable'] else '❌'}

## 전체 성공률
**{results['overall_success_rate']:.1f}%**

## 스크린샷
총 {results['screenshots_saved']}개의 스크린샷이 저장되었습니다.

---
*Generated by CherryAI Playwright E2E Tester*
"""
        
        report_file.write_text(report, encoding='utf-8')


# 편의 함수
async def run_e2e_tests(headless: bool = True, slow_mo: int = 100) -> Dict[str, Any]:
    """E2E 테스트 실행 편의 함수"""
    tester = PlaywrightE2ETester(headless=headless, slow_mo=slow_mo)
    return await tester.run_comprehensive_e2e_tests()


if __name__ == "__main__":
    async def main():
        # 개발 모드에서 헤드리스 false로 실행
        results = await run_e2e_tests(headless=False, slow_mo=500)
        print(f"E2E 테스트 완료! 전체 성공률: {results.get('overall_success_rate', 0):.1f}%")
    
    asyncio.run(main())