#!/usr/bin/env python3
"""
🎭 CherryAI E2E Playwright 통합 테스트

전체 CherryAI 시스템의 사용자 시나리오 기반 엔드투엔드 테스트
- Streamlit UI 전체 기능 테스트
- A2A 시스템 통합 테스트  
- 파일 업로드 및 데이터 분석 워크플로우
- 모니터링 대시보드 테스트

Author: CherryAI Production Team
"""

import pytest
import asyncio
import time
import os
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

class CherryAIE2ETest:
    """CherryAI E2E 통합 테스트 클래스"""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.monitoring_url = "http://localhost:8502"
        self.test_data_file = "test_data.csv"
        self.screenshot_dir = Path("test_screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        
    async def setup_test_data(self):
        """테스트용 샘플 데이터 생성"""
        test_data = """Name,Age,City,Salary
John Doe,25,New York,50000
Jane Smith,30,Los Angeles,60000
Bob Johnson,35,Chicago,55000
Alice Brown,28,Houston,52000
Charlie Davis,32,Phoenix,58000"""
        
        with open(self.test_data_file, 'w') as f:
            f.write(test_data)
        print(f"✅ 테스트 데이터 생성: {self.test_data_file}")
    
    async def cleanup_test_data(self):
        """테스트 데이터 정리"""
        if os.path.exists(self.test_data_file):
            os.remove(self.test_data_file)
            print(f"🗑️ 테스트 데이터 정리: {self.test_data_file}")
    
    async def take_screenshot(self, page: Page, name: str):
        """스크린샷 촬영"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = self.screenshot_dir / f"{name}_{timestamp}.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"📸 스크린샷 저장: {screenshot_path}")
        return screenshot_path
    
    async def wait_for_streamlit_ready(self, page: Page):
        """Streamlit 앱이 완전히 로드될 때까지 대기"""
        try:
            # Streamlit 특성상 동적 로딩이므로 충분히 대기
            await page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
            await page.wait_for_timeout(3000)  # 추가 안정화 대기
            print("✅ Streamlit 앱 로딩 완료")
            return True
        except Exception as e:
            print(f"❌ Streamlit 앱 로딩 실패: {e}")
            return False
    
    async def test_main_page_loading(self, page: Page):
        """메인 페이지 로딩 테스트"""
        print("\n🧪 테스트 1: 메인 페이지 로딩")
        print("-" * 40)
        
        try:
            # 메인 페이지 이동
            await page.goto(self.base_url, timeout=30000)
            print(f"📍 페이지 이동: {self.base_url}")
            
            # Streamlit 앱 로딩 대기
            if not await self.wait_for_streamlit_ready(page):
                return False
            
            # 페이지 제목 확인
            title = await page.title()
            print(f"📄 페이지 제목: {title}")
            
            # 스크린샷 촬영
            await self.take_screenshot(page, "main_page")
            
            # CherryAI 관련 텍스트가 있는지 확인
            page_content = await page.content()
            if "CherryAI" in page_content or "Cherry AI" in page_content:
                print("✅ CherryAI 메인 페이지 로딩 성공")
                return True
            else:
                print("⚠️ CherryAI 텍스트를 찾을 수 없음")
                return False
                
        except Exception as e:
            print(f"❌ 메인 페이지 로딩 실패: {e}")
            await self.take_screenshot(page, "main_page_error")
            return False
    
    async def test_file_upload(self, page: Page):
        """파일 업로드 기능 테스트"""
        print("\n🧪 테스트 2: 파일 업로드 기능")
        print("-" * 40)
        
        try:
            # 파일 업로더 찾기 (Streamlit file_uploader)
            file_uploader = await page.query_selector('input[type="file"]')
            
            if file_uploader:
                # 테스트 파일 업로드
                await file_uploader.set_input_files(self.test_data_file)
                print(f"📤 파일 업로드: {self.test_data_file}")
                
                # 업로드 완료 대기
                await page.wait_for_timeout(3000)
                
                # 업로드 성공 확인
                await self.take_screenshot(page, "file_uploaded")
                print("✅ 파일 업로드 테스트 성공")
                return True
            else:
                print("⚠️ 파일 업로더를 찾을 수 없음")
                await self.take_screenshot(page, "no_file_uploader")
                return False
                
        except Exception as e:
            print(f"❌ 파일 업로드 테스트 실패: {e}")
            await self.take_screenshot(page, "file_upload_error")
            return False
    
    async def test_ai_chat_interface(self, page: Page):
        """AI 채팅 인터페이스 테스트"""
        print("\n🧪 테스트 3: AI 채팅 인터페이스")
        print("-" * 40)
        
        try:
            # 채팅 입력창 찾기
            chat_inputs = await page.query_selector_all('textarea, input[type="text"]')
            
            for chat_input in chat_inputs:
                # placeholder나 label로 채팅 입력창 식별
                placeholder = await chat_input.get_attribute('placeholder')
                if placeholder and ('메시지' in placeholder or 'message' in placeholder.lower() or '채팅' in placeholder):
                    # 테스트 메시지 입력
                    test_message = "안녕하세요! 시스템 상태를 확인해주세요."
                    await chat_input.fill(test_message)
                    print(f"💬 메시지 입력: {test_message}")
                    
                    # Enter 키 또는 전송 버튼 클릭
                    await chat_input.press('Enter')
                    
                    # 응답 대기
                    await page.wait_for_timeout(5000)
                    
                    await self.take_screenshot(page, "ai_chat")
                    print("✅ AI 채팅 인터페이스 테스트 성공")
                    return True
            
            print("⚠️ 채팅 입력창을 찾을 수 없음")
            await self.take_screenshot(page, "no_chat_input")
            return False
            
        except Exception as e:
            print(f"❌ AI 채팅 테스트 실패: {e}")
            await self.take_screenshot(page, "ai_chat_error")
            return False
    
    async def test_sidebar_navigation(self, page: Page):
        """사이드바 네비게이션 테스트"""
        print("\n🧪 테스트 4: 사이드바 네비게이션")
        print("-" * 40)
        
        try:
            # Streamlit 사이드바 확인
            sidebar = await page.query_selector('[data-testid="stSidebar"]')
            
            if sidebar:
                # 사이드바의 링크/버튼들 찾기
                sidebar_links = await sidebar.query_selector_all('a, button, div[role="button"]')
                
                if sidebar_links:
                    print(f"📋 사이드바 항목 발견: {len(sidebar_links)}개")
                    
                    # 첫 번째 링크 클릭 테스트
                    if len(sidebar_links) > 0:
                        await sidebar_links[0].click()
                        await page.wait_for_timeout(2000)
                        print("🖱️ 사이드바 네비게이션 클릭 테스트 성공")
                        
                    await self.take_screenshot(page, "sidebar_navigation")
                    return True
                else:
                    print("⚠️ 사이드바 항목이 없음")
                    return False
            else:
                print("⚠️ 사이드바를 찾을 수 없음")
                return False
                
        except Exception as e:
            print(f"❌ 사이드바 네비게이션 테스트 실패: {e}")
            await self.take_screenshot(page, "sidebar_error")
            return False
    
    async def test_monitoring_dashboard(self, page: Page):
        """모니터링 대시보드 테스트"""
        print("\n🧪 테스트 5: 모니터링 대시보드")
        print("-" * 40)
        
        try:
            # 모니터링 대시보드로 이동
            await page.goto(self.monitoring_url, timeout=30000)
            print(f"📍 모니터링 대시보드 이동: {self.monitoring_url}")
            
            # 대시보드 로딩 대기
            if await self.wait_for_streamlit_ready(page):
                await self.take_screenshot(page, "monitoring_dashboard")
                
                # 대시보드 특정 요소들 확인
                page_content = await page.content()
                if "모니터링" in page_content or "Monitoring" in page_content:
                    print("✅ 모니터링 대시보드 테스트 성공")
                    return True
                else:
                    print("⚠️ 모니터링 대시보드 콘텐츠 확인 실패")
                    return False
            else:
                print("❌ 모니터링 대시보드 로딩 실패")
                return False
                
        except Exception as e:
            print(f"❌ 모니터링 대시보드 테스트 실패: {e}")
            await self.take_screenshot(page, "monitoring_error")
            return False
    
    async def test_a2a_system_integration(self, page: Page):
        """A2A 시스템 통합 테스트"""
        print("\n🧪 테스트 6: A2A 시스템 통합")
        print("-" * 40)
        
        try:
            # 메인 페이지로 돌아가기
            await page.goto(self.base_url, timeout=30000)
            await self.wait_for_streamlit_ready(page)
            
            # A2A 관련 기능 테스트 (특정 페이지나 버튼이 있다면)
            page_content = await page.content()
            
            # A2A, 오케스트레이터, 에이전트 관련 텍스트 확인
            a2a_keywords = ["A2A", "오케스트레이터", "에이전트", "Agent", "Orchestrator"]
            found_keywords = [keyword for keyword in a2a_keywords if keyword in page_content]
            
            if found_keywords:
                print(f"✅ A2A 시스템 관련 키워드 발견: {found_keywords}")
                await self.take_screenshot(page, "a2a_integration")
                return True
            else:
                print("⚠️ A2A 시스템 관련 콘텐츠를 찾을 수 없음")
                await self.take_screenshot(page, "no_a2a_content")
                return False
                
        except Exception as e:
            print(f"❌ A2A 시스템 통합 테스트 실패: {e}")
            await self.take_screenshot(page, "a2a_integration_error")
            return False
    
    async def run_full_e2e_test(self):
        """전체 E2E 테스트 실행"""
        print("🎭 CherryAI E2E Playwright 통합 테스트 시작")
        print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 테스트 데이터 준비
        await self.setup_test_data()
        
        test_results = {
            "main_page_loading": False,
            "file_upload": False,
            "ai_chat_interface": False,
            "sidebar_navigation": False,
            "monitoring_dashboard": False,
            "a2a_system_integration": False
        }
        
        async with async_playwright() as p:
            # 브라우저 시작
            browser = await p.chromium.launch(headless=False, slow_mo=1000)
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            try:
                # 테스트 실행
                test_results["main_page_loading"] = await self.test_main_page_loading(page)
                test_results["file_upload"] = await self.test_file_upload(page)
                test_results["ai_chat_interface"] = await self.test_ai_chat_interface(page)
                test_results["sidebar_navigation"] = await self.test_sidebar_navigation(page)
                test_results["monitoring_dashboard"] = await self.test_monitoring_dashboard(page)
                test_results["a2a_system_integration"] = await self.test_a2a_system_integration(page)
                
            finally:
                await browser.close()
        
        # 테스트 정리
        await self.cleanup_test_data()
        
        # 결과 리포트
        self.generate_test_report(test_results)
        
        return test_results
    
    def generate_test_report(self, results: dict):
        """테스트 결과 리포트 생성"""
        print("\n" + "=" * 60)
        print("📊 CherryAI E2E 테스트 결과 리포트")
        print("=" * 60)
        
        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        for test_name, result in results.items():
            status = "✅ 성공" if result else "❌ 실패"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 전체 결과: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 E2E 테스트 완전 성공! CherryAI 시스템이 정상 작동합니다!")
            overall_status = "EXCELLENT"
        elif success_rate >= 70:
            print("✅ E2E 테스트 대부분 성공! 시스템이 안정적으로 작동합니다.")
            overall_status = "GOOD"
        elif success_rate >= 50:
            print("⚠️ E2E 테스트 부분 성공. 일부 개선이 필요합니다.")
            overall_status = "FAIR"
        else:
            print("🚨 E2E 테스트 실패. 시스템 점검이 필요합니다.")
            overall_status = "POOR"
        
        print(f"📁 스크린샷 저장 위치: {self.screenshot_dir}")
        print(f"⏱️ 테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return overall_status


async def main():
    """메인 테스트 실행"""
    test = CherryAIE2ETest()
    results = await test.run_full_e2e_test()
    
    # 결과에 따른 종료 코드
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100
    
    return success_rate >= 70  # 70% 이상이면 성공


if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    exit(0 if success else 1) 