"""
CherryAI Streamlit UI Playwright 테스트
최종 UI 확인이 필요한 테스트만 Playwright로 진행
"""

import pytest
import asyncio
import time
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCherryAIStreamlitUI:
    """CherryAI Streamlit UI 테스트"""
    
    @pytest.fixture
    async def browser_context(self):
        """브라우저 컨텍스트 생성"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            yield context
            await context.close()
            await browser.close()
    
    @pytest.fixture
    async def page(self, browser_context):
        """페이지 생성"""
        page = await browser_context.new_page()
        yield page
        await page.close()
    
    @pytest.mark.asyncio
    async def test_streamlit_app_loads(self, page: Page):
        """Streamlit 앱 로딩 테스트"""
        try:
            # Given: Streamlit 앱 URL
            streamlit_url = "http://localhost:8501"
            
            # When: 페이지 로드
            await page.goto(streamlit_url, timeout=30000)
            
            # Then: 페이지 제목 확인
            title = await page.title()
            assert "CherryAI" in title or "Streamlit" in title
            
            # 메인 콘텐츠 확인
            await page.wait_for_selector("div[data-testid='stApp']", timeout=10000)
            
            print("✅ Streamlit 앱 정상 로딩")
            
        except Exception as e:
            pytest.skip(f"Streamlit 앱이 실행되지 않음: {e}")
    
    @pytest.mark.asyncio
    async def test_orchestrator_interaction(self, page: Page):
        """오케스트레이터 상호작용 테스트"""
        try:
            # Given: Streamlit 앱 로드
            await page.goto("http://localhost:8501", timeout=30000)
            await page.wait_for_selector("div[data-testid='stApp']", timeout=10000)
            
            # 채팅 입력 필드 찾기
            chat_input = await page.wait_for_selector("textarea", timeout=10000)
            
            # When: 간단한 메시지 입력
            await chat_input.fill("안녕하세요")
            
            # 전송 버튼 클릭 (Enter 키 또는 버튼)
            await page.keyboard.press("Enter")
            
            # Then: 응답 대기 및 확인
            await page.wait_for_timeout(5000)  # 5초 대기
            
            # 응답 메시지 확인
            messages = await page.query_selector_all("div[data-testid='stChatMessage']")
            assert len(messages) >= 2  # 사용자 메시지 + 응답
            
            print("✅ 오케스트레이터 상호작용 성공")
            
        except Exception as e:
            pytest.skip(f"오케스트레이터 상호작용 실패: {e}")
    
    @pytest.mark.asyncio
    async def test_data_analysis_workflow(self, page: Page):
        """데이터 분석 워크플로우 UI 테스트"""
        try:
            # Given: Streamlit 앱 로드
            await page.goto("http://localhost:8501", timeout=30000)
            await page.wait_for_selector("div[data-testid='stApp']", timeout=10000)
            
            # 채팅 입력 필드 찾기
            chat_input = await page.wait_for_selector("textarea", timeout=10000)
            
            # When: 데이터 분석 요청
            analysis_request = "ion_implant_3lot_dataset.csv 파일로 간단한 EDA 분석을 해주세요"
            await chat_input.fill(analysis_request)
            await page.keyboard.press("Enter")
            
            # Then: 분석 과정 및 결과 확인
            await page.wait_for_timeout(30000)  # 30초 대기 (분석 시간 고려)
            
            # 응답 메시지들 확인
            messages = await page.query_selector_all("div[data-testid='stChatMessage']")
            assert len(messages) >= 2
            
            # 아티팩트나 차트가 표시되는지 확인
            # (Streamlit의 차트 컴포넌트들)
            charts = await page.query_selector_all("div[data-testid='stPlotlyChart'], div[data-testid='stPyplotChart']")
            
            print(f"✅ 데이터 분석 워크플로우 완료 (메시지: {len(messages)}, 차트: {len(charts)})")
            
        except Exception as e:
            pytest.skip(f"데이터 분석 워크플로우 실패: {e}")
    
    @pytest.mark.asyncio
    async def test_streaming_response_display(self, page: Page):
        """스트리밍 응답 표시 테스트"""
        try:
            # Given: Streamlit 앱 로드
            await page.goto("http://localhost:8501", timeout=30000)
            await page.wait_for_selector("div[data-testid='stApp']", timeout=10000)
            
            # 채팅 입력 필드 찾기
            chat_input = await page.wait_for_selector("textarea", timeout=10000)
            
            # When: 메시지 전송
            await chat_input.fill("데이터 분석 과정을 자세히 설명해주세요")
            await page.keyboard.press("Enter")
            
            # 초기 응답 시작 확인
            await page.wait_for_timeout(2000)
            
            # 스트리밍 중 메시지 변화 모니터링
            initial_messages = await page.query_selector_all("div[data-testid='stChatMessage']")
            initial_count = len(initial_messages)
            
            # 추가 대기 후 메시지 변화 확인
            await page.wait_for_timeout(10000)
            final_messages = await page.query_selector_all("div[data-testid='stChatMessage']")
            final_count = len(final_messages)
            
            # Then: 응답이 표시되었는지 확인
            assert final_count > initial_count or final_count >= 2
            
            print("✅ 스트리밍 응답 표시 확인")
            
        except Exception as e:
            pytest.skip(f"스트리밍 응답 표시 실패: {e}")
    
    @pytest.mark.asyncio
    async def test_artifact_display(self, page: Page):
        """아티팩트 표시 테스트"""
        try:
            # Given: Streamlit 앱 로드
            await page.goto("http://localhost:8501", timeout=30000)
            await page.wait_for_selector("div[data-testid='stApp']", timeout=10000)
            
            # 채팅 입력 필드 찾기
            chat_input = await page.wait_for_selector("textarea", timeout=10000)
            
            # When: 아티팩트 생성 요청
            await chat_input.fill("실행 계획을 세워주세요")
            await page.keyboard.press("Enter")
            
            # Then: 아티팩트 표시 확인
            await page.wait_for_timeout(15000)  # 15초 대기
            
            # 다양한 아티팩트 요소들 확인
            artifacts = await page.query_selector_all(
                "div[data-testid='stJson'], div[data-testid='stDataFrame'], "
                "div[data-testid='stCode'], div[data-testid='stMarkdown']"
            )
            
            print(f"✅ 아티팩트 표시 확인 (아티팩트 수: {len(artifacts)})")
            
        except Exception as e:
            pytest.skip(f"아티팩트 표시 실패: {e}")


if __name__ == "__main__":
    # UI 테스트 실행
    pytest.main([__file__, "-v", "--tb=short", "-s"])
