#!/usr/bin/env python3
"""
CherryAI Streamlit UI Playwright 자동화 테스트
파일 업로드부터 분석 결과까지 전체 워크플로우 검증

Author: CherryAI Team
"""

import asyncio
import os
import time
import json
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright

class CherryAIPlaywrightTest:
    """CherryAI UI 자동화 테스트"""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.test_file = "test_data_for_playwright.csv"
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": [],
            "overall_success": False,
            "errors": []
        }
    
    async def run_comprehensive_test(self):
        """종합 UI 테스트 실행"""
        print("🧪 CherryAI Playwright UI 자동화 테스트 시작")
        print("=" * 60)
        
        async with async_playwright() as p:
            # 브라우저 시작 (headless=False로 시각적 확인 가능)
            browser = await p.chromium.launch(headless=False, slow_mo=1000)
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 720}
            )
            page = await context.new_page()
            
            try:
                # 1. 기본 접속 테스트
                await self._test_basic_navigation(page)
                
                # 2. UI 요소 확인 테스트
                await self._test_ui_elements(page)
                
                # 3. 파일 업로드 테스트
                await self._test_file_upload(page)
                
                # 4. 분석 실행 테스트
                await self._test_analysis_execution(page)
                
                # 5. 결과 표시 테스트
                await self._test_results_display(page)
                
                # 전체 결과 계산
                success_count = sum(1 for test in self.results["tests"] if test["success"])
                total_count = len(self.results["tests"])
                self.results["overall_success"] = success_count == total_count
                
                print(f"\n📊 테스트 결과: {success_count}/{total_count} 성공")
                
            except Exception as e:
                self.results["errors"].append(f"전체 테스트 실패: {str(e)}")
                print(f"❌ 전체 테스트 실패: {e}")
                
            finally:
                await browser.close()
        
        return self.results
    
    async def _test_basic_navigation(self, page):
        """기본 접속 및 네비게이션 테스트"""
        print("\n1️⃣ 기본 접속 테스트")
        
        try:
            # 메인 페이지 로드
            await page.goto(self.base_url, timeout=30000)
            await page.wait_for_load_state('networkidle', timeout=30000)
            
            # 페이지 제목 확인
            title = await page.title()
            print(f"📄 페이지 제목: {title}")
            
            # Streamlit 앱이 로드되었는지 확인
            streamlit_app = await page.query_selector('[data-testid="stApp"]')
            
            success = streamlit_app is not None
            self._log_test("기본 접속", success, f"제목: {title}")
            
            if success:
                print("✅ 기본 접속 성공")
            else:
                print("❌ Streamlit 앱 로드 실패")
                
        except Exception as e:
            self._log_test("기본 접속", False, f"오류: {str(e)}")
            print(f"❌ 기본 접속 실패: {e}")
    
    async def _test_ui_elements(self, page):
        """UI 요소 확인 테스트"""
        print("\n2️⃣ UI 요소 확인 테스트")
        
        try:
            # 사이드바 확인
            sidebar = await page.query_selector('[data-testid="stSidebar"]')
            sidebar_ok = sidebar is not None
            
            # 메인 컨테이너 확인
            main_container = await page.query_selector('[data-testid="stAppViewContainer"]')
            main_ok = main_container is not None
            
            # 파일 업로더 찾기
            file_uploader = await page.query_selector('[data-testid="stFileUploader"]')
            uploader_ok = file_uploader is not None
            
            success = sidebar_ok and main_ok and uploader_ok
            details = f"사이드바: {sidebar_ok}, 메인: {main_ok}, 업로더: {uploader_ok}"
            
            self._log_test("UI 요소 확인", success, details)
            
            if success:
                print("✅ 주요 UI 요소 확인 완료")
            else:
                print(f"⚠️ 일부 UI 요소 누락: {details}")
                
        except Exception as e:
            self._log_test("UI 요소 확인", False, f"오류: {str(e)}")
            print(f"❌ UI 요소 확인 실패: {e}")
    
    async def _test_file_upload(self, page):
        """파일 업로드 테스트"""
        print("\n3️⃣ 파일 업로드 테스트")
        
        try:
            # 파일 업로더 찾기
            file_input = await page.query_selector('input[type="file"]')
            
            if file_input:
                # 테스트 파일 업로드
                await file_input.set_input_files(self.test_file)
                await page.wait_for_timeout(3000)  # 업로드 완료 대기
                
                # 업로드 성공 확인 (파일명 표시 등)
                uploaded_file = await page.query_selector_all('text="test_data_for_playwright.csv"')
                success = len(uploaded_file) > 0
                
                self._log_test("파일 업로드", success, f"파일: {self.test_file}")
                
                if success:
                    print("✅ 파일 업로드 성공")
                else:
                    print("⚠️ 파일 업로드 완료되었으나 UI에 표시 안됨")
                    
            else:
                self._log_test("파일 업로드", False, "파일 입력 요소를 찾을 수 없음")
                print("❌ 파일 업로더를 찾을 수 없음")
                
        except Exception as e:
            self._log_test("파일 업로드", False, f"오류: {str(e)}")
            print(f"❌ 파일 업로드 실패: {e}")
    
    async def _test_analysis_execution(self, page):
        """분석 실행 테스트"""
        print("\n4️⃣ 분석 실행 테스트")
        
        try:
            # 분석 버튼 또는 입력 필드 찾기
            # Streamlit 텍스트 입력 찾기
            text_input = await page.query_selector('[data-testid="stTextInput"] input')
            
            if text_input:
                # 간단한 분석 요청 입력
                await text_input.fill("데이터를 분석해주세요. 기본 통계와 시각화를 보여주세요.")
                await page.keyboard.press('Enter')
                
                # 분석 실행 대기 (최대 30초)
                await page.wait_for_timeout(5000)
                
                # 결과 또는 진행 상황 확인
                progress_elements = await page.query_selector_all('[data-testid="stProgress"]')
                result_elements = await page.query_selector_all('[data-testid="stMarkdown"]')
                
                success = len(progress_elements) > 0 or len(result_elements) > 2
                
                self._log_test("분석 실행", success, f"진행요소: {len(progress_elements)}, 결과요소: {len(result_elements)}")
                
                if success:
                    print("✅ 분석 실행 시작 확인")
                else:
                    print("⚠️ 분석 실행 상태 불명확")
                    
            else:
                self._log_test("분석 실행", False, "텍스트 입력 필드를 찾을 수 없음")
                print("❌ 분석 입력 필드를 찾을 수 없음")
                
        except Exception as e:
            self._log_test("분석 실행", False, f"오류: {str(e)}")
            print(f"❌ 분석 실행 테스트 실패: {e}")
    
    async def _test_results_display(self, page):
        """결과 표시 테스트"""
        print("\n5️⃣ 결과 표시 테스트")
        
        try:
            # 결과 표시 대기 (최대 60초)
            print("⏳ 분석 결과 대기 중...")
            
            # 다양한 결과 요소들 확인
            await page.wait_for_timeout(10000)  # 10초 대기
            
            # 차트/그래프 요소 확인
            chart_elements = await page.query_selector_all('[data-testid="stPlotlyChart"]')
            
            # 테이블 요소 확인  
            table_elements = await page.query_selector_all('[data-testid="stDataFrame"]')
            
            # 텍스트 결과 확인
            markdown_elements = await page.query_selector_all('[data-testid="stMarkdown"]')
            
            # 성공 기준: 차트, 테이블, 또는 상당한 텍스트 결과 중 하나라도 있으면 성공
            success = len(chart_elements) > 0 or len(table_elements) > 0 or len(markdown_elements) > 5
            
            details = f"차트: {len(chart_elements)}, 테이블: {len(table_elements)}, 텍스트: {len(markdown_elements)}"
            
            self._log_test("결과 표시", success, details)
            
            if success:
                print("✅ 분석 결과 표시 확인")
            else:
                print("⚠️ 분석 결과 표시 불충분")
                
        except Exception as e:
            self._log_test("결과 표시", False, f"오류: {str(e)}")
            print(f"❌ 결과 표시 테스트 실패: {e}")
    
    def _log_test(self, test_name: str, success: bool, details: str = ""):
        """테스트 결과 로깅"""
        self.results["tests"].append({
            "name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

async def main():
    """메인 테스트 실행"""
    tester = CherryAIPlaywrightTest()
    results = await tester.run_comprehensive_test()
    
    # 결과 파일 저장
    results_file = f"playwright_test_results_{int(time.time())}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 결과 저장: {results_file}")
    
    # 최종 상태 반환
    if results["overall_success"]:
        print("🎉 모든 테스트 통과!")
        return True
    else:
        print("⚠️ 일부 테스트 실패")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 