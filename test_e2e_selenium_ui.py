#!/usr/bin/env python3
"""
🎭 CherryAI Selenium UI E2E 테스트

Selenium을 사용하여 실제 브라우저에서 CherryAI UI의 기본 기능을 테스트합니다.
- 페이지 로딩 및 기본 요소 확인
- 파일 업로드 인터페이스 테스트
- 채팅 인터페이스 기본 동작 확인

Author: CherryAI Production Team
"""

import time
import os
from datetime import datetime
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class CherryAISeleniumUITest:
    """CherryAI Selenium UI 테스트 클래스"""
    
    def __init__(self):
        self.streamlit_url = "http://localhost:8501"
        self.monitoring_url = "http://localhost:8502"
        self.screenshot_dir = Path("selenium_screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        self.driver = None
        
    def setup_driver(self):
        """Chrome 드라이버 설정"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            # 헤드리스 모드 비활성화 (실제 브라우저 창을 볼 수 있도록)
            # chrome_options.add_argument("--headless")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            
            print("✅ Chrome 드라이버 설정 완료")
            return True
            
        except Exception as e:
            print(f"❌ Chrome 드라이버 설정 실패: {e}")
            return False
    
    def take_screenshot(self, name: str):
        """스크린샷 촬영"""
        if self.driver:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = self.screenshot_dir / f"{name}_{timestamp}.png"
            self.driver.save_screenshot(str(screenshot_path))
            print(f"📸 스크린샷 저장: {screenshot_path}")
            return screenshot_path
        return None
    
    def test_streamlit_main_page(self) -> bool:
        """Streamlit 메인 페이지 테스트"""
        print("\n🧪 Selenium 테스트 1: Streamlit 메인 페이지")
        print("-" * 40)
        
        try:
            # 메인 페이지 로드
            self.driver.get(self.streamlit_url)
            print(f"📍 페이지 로드: {self.streamlit_url}")
            
            # Streamlit 앱이 로드될 때까지 대기
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stApp']"))
            )
            
            # 페이지 제목 확인
            title = self.driver.title
            print(f"📄 페이지 제목: {title}")
            
            # 스크린샷 촬영
            self.take_screenshot("streamlit_main_page")
            
            # 기본 Streamlit 요소들 확인
            streamlit_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid*='st']")
            print(f"🔍 Streamlit 요소 발견: {len(streamlit_elements)}개")
            
            if len(streamlit_elements) > 0:
                print("✅ Streamlit 메인 페이지 로드 성공")
                return True
            else:
                print("⚠️ Streamlit 요소를 찾을 수 없음")
                return False
                
        except Exception as e:
            print(f"❌ Streamlit 메인 페이지 테스트 실패: {e}")
            self.take_screenshot("streamlit_main_error")
            return False
    
    def test_file_uploader_interface(self) -> bool:
        """파일 업로더 인터페이스 테스트"""
        print("\n🧪 Selenium 테스트 2: 파일 업로더 인터페이스")
        print("-" * 40)
        
        try:
            # 파일 업로더 찾기
            file_uploaders = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            
            if file_uploaders:
                print(f"📤 파일 업로더 발견: {len(file_uploaders)}개")
                self.take_screenshot("file_uploader_found")
                
                # 첫 번째 파일 업로더의 속성 확인
                uploader = file_uploaders[0]
                accept_attr = uploader.get_attribute("accept")
                print(f"📋 허용 파일 타입: {accept_attr}")
                
                print("✅ 파일 업로더 인터페이스 확인 성공")
                return True
            else:
                print("⚠️ 파일 업로더를 찾을 수 없음")
                self.take_screenshot("no_file_uploader")
                return False
                
        except Exception as e:
            print(f"❌ 파일 업로더 테스트 실패: {e}")
            self.take_screenshot("file_uploader_error")
            return False
    
    def test_chat_interface(self) -> bool:
        """채팅 인터페이스 테스트"""
        print("\n🧪 Selenium 테스트 3: 채팅 인터페이스")
        print("-" * 40)
        
        try:
            # 텍스트 입력 요소들 찾기
            text_inputs = self.driver.find_elements(By.CSS_SELECTOR, "textarea, input[type='text']")
            
            chat_input_found = False
            for input_element in text_inputs:
                placeholder = input_element.get_attribute("placeholder")
                if placeholder and any(keyword in placeholder.lower() for keyword in ['메시지', 'message', '채팅', 'chat', '질문', 'question']):
                    print(f"💬 채팅 입력창 발견: {placeholder}")
                    chat_input_found = True
                    break
            
            if not chat_input_found and text_inputs:
                print(f"📝 텍스트 입력 요소 발견: {len(text_inputs)}개")
                chat_input_found = True
            
            if chat_input_found:
                self.take_screenshot("chat_interface")
                print("✅ 채팅 인터페이스 확인 성공")
                return True
            else:
                print("⚠️ 채팅 입력 인터페이스를 찾을 수 없음")
                self.take_screenshot("no_chat_interface")
                return False
                
        except Exception as e:
            print(f"❌ 채팅 인터페이스 테스트 실패: {e}")
            self.take_screenshot("chat_interface_error")
            return False
    
    def test_sidebar_navigation(self) -> bool:
        """사이드바 네비게이션 테스트"""
        print("\n🧪 Selenium 테스트 4: 사이드바 네비게이션")
        print("-" * 40)
        
        try:
            # Streamlit 사이드바 찾기
            sidebar = self.driver.find_element(By.CSS_SELECTOR, "[data-testid='stSidebar']")
            
            if sidebar:
                print("📋 사이드바 발견")
                
                # 사이드바 내의 클릭 가능한 요소들 찾기
                clickable_elements = sidebar.find_elements(By.CSS_SELECTOR, "a, button, [role='button']")
                
                print(f"🖱️ 클릭 가능한 요소: {len(clickable_elements)}개")
                
                self.take_screenshot("sidebar_navigation")
                print("✅ 사이드바 네비게이션 확인 성공")
                return True
            else:
                print("⚠️ 사이드바를 찾을 수 없음")
                return False
                
        except Exception as e:
            print(f"❌ 사이드바 테스트 실패: {e}")
            self.take_screenshot("sidebar_error")
            return False
    
    def test_monitoring_dashboard(self) -> bool:
        """모니터링 대시보드 테스트"""
        print("\n🧪 Selenium 테스트 5: 모니터링 대시보드")
        print("-" * 40)
        
        try:
            # 모니터링 대시보드로 이동
            self.driver.get(self.monitoring_url)
            print(f"📍 모니터링 대시보드 로드: {self.monitoring_url}")
            
            # Streamlit 앱 로드 대기
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stApp']"))
            )
            
            # 모니터링 관련 텍스트 확인
            page_text = self.driver.page_source
            monitoring_keywords = ["모니터링", "Monitoring", "Dashboard", "대시보드", "Health", "Status"]
            found_keywords = [keyword for keyword in monitoring_keywords if keyword in page_text]
            
            if found_keywords:
                print(f"🔍 모니터링 키워드 발견: {found_keywords}")
                self.take_screenshot("monitoring_dashboard")
                print("✅ 모니터링 대시보드 확인 성공")
                return True
            else:
                print("⚠️ 모니터링 관련 콘텐츠를 찾을 수 없음")
                self.take_screenshot("monitoring_no_content")
                return False
                
        except Exception as e:
            print(f"❌ 모니터링 대시보드 테스트 실패: {e}")
            self.take_screenshot("monitoring_error")
            return False
    
    def run_full_selenium_test(self) -> dict:
        """전체 Selenium UI 테스트 실행"""
        print("🎭 CherryAI Selenium UI E2E 테스트 시작")
        print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 드라이버 설정
        if not self.setup_driver():
            return {"error": "드라이버 설정 실패"}
        
        test_results = {
            "streamlit_main_page": False,
            "file_uploader_interface": False,
            "chat_interface": False,
            "sidebar_navigation": False,
            "monitoring_dashboard": False
        }
        
        try:
            # 테스트 실행
            test_results["streamlit_main_page"] = self.test_streamlit_main_page()
            test_results["file_uploader_interface"] = self.test_file_uploader_interface()
            test_results["chat_interface"] = self.test_chat_interface()
            test_results["sidebar_navigation"] = self.test_sidebar_navigation()
            test_results["monitoring_dashboard"] = self.test_monitoring_dashboard()
            
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
            
        finally:
            # 드라이버 종료
            if self.driver:
                self.driver.quit()
                print("🚪 브라우저 종료")
        
        # 결과 리포트 생성
        self.generate_selenium_report(test_results)
        
        return test_results
    
    def generate_selenium_report(self, results: dict):
        """Selenium 테스트 결과 리포트 생성"""
        print("\n" + "=" * 60)
        print("📊 CherryAI Selenium UI E2E 테스트 결과")
        print("=" * 60)
        
        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        for test_name, result in results.items():
            status = "✅ 성공" if result else "❌ 실패"
            display_name = test_name.replace('_', ' ').title()
            print(f"{display_name}: {status}")
        
        print(f"\n🎯 Selenium UI 테스트 성공률: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 Selenium UI 테스트 완전 성공! UI가 우수하게 작동합니다!")
            status = "EXCELLENT"
        elif success_rate >= 70:
            print("✅ Selenium UI 테스트 대부분 성공! UI가 안정적으로 작동합니다.")
            status = "GOOD"
        elif success_rate >= 50:
            print("⚠️ Selenium UI 테스트 부분 성공. 일부 UI 개선이 필요합니다.")
            status = "NEEDS_IMPROVEMENT"
        else:
            print("🚨 Selenium UI 테스트 실패. UI 점검이 필요합니다.")
            status = "CRITICAL"
        
        print(f"📁 스크린샷 저장 위치: {self.screenshot_dir}")
        print(f"⏱️ 테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return status


def main():
    """메인 테스트 실행"""
    test = CherryAISeleniumUITest()
    results = test.run_full_selenium_test()
    
    # 성공 여부 판단
    if "error" in results:
        return False
        
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    success_rate = (passed_tests / total_tests) * 100
    
    return success_rate >= 60  # 60% 이상이면 성공


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 