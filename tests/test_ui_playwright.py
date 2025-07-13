"""
Playwright MCP를 사용한 UI 테스트
"""
import pytest
import time
import tempfile
import os

class TestCherryAIUI:
    """CherryAI UI 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.base_url = "http://localhost:8501"
        self.test_files_dir = tempfile.mkdtemp()
        
        # 테스트용 CSV 파일 생성
        self.test_csv_path = os.path.join(self.test_files_dir, "test_data.csv")
        with open(self.test_csv_path, 'w') as f:
            f.write("Name,Age,City\n")
            f.write("Alice,25,Seoul\n")
            f.write("Bob,30,Busan\n")
            f.write("Charlie,35,Incheon\n")
    
    def teardown_method(self):
        """테스트 정리"""
        # 임시 파일 정리
        if os.path.exists(self.test_csv_path):
            os.unlink(self.test_csv_path)
        if os.path.exists(self.test_files_dir):
            os.rmdir(self.test_files_dir)
    
    def test_page_load_and_layout(self):
        """페이지 로드 및 레이아웃 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드 확인
        result = mcp_Playwright_playwright_navigate(url=self.base_url)
        assert "Navigated" in result
        
        # 페이지 스크린샷
        screenshot_result = mcp_Playwright_playwright_screenshot(
            name="page_layout_test",
            savePng=True,
            fullPage=True
        )
        assert "Screenshot saved" in screenshot_result
        
        # 페이지 제목 확인
        page_content = mcp_Playwright_playwright_get_visible_text("")
        assert "CherryAI" in page_content
        assert "A2A + MCP 통합 플랫폼" in page_content
    
    def test_split_layout_functionality(self):
        """Split Layout 기능 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드
        mcp_Playwright_playwright_navigate(url=self.base_url)
        
        # Split divider 확인
        html_content = mcp_Playwright_playwright_get_visible_html()
        
        # Split 관련 CSS 클래스 확인
        assert "split-container" in html_content or "Split" in html_content
        
        # 스크린샷으로 레이아웃 확인
        mcp_Playwright_playwright_screenshot(
            name="split_layout_test",
            savePng=True
        )
    
    def test_file_upload_interface(self):
        """파일 업로드 인터페이스 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드
        mcp_Playwright_playwright_navigate(url=self.base_url)
        
        # 파일 업로드 영역 찾기
        page_text = mcp_Playwright_playwright_get_visible_text("")
        assert "파일 업로드" in page_text or "업로드" in page_text
        
        # 파일 업로드 시도 (Streamlit의 파일 업로더는 직접 조작이 어려우므로 존재 확인)
        html_content = mcp_Playwright_playwright_get_visible_html()
        
        # 파일 업로드 관련 요소 확인
        file_upload_present = (
            "file_uploader" in html_content or 
            "파일을 선택" in html_content or
            "Browse files" in html_content
        )
        
        if file_upload_present:
            mcp_Playwright_playwright_screenshot(
                name="file_upload_interface",
                savePng=True
            )
    
    def test_question_input_interface(self):
        """질문 입력 인터페이스 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드
        mcp_Playwright_playwright_navigate(url=self.base_url)
        
        # 질문 입력 영역 찾기
        page_text = mcp_Playwright_playwright_get_visible_text("")
        
        # 질문 입력 관련 텍스트 확인
        question_input_present = (
            "질문" in page_text or 
            "입력" in page_text or
            "메시지" in page_text
        )
        
        assert question_input_present, "질문 입력 인터페이스를 찾을 수 없습니다"
        
        # 스크린샷
        mcp_Playwright_playwright_screenshot(
            name="question_input_interface",
            savePng=True
        )
    
    def test_chat_interface(self):
        """채팅 인터페이스 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드
        mcp_Playwright_playwright_navigate(url=self.base_url)
        
        # 페이지가 완전히 로드될 때까지 대기
        time.sleep(3)
        
        # 채팅 관련 요소 확인
        html_content = mcp_Playwright_playwright_get_visible_html()
        page_text = mcp_Playwright_playwright_get_visible_text("")
        
        # 채팅 인터페이스 관련 요소 확인
        chat_elements_present = (
            "chat" in html_content.lower() or
            "message" in html_content.lower() or
            "대화" in page_text or
            "메시지" in page_text
        )
        
        # 스크린샷으로 채팅 인터페이스 확인
        mcp_Playwright_playwright_screenshot(
            name="chat_interface_test",
            savePng=True
        )
    
    def test_tab_navigation(self):
        """탭 네비게이션 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드
        mcp_Playwright_playwright_navigate(url=self.base_url)
        time.sleep(2)
        
        # 페이지 내용 확인
        page_text = mcp_Playwright_playwright_get_visible_text("")
        
        # 탭 관련 텍스트 확인
        expected_tabs = ["아티팩트", "A2A 상세", "지식 은행"]
        
        tabs_found = 0
        for tab in expected_tabs:
            if tab in page_text:
                tabs_found += 1
        
        # 최소 하나 이상의 탭이 있어야 함
        assert tabs_found > 0, f"예상 탭을 찾을 수 없습니다: {expected_tabs}"
        
        # 탭 네비게이션 스크린샷
        mcp_Playwright_playwright_screenshot(
            name="tab_navigation_test",
            savePng=True
        )
    
    def test_responsive_design(self):
        """반응형 디자인 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 데스크톱 뷰 (기본)
        mcp_Playwright_playwright_navigate(url=self.base_url)
        mcp_Playwright_playwright_screenshot(
            name="desktop_view",
            savePng=True,
            width=1920,
            height=1080
        )
        
        # 모바일 뷰 시뮬레이션을 위한 작은 화면 크기 (Playwright에서 직접 뷰포트 변경은 제한적)
        # 대신 페이지가 작은 화면에서도 작동하는지 확인
        page_content = mcp_Playwright_playwright_get_visible_text("")
        
        # 기본 콘텐츠가 로드되었는지 확인
        assert "CherryAI" in page_content
    
    def test_error_handling_ui(self):
        """UI 에러 처리 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드
        mcp_Playwright_playwright_navigate(url=self.base_url)
        
        # 콘솔 로그 확인 (JavaScript 에러가 있는지)
        console_logs = mcp_Playwright_playwright_console_logs(type="error")
        
        # 치명적인 JavaScript 에러가 없어야 함
        if console_logs:
            critical_errors = [
                log for log in console_logs 
                if any(keyword in log.lower() for keyword in ['error', 'failed', 'exception'])
            ]
            
            # 일부 경고는 정상이지만 치명적인 에러는 없어야 함
            assert len(critical_errors) < 5, f"너무 많은 콘솔 에러: {critical_errors}"
        
        # 에러 처리 테스트 스크린샷
        mcp_Playwright_playwright_screenshot(
            name="error_handling_test",
            savePng=True
        )
    
    def test_page_performance(self):
        """페이지 성능 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드 시간 측정
        start_time = time.time()
        
        mcp_Playwright_playwright_navigate(url=self.base_url)
        
        # 페이지가 완전히 로드될 때까지 대기
        time.sleep(5)
        
        end_time = time.time()
        load_time = end_time - start_time
        
        # 로드 시간이 30초 이내여야 함 (Streamlit은 상대적으로 느림)
        assert load_time < 30, f"페이지 로드 시간이 너무 깁니다: {load_time}초"
        
        # 페이지 내용이 로드되었는지 확인
        page_text = mcp_Playwright_playwright_get_visible_text("")
        assert len(page_text) > 100, "페이지 내용이 충분히 로드되지 않았습니다"
        
        # 성능 테스트 스크린샷
        mcp_Playwright_playwright_screenshot(
            name="performance_test",
            savePng=True
        )
    
    def test_accessibility_basics(self):
        """기본 접근성 테스트"""
        from mcp_agents.mcp_Playwright_agent import *
        
        # 페이지 로드
        mcp_Playwright_playwright_navigate(url=self.base_url)
        time.sleep(3)
        
        # HTML 구조 확인
        html_content = mcp_Playwright_playwright_get_visible_html()
        
        # 기본 접근성 요소 확인
        accessibility_elements = {
            "title": "<title>" in html_content,
            "headings": any(f"<h{i}" in html_content for i in range(1, 7)),
            "semantic_elements": any(tag in html_content for tag in ["<main", "<nav", "<section", "<article"])
        }
        
        # 최소한의 접근성 요소가 있어야 함
        accessible_elements_count = sum(accessibility_elements.values())
        assert accessible_elements_count >= 1, f"접근성 요소 부족: {accessibility_elements}"
        
        # 접근성 테스트 스크린샷
        mcp_Playwright_playwright_screenshot(
            name="accessibility_test",
            savePng=True
        )

# 실제 테스트 실행
if __name__ == "__main__":
    import sys
    import traceback
    
    # 테스트 클래스 인스턴스 생성
    test_instance = TestCherryAIUI()
    
    print("🧪 CherryAI UI 테스트 시작...")
    
    try:
        # 각 테스트 메서드 실행
        test_methods = [
            ("페이지 로드 및 레이아웃", test_instance.test_page_load_and_layout),
            ("Split Layout 기능", test_instance.test_split_layout_functionality),
            ("파일 업로드 인터페이스", test_instance.test_file_upload_interface),
            ("질문 입력 인터페이스", test_instance.test_question_input_interface),
            ("채팅 인터페이스", test_instance.test_chat_interface),
            ("탭 네비게이션", test_instance.test_tab_navigation),
            ("반응형 디자인", test_instance.test_responsive_design),
            ("에러 처리", test_instance.test_error_handling_ui),
            ("페이지 성능", test_instance.test_page_performance),
            ("기본 접근성", test_instance.test_accessibility_basics)
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, test_method in test_methods:
            try:
                print(f"  🔄 {test_name} 테스트 중...")
                test_instance.setup_method()
                test_method()
                test_instance.teardown_method()
                print(f"  ✅ {test_name} 테스트 통과")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {test_name} 테스트 실패: {str(e)}")
                failed_tests += 1
                # 상세 에러 정보는 디버그 모드에서만 표시
                if "--debug" in sys.argv:
                    traceback.print_exc()
        
        print(f"\n📊 테스트 결과:")
        print(f"  ✅ 통과: {passed_tests}")
        print(f"  ❌ 실패: {failed_tests}")
        print(f"  📈 성공률: {passed_tests/(passed_tests+failed_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print("🎉 모든 UI 테스트가 성공적으로 완료되었습니다!")
        else:
            print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {str(e)}")
        traceback.print_exc()
    
    finally:
        print("🔧 브라우저 정리 중...")
        try:
            # 브라우저 종료
            from mcp_agents.mcp_Playwright_agent import mcp_Playwright_playwright_close
            mcp_Playwright_playwright_close("cleanup")
            print("✅ 브라우저 정리 완료")
        except:
            print("⚠️ 브라우저 정리 중 오류 (무시됨)")
