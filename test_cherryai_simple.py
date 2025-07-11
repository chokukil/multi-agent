#!/usr/bin/env python3
"""
CherryAI Simple HTTP-based Testing
Playwright 문제로 인한 대안 테스트

Author: CherryAI Team
"""

import requests
import time
import json
import os
from datetime import datetime

class CherryAISimpleTest:
    """간단한 HTTP 기반 CherryAI 테스트"""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.a2a_ports = [8100, 8200, 8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314]
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": [],
            "overall_success": False,
            "errors": []
        }
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("🧪 CherryAI Simple HTTP-based Testing")
        print("=" * 60)
        
        # 1. Streamlit UI 접속 테스트
        self._test_streamlit_accessibility()
        
        # 2. A2A 서버들 상태 확인
        self._test_a2a_servers()
        
        # 3. 시스템 구성 요소 확인
        self._test_system_components()
        
        # 4. 파일 시스템 검증
        self._test_file_system()
        
        # 결과 계산
        success_count = sum(1 for test in self.results["tests"] if test["success"])
        total_count = len(self.results["tests"])
        self.results["overall_success"] = success_count >= total_count * 0.8  # 80% 이상 성공
        
        print(f"\n📊 테스트 결과: {success_count}/{total_count} 성공")
        
        return self.results
    
    def _test_streamlit_accessibility(self):
        """Streamlit UI 접근성 테스트"""
        print("\n1️⃣ Streamlit UI 접근성 테스트")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            success = response.status_code == 200
            
            if success:
                print("✅ Streamlit UI 정상 접근")
                content_length = len(response.content)
                self._log_test("Streamlit 접근", True, f"상태코드: {response.status_code}, 크기: {content_length}bytes")
            else:
                print(f"❌ Streamlit UI 접근 실패: {response.status_code}")
                self._log_test("Streamlit 접근", False, f"상태코드: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Streamlit UI 접근 오류: {e}")
            self._log_test("Streamlit 접근", False, f"오류: {str(e)}")
    
    def _test_a2a_servers(self):
        """A2A 서버들 상태 테스트"""
        print("\n2️⃣ A2A 서버 상태 테스트")
        
        active_servers = 0
        total_servers = len(self.a2a_ports)
        
        for port in self.a2a_ports:
            try:
                # A2A Agent Card 엔드포인트 확인
                url = f"http://localhost:{port}/.well-known/agent.json"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    active_servers += 1
                    agent_data = response.json()
                    agent_name = agent_data.get("name", "Unknown")
                    print(f"✅ 포트 {port}: {agent_name}")
                else:
                    print(f"❌ 포트 {port}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 포트 {port}: 연결 실패")
        
        success = active_servers >= total_servers * 0.7  # 70% 이상 활성화
        self._log_test("A2A 서버 상태", success, f"{active_servers}/{total_servers} 서버 활성")
        
        if success:
            print(f"✅ A2A 서버 상태 양호: {active_servers}/{total_servers}")
        else:
            print(f"⚠️ A2A 서버 상태 불량: {active_servers}/{total_servers}")
    
    def _test_system_components(self):
        """시스템 구성 요소 테스트"""
        print("\n3️⃣ 시스템 구성 요소 테스트")
        
        # 핵심 Python 모듈 import 테스트
        core_modules = [
            "core.user_file_tracker",
            "core.enhanced_langfuse_tracer", 
            "core.session_data_manager",
            "core.universal_data_analysis_router",
            "core.specialized_data_agents",
            "core.multi_agent_orchestrator",
            "core.auto_data_profiler",
            "core.advanced_code_tracker",
            "core.intelligent_result_interpreter"
        ]
        
        import sys
        sys.path.insert(0, os.getcwd())
        
        working_modules = 0
        for module_name in core_modules:
            try:
                __import__(module_name)
                working_modules += 1
                print(f"✅ {module_name}")
            except ImportError as e:
                print(f"❌ {module_name}: {e}")
            except Exception as e:
                print(f"⚠️ {module_name}: {e}")
        
        success = working_modules >= len(core_modules) * 0.8
        self._log_test("시스템 구성 요소", success, f"{working_modules}/{len(core_modules)} 모듈 정상")
    
    def _test_file_system(self):
        """파일 시스템 및 데이터 구조 테스트"""
        print("\n4️⃣ 파일 시스템 테스트")
        
        # 중요 디렉터리 확인
        important_dirs = [
            "core",
            "ui", 
            "a2a_ds_servers",
            "artifacts",
            "logs"
        ]
        
        existing_dirs = 0
        for dirname in important_dirs:
            if os.path.exists(dirname) and os.path.isdir(dirname):
                existing_dirs += 1
                print(f"✅ {dirname}/ 디렉터리 존재")
            else:
                print(f"❌ {dirname}/ 디렉터리 없음")
        
        # 중요 파일 확인
        important_files = [
            "ai.py",
            "a2a_orchestrator.py",
            "ai_ds_team_system_start.sh",
            "ai_ds_team_system_stop.sh"
        ]
        
        existing_files = 0
        for filename in important_files:
            if os.path.exists(filename):
                existing_files += 1
                print(f"✅ {filename} 파일 존재")
            else:
                print(f"❌ {filename} 파일 없음")
        
        total_items = len(important_dirs) + len(important_files)
        existing_items = existing_dirs + existing_files
        success = existing_items >= total_items * 0.9
        
        self._log_test("파일 시스템", success, f"{existing_items}/{total_items} 항목 확인")
    
    def _log_test(self, test_name: str, success: bool, details: str = ""):
        """테스트 결과 로깅"""
        self.results["tests"].append({
            "name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

def main():
    """메인 테스트 실행"""
    tester = CherryAISimpleTest()
    results = tester.run_comprehensive_test()
    
    # 결과 파일 저장
    results_file = f"simple_test_results_{int(time.time())}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 결과 저장: {results_file}")
    
    # 최종 상태 출력
    if results["overall_success"]:
        print("🎉 시스템 상태 양호!")
        return True
    else:
        print("⚠️ 시스템 일부 문제 발견")
        return False

if __name__ == "__main__":
    main() 