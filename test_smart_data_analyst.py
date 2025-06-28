#!/usr/bin/env python3
"""
Smart Data Analyst 자동 테스트 스크립트
Playwright MCP 대신 requests를 사용한 자동화 테스트
"""

import time
import requests
import json
import os
import pandas as pd

class SmartDataAnalystTester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.test_results = []
        
    def check_server_health(self):
        """서버 상태 확인"""
        print("🔍 서버 상태 확인 중...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                print("✅ Smart Data Analyst 서버 정상 응답")
                self.test_results.append(("서버 연결", True, "HTTP 200 응답"))
                return True
            else:
                print(f"❌ 서버 응답 오류: {response.status_code}")
                self.test_results.append(("서버 연결", False, f"HTTP {response.status_code}"))
                return False
        except Exception as e:
            print(f"❌ 서버 연결 실패: {e}")
            self.test_results.append(("서버 연결", False, str(e)))
            return False
    
    def check_a2a_servers(self):
        """A2A 서버들 상태 확인"""
        print("🔍 A2A 서버들 상태 확인 중...")
        
        servers = {
            "Orchestrator": "http://localhost:8100",
            "Pandas Data Analyst": "http://localhost:8200", 
            "EDA Tools": "http://localhost:8203",
            "Data Visualization": "http://localhost:8202"
        }
        
        server_status = {}
        for name, url in servers.items():
            try:
                response = requests.get(f"{url}/.well-known/agent.json", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {name}: 정상")
                    server_status[name] = True
                    self.test_results.append((f"A2A {name}", True, "에이전트 카드 응답"))
                else:
                    print(f"❌ {name}: HTTP {response.status_code}")
                    server_status[name] = False
                    self.test_results.append((f"A2A {name}", False, f"HTTP {response.status_code}"))
            except Exception as e:
                print(f"❌ {name}: 연결 실패")
                server_status[name] = False
                self.test_results.append((f"A2A {name}", False, "연결 실패"))
        
        return server_status
    
    def prepare_test_data(self):
        """테스트용 샘플 데이터 준비"""
        print("📊 테스트 데이터 준비 중...")
        
        test_data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Age': [25, 30, 35, 28, 32],
            'Salary': [50000, 60000, 70000, 55000, 65000],
            'Department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing'],
            'Experience': [2, 5, 8, 3, 6]
        }
        
        df = pd.DataFrame(test_data)
        
        os.makedirs("a2a_ds_servers/artifacts/data/shared_dataframes", exist_ok=True)
        test_file_path = "a2a_ds_servers/artifacts/data/shared_dataframes/test_data.csv"
        df.to_csv(test_file_path, index=False)
        
        print(f"✅ 테스트 데이터 생성 완료: {test_file_path}")
        self.test_results.append(("테스트 데이터 생성", True, "5행 5열 CSV 파일 생성"))
        return test_file_path
    
    def test_a2a_integration(self):
        """A2A 통합 테스트"""
        print("🤖 A2A 통합 테스트 중...")
        
        analysis_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "test_data.csv 파일의 기본 통계를 분석해주세요."}],
                    "messageId": f"test_{int(time.time())}"
                },
                "metadata": {}
            }
        }
        
        try:
            response = requests.post(
                "http://localhost:8200/",
                json=analysis_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    print("✅ A2A 분석 요청 성공")
                    self.test_results.append(("A2A 분석 요청", True, "분석 결과 수신"))
                else:
                    print("❌ A2A 응답에 결과 없음")
                    self.test_results.append(("A2A 분석 요청", False, "결과 없음"))
            else:
                print(f"❌ A2A 요청 실패: {response.status_code}")
                self.test_results.append(("A2A 분석 요청", False, f"HTTP {response.status_code}"))
                
        except Exception as e:
            print(f"❌ A2A 통합 테스트 실패: {e}")
            self.test_results.append(("A2A 분석 요청", False, str(e)))
    
    def generate_test_report(self):
        """테스트 결과 보고서 생성"""
        print("\n" + "="*70)
        print("📊 Smart Data Analyst 자동 테스트 결과 보고서")
        print("="*70)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r[1]])
        failed_tests = total_tests - passed_tests
        
        print(f"총 테스트: {total_tests}")
        print(f"성공: {passed_tests}")
        print(f"실패: {failed_tests}")
        print(f"성공률: {(passed_tests/total_tests*100):.1f}%")
        
        print("\n상세 결과:")
        print("-" * 70)
        for test_name, passed, details in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name:<25}: {details}")
        
        return passed_tests == total_tests
    
    def run_full_test_suite(self):
        """전체 테스트 실행"""
        print("🚀 Smart Data Analyst 자동 테스트 시작")
        print("Playwright MCP 대신 HTTP 요청 기반 테스트 수행")
        print("="*70)
        
        if not self.check_server_health():
            print("❌ 기본 서버 테스트 실패 - 테스트 중단")
            return False
        
        a2a_status = self.check_a2a_servers()
        active_servers = sum(a2a_status.values())
        total_servers = len(a2a_status)
        print(f"📊 A2A 서버 상태: {active_servers}/{total_servers} 활성")
        
        self.prepare_test_data()
        
        if active_servers > 0:
            self.test_a2a_integration()
        else:
            print("⚠️ A2A 서버가 비활성화되어 통합 테스트 건너뜀")
        
        success = self.generate_test_report()
        
        if success:
            print("\n🎉 모든 테스트 통과!")
        else:
            print("\n⚠️ 일부 테스트 실패")
        
        return success

def main():
    tester = SmartDataAnalystTester()
    success = tester.run_full_test_suite()
    return success

if __name__ == "__main__":
    main()
