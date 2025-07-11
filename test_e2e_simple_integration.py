#!/usr/bin/env python3
"""
🧪 CherryAI 간단 E2E 통합 테스트

HTTP 요청 기반으로 CherryAI 시스템의 기본 기능을 테스트합니다.
- 웹 서버 상태 확인
- A2A 에이전트 연결 상태 확인
- 기본 API 엔드포인트 테스트

Author: CherryAI Production Team
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Any

class CherryAISimpleE2ETest:
    """CherryAI 간단 E2E 테스트 클래스"""
    
    def __init__(self):
        self.streamlit_url = "http://localhost:8501"
        self.monitoring_url = "http://localhost:8502"
        self.orchestrator_url = "http://localhost:8100"
        
        # A2A 에이전트 포트 목록
        self.a2a_agents = {
            "Orchestrator": 8100,
            "Data Cleaning": 8306,
            "Data Loader": 8307,
            "Data Visualization": 8308,
            "Data Wrangling": 8309,
            "Feature Engineering": 8310,
            "SQL Database": 8311,
            "EDA Tools": 8312,
            "H2O ML": 8313,
            "MLflow Tools": 8314
        }
    
    def test_web_server_availability(self) -> Dict[str, Any]:
        """웹 서버 가용성 테스트"""
        print("\n🧪 테스트 1: 웹 서버 가용성")
        print("-" * 40)
        
        results = {}
        
        # Streamlit 메인 앱 테스트
        try:
            response = requests.get(self.streamlit_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Streamlit 메인 앱: {self.streamlit_url}")
                results["streamlit_main"] = True
            else:
                print(f"❌ Streamlit 메인 앱: HTTP {response.status_code}")
                results["streamlit_main"] = False
        except Exception as e:
            print(f"❌ Streamlit 메인 앱 연결 실패: {e}")
            results["streamlit_main"] = False
        
        # 모니터링 대시보드 테스트
        try:
            response = requests.get(self.monitoring_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ 모니터링 대시보드: {self.monitoring_url}")
                results["monitoring_dashboard"] = True
            else:
                print(f"❌ 모니터링 대시보드: HTTP {response.status_code}")
                results["monitoring_dashboard"] = False
        except Exception as e:
            print(f"❌ 모니터링 대시보드 연결 실패: {e}")
            results["monitoring_dashboard"] = False
        
        return results
    
    def test_a2a_agents_connectivity(self) -> Dict[str, Any]:
        """A2A 에이전트 연결성 테스트"""
        print("\n🧪 테스트 2: A2A 에이전트 연결성")
        print("-" * 40)
        
        results = {}
        healthy_count = 0
        
        for agent_name, port in self.a2a_agents.items():
            try:
                # Agent card 확인
                url = f"http://localhost:{port}/.well-known/agent.json"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    agent_info = response.json()
                    agent_display_name = agent_info.get("name", agent_name)
                    print(f"✅ {agent_name} (port {port}): {agent_display_name}")
                    results[agent_name] = {
                        "status": "healthy",
                        "name": agent_display_name,
                        "port": port
                    }
                    healthy_count += 1
                else:
                    print(f"❌ {agent_name} (port {port}): HTTP {response.status_code}")
                    results[agent_name] = {
                        "status": "error",
                        "port": port,
                        "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                print(f"⚠️ {agent_name} (port {port}): 연결 실패 - {str(e)}")
                results[agent_name] = {
                    "status": "offline",
                    "port": port,
                    "error": str(e)
                }
        
        print(f"\n📊 A2A 에이전트 상태: {healthy_count}/{len(self.a2a_agents)} 정상")
        results["summary"] = {
            "total": len(self.a2a_agents),
            "healthy": healthy_count,
            "success_rate": (healthy_count / len(self.a2a_agents)) * 100
        }
        
        return results
    
    def test_orchestrator_functionality(self) -> Dict[str, Any]:
        """오케스트레이터 기능 테스트"""
        print("\n🧪 테스트 3: 오케스트레이터 기능")
        print("-" * 40)
        
        try:
            # Agent card 확인
            url = f"{self.orchestrator_url}/.well-known/agent.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                agent_info = response.json()
                print(f"✅ 오케스트레이터 이름: {agent_info.get('name')}")
                print(f"✅ 스킬 수: {len(agent_info.get('skills', []))}")
                print(f"✅ 스트리밍 지원: {agent_info.get('capabilities', {}).get('streaming', False)}")
                
                return {
                    "status": "healthy",
                    "name": agent_info.get("name"),
                    "skills": len(agent_info.get("skills", [])),
                    "streaming": agent_info.get("capabilities", {}).get("streaming", False),
                    "capabilities": agent_info.get("capabilities", {})
                }
            else:
                print(f"❌ 오케스트레이터 Agent Card: HTTP {response.status_code}")
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            print(f"❌ 오케스트레이터 테스트 실패: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_system_health_endpoints(self) -> Dict[str, Any]:
        """시스템 헬스 엔드포인트 테스트"""
        print("\n🧪 테스트 4: 시스템 헬스 체크")
        print("-" * 40)
        
        results = {}
        
        # 각 서비스의 기본 응답 확인
        services = {
            "streamlit_main": self.streamlit_url,
            "monitoring": self.monitoring_url,
            "orchestrator": self.orchestrator_url
        }
        
        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service_name}: 정상 응답")
                    results[service_name] = True
                else:
                    print(f"⚠️ {service_name}: HTTP {response.status_code}")
                    results[service_name] = False
            except Exception as e:
                print(f"❌ {service_name}: 연결 실패")
                results[service_name] = False
        
        return results
    
    def test_data_flow_simulation(self) -> Dict[str, Any]:
        """데이터 플로우 시뮬레이션 테스트"""
        print("\n🧪 테스트 5: 데이터 플로우 시뮬레이션")
        print("-" * 40)
        
        try:
            # 간단한 데이터 처리 시뮬레이션
            # (실제 파일 업로드는 브라우저 테스트가 필요하므로 API 레벨에서 테스트)
            
            print("📊 데이터 처리 파이프라인 검증:")
            print("   1. 데이터 로더 에이전트 상태 확인")
            data_loader_status = requests.get("http://localhost:8307/.well-known/agent.json", timeout=5)
            
            print("   2. 데이터 클리닝 에이전트 상태 확인")
            cleaning_status = requests.get("http://localhost:8306/.well-known/agent.json", timeout=5)
            
            print("   3. EDA 도구 에이전트 상태 확인")
            eda_status = requests.get("http://localhost:8312/.well-known/agent.json", timeout=5)
            
            pipeline_health = all([
                data_loader_status.status_code == 200,
                cleaning_status.status_code == 200,
                eda_status.status_code == 200
            ])
            
            if pipeline_health:
                print("✅ 데이터 처리 파이프라인 정상")
                return {"status": "healthy", "pipeline": True}
            else:
                print("⚠️ 데이터 처리 파이프라인 일부 문제")
                return {"status": "partial", "pipeline": False}
                
        except Exception as e:
            print(f"❌ 데이터 플로우 테스트 실패: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_full_e2e_test(self) -> Dict[str, Any]:
        """전체 E2E 테스트 실행"""
        print("🧪 CherryAI 간단 E2E 통합 테스트 시작")
        print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        all_results = {}
        
        # 테스트 실행
        all_results["web_servers"] = self.test_web_server_availability()
        all_results["a2a_agents"] = self.test_a2a_agents_connectivity()
        all_results["orchestrator"] = self.test_orchestrator_functionality()
        all_results["health_endpoints"] = self.test_system_health_endpoints()
        all_results["data_flow"] = self.test_data_flow_simulation()
        
        # 결과 리포트 생성
        self.generate_test_report(all_results)
        
        return all_results
    
    def generate_test_report(self, results: Dict[str, Any]):
        """테스트 결과 리포트 생성"""
        print("\n" + "=" * 60)
        print("📊 CherryAI 간단 E2E 테스트 결과 리포트")
        print("=" * 60)
        
        # 웹 서버 상태
        web_results = results["web_servers"]
        web_success = sum(1 for v in web_results.values() if v)
        print(f"🌐 웹 서버: {web_success}/{len(web_results)} 정상")
        
        # A2A 에이전트 상태
        a2a_results = results["a2a_agents"]
        if "summary" in a2a_results:
            a2a_summary = a2a_results["summary"]
            print(f"🤖 A2A 에이전트: {a2a_summary['healthy']}/{a2a_summary['total']} 정상 ({a2a_summary['success_rate']:.1f}%)")
        
        # 오케스트레이터 상태
        orch_results = results["orchestrator"]
        orch_status = "✅ 정상" if orch_results.get("status") == "healthy" else "❌ 오류"
        print(f"🎯 오케스트레이터: {orch_status}")
        
        # 헬스 엔드포인트 상태
        health_results = results["health_endpoints"]
        health_success = sum(1 for v in health_results.values() if v)
        print(f"💚 헬스 체크: {health_success}/{len(health_results)} 정상")
        
        # 데이터 플로우 상태
        flow_results = results["data_flow"]
        flow_status = "✅ 정상" if flow_results.get("status") == "healthy" else "❌ 오류"
        print(f"📊 데이터 플로우: {flow_status}")
        
        # 전체 평가
        total_tests = 5
        successful_tests = 0
        
        if web_success >= len(web_results) * 0.8:  # 80% 이상 성공
            successful_tests += 1
        if a2a_results.get("summary", {}).get("success_rate", 0) >= 80:
            successful_tests += 1
        if orch_results.get("status") == "healthy":
            successful_tests += 1
        if health_success >= len(health_results) * 0.8:
            successful_tests += 1
        if flow_results.get("status") == "healthy":
            successful_tests += 1
        
        overall_success = (successful_tests / total_tests) * 100
        
        print(f"\n🎯 전체 E2E 테스트 성공률: {overall_success:.1f}%")
        
        if overall_success >= 90:
            print("🎉 E2E 테스트 완전 성공! CherryAI 시스템이 우수하게 작동합니다!")
            status = "EXCELLENT"
        elif overall_success >= 70:
            print("✅ E2E 테스트 대부분 성공! 시스템이 안정적으로 작동합니다.")
            status = "GOOD"
        elif overall_success >= 50:
            print("⚠️ E2E 테스트 부분 성공. 일부 개선이 필요합니다.")
            status = "NEEDS_IMPROVEMENT"
        else:
            print("🚨 E2E 테스트 실패. 시스템 점검이 필요합니다.")
            status = "CRITICAL"
        
        print(f"⏱️ 테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return status


def main():
    """메인 테스트 실행"""
    test = CherryAISimpleE2ETest()
    results = test.run_full_e2e_test()
    
    # 성공 여부 판단
    success_indicators = [
        results["web_servers"].get("streamlit_main", False),
        results["a2a_agents"].get("summary", {}).get("success_rate", 0) >= 80,
        results["orchestrator"].get("status") == "healthy"
    ]
    
    overall_success = sum(success_indicators) >= 2  # 3개 중 2개 이상 성공
    
    return overall_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 