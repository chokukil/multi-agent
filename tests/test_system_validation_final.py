#!/usr/bin/env python3
"""
CherryAI v9 System Validation - Final Comprehensive Test
Phase 5: Complete System Validation
"""

import asyncio
import httpx
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CherryAI_v9_SystemValidator:
    """CherryAI v9 시스템 종합 검증"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()
        
        # Component endpoints
        self.endpoints = {
            "v9_orchestrator": "http://localhost:8100",
            "python_repl_agent": "http://localhost:8315",
            "streamlit_ui": "http://localhost:8501",
            "langfuse_dashboard": "http://mangugil.synology.me:3001"
        }
        
        # AI DS Team agents
        self.ai_ds_agents = {
            "data_cleaning": 8306,
            "data_loader": 8307,
            "data_visualization": 8308,
            "data_wrangling": 8309,
            "feature_engineering": 8310,
            "sql_database": 8311,
            "eda_tools": 8312,
            "h2o_ml": 8313,
            "mlflow_tools": 8314
        }
    
    def log_test_result(self, test_name: str, status: str, message: str = "", details: Dict = None):
        """테스트 결과 로깅"""
        result = {
            "test_name": test_name,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.test_results.append(result)
        
        # Console output
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {message}")
    
    async def test_component_health(self, name: str, url: str) -> bool:
        """컴포넌트 헬스 체크"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/.well-known/agent.json")
                if response.status_code == 200:
                    agent_info = response.json()
                    self.log_test_result(
                        f"Health Check - {name}",
                        "PASS",
                        f"Agent '{agent_info.get('name', 'Unknown')}' is healthy",
                        {"agent_info": agent_info}
                    )
                    return True
                else:
                    self.log_test_result(
                        f"Health Check - {name}",
                        "FAIL",
                        f"HTTP {response.status_code}",
                        {"status_code": response.status_code}
                    )
                    return False
        except Exception as e:
            self.log_test_result(
                f"Health Check - {name}",
                "FAIL",
                f"Connection failed: {str(e)}",
                {"error": str(e)}
            )
            return False
    
    async def test_ai_ds_team_agents(self) -> int:
        """AI DS Team 에이전트 상태 테스트"""
        healthy_count = 0
        
        for agent_name, port in self.ai_ds_agents.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_info = response.json()
                        self.log_test_result(
                            f"AI DS Agent - {agent_name}",
                            "PASS",
                            f"Port {port} - {agent_info.get('name', 'Unknown')}",
                            {"port": port, "agent_info": agent_info}
                        )
                        healthy_count += 1
                    else:
                        self.log_test_result(
                            f"AI DS Agent - {agent_name}",
                            "FAIL",
                            f"Port {port} - HTTP {response.status_code}",
                            {"port": port, "status_code": response.status_code}
                        )
            except Exception as e:
                self.log_test_result(
                    f"AI DS Agent - {agent_name}",
                    "FAIL",
                    f"Port {port} - Connection failed: {str(e)}",
                    {"port": port, "error": str(e)}
                )
        
        return healthy_count
    
    async def test_orchestrator_functionality(self) -> bool:
        """v9 Orchestrator 기능 테스트"""
        try:
            # Test agent discovery
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test basic functionality
                test_message = {
                    "parts": [
                        {
                            "type": "text",
                            "text": "Test system validation query"
                        }
                    ]
                }
                
                response = await client.post(
                    f"{self.endpoints['v9_orchestrator']}/task",
                    json=test_message,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    self.log_test_result(
                        "v9 Orchestrator - Functionality",
                        "PASS",
                        "Successfully processed test request",
                        {"response_status": response.status_code}
                    )
                    return True
                else:
                    self.log_test_result(
                        "v9 Orchestrator - Functionality",
                        "FAIL",
                        f"HTTP {response.status_code}",
                        {"status_code": response.status_code}
                    )
                    return False
                    
        except Exception as e:
            self.log_test_result(
                "v9 Orchestrator - Functionality",
                "FAIL",
                f"Test failed: {str(e)}",
                {"error": str(e)}
            )
            return False
    
    async def test_python_repl_agent_functionality(self) -> bool:
        """Python REPL Agent 기능 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                test_code = "print('Hello CherryAI v9!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')"
                
                test_message = {
                    "parts": [
                        {
                            "type": "text",
                            "text": f"Execute this Python code: {test_code}"
                        }
                    ]
                }
                
                response = await client.post(
                    f"{self.endpoints['python_repl_agent']}/task",
                    json=test_message,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    self.log_test_result(
                        "Python REPL Agent - Functionality",
                        "PASS",
                        "Successfully executed Python code",
                        {"response_status": response.status_code}
                    )
                    return True
                else:
                    self.log_test_result(
                        "Python REPL Agent - Functionality",
                        "FAIL",
                        f"HTTP {response.status_code}",
                        {"status_code": response.status_code}
                    )
                    return False
                    
        except Exception as e:
            self.log_test_result(
                "Python REPL Agent - Functionality",
                "FAIL",
                f"Test failed: {str(e)}",
                {"error": str(e)}
            )
            return False
    
    async def test_streamlit_ui_accessibility(self) -> bool:
        """Streamlit UI 접근성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.endpoints['streamlit_ui'])
                if response.status_code == 200:
                    self.log_test_result(
                        "Streamlit UI - Accessibility",
                        "PASS",
                        "UI is accessible and responding",
                        {"response_status": response.status_code}
                    )
                    return True
                else:
                    self.log_test_result(
                        "Streamlit UI - Accessibility",
                        "FAIL",
                        f"HTTP {response.status_code}",
                        {"status_code": response.status_code}
                    )
                    return False
        except Exception as e:
            self.log_test_result(
                "Streamlit UI - Accessibility",
                "FAIL",
                f"Connection failed: {str(e)}",
                {"error": str(e)}
            )
            return False
    
    async def test_langfuse_integration(self) -> bool:
        """Langfuse 통합 테스트"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.endpoints['langfuse_dashboard'])
                if response.status_code == 200:
                    self.log_test_result(
                        "Langfuse Integration",
                        "PASS",
                        "Langfuse dashboard is accessible",
                        {"response_status": response.status_code}
                    )
                    return True
                else:
                    self.log_test_result(
                        "Langfuse Integration",
                        "FAIL",
                        f"HTTP {response.status_code}",
                        {"status_code": response.status_code}
                    )
                    return False
        except Exception as e:
            self.log_test_result(
                "Langfuse Integration",
                "FAIL",
                f"Connection failed: {str(e)}",
                {"error": str(e)}
            )
            return False
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """종합적인 시스템 검증 실행"""
        print("🚀 CherryAI v9 System Validation Started")
        print("=" * 60)
        
        # Phase 1: Component Health Checks
        print("\n📋 Phase 1: Component Health Checks")
        print("-" * 40)
        
        v9_health = await self.test_component_health("v9_orchestrator", self.endpoints['v9_orchestrator'])
        repl_health = await self.test_component_health("python_repl_agent", self.endpoints['python_repl_agent'])
        
        # Phase 2: AI DS Team Agents
        print("\n🤖 Phase 2: AI DS Team Agents")
        print("-" * 40)
        
        healthy_agents = await self.test_ai_ds_team_agents()
        
        # Phase 3: Functionality Tests
        print("\n⚙️ Phase 3: Functionality Tests")
        print("-" * 40)
        
        orchestrator_func = await self.test_orchestrator_functionality()
        repl_func = await self.test_python_repl_agent_functionality()
        
        # Phase 4: UI and Integration Tests
        print("\n🎨 Phase 4: UI and Integration Tests")
        print("-" * 40)
        
        ui_accessible = await self.test_streamlit_ui_accessibility()
        langfuse_integrated = await self.test_langfuse_integration()
        
        # Generate Summary Report
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        total_tests = len(self.test_results)
        
        summary = {
            "validation_time": self.start_time.isoformat(),
            "duration_seconds": duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "core_components": {
                "v9_orchestrator": v9_health,
                "python_repl_agent": repl_health,
                "streamlit_ui": ui_accessible,
                "langfuse_integration": langfuse_integrated
            },
            "ai_ds_team_agents": {
                "healthy_count": healthy_agents,
                "total_count": len(self.ai_ds_agents),
                "health_rate": (healthy_agents / len(self.ai_ds_agents)) * 100
            },
            "functionality_tests": {
                "orchestrator": orchestrator_func,
                "python_repl": repl_func
            },
            "detailed_results": self.test_results
        }
        
        # Print Summary
        print("\n📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"🕒 Duration: {duration:.2f} seconds")
        print(f"📝 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"🤖 AI DS Agents: {healthy_agents}/{len(self.ai_ds_agents)} ({summary['ai_ds_team_agents']['health_rate']:.1f}%)")
        
        # Overall Status
        if summary['success_rate'] >= 90:
            print("\n🎉 SYSTEM STATUS: EXCELLENT")
        elif summary['success_rate'] >= 75:
            print("\n✅ SYSTEM STATUS: GOOD")
        elif summary['success_rate'] >= 50:
            print("\n⚠️ SYSTEM STATUS: NEEDS ATTENTION")
        else:
            print("\n❌ SYSTEM STATUS: CRITICAL ISSUES")
        
        return summary

async def main():
    """메인 검증 실행"""
    validator = CherryAI_v9_SystemValidator()
    
    try:
        summary = await validator.run_comprehensive_validation()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"logs/system_validation_{timestamp}.json"
        
        os.makedirs("logs", exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(main()) 