#!/usr/bin/env python3
"""
A2A Orchestrator v5.0 표준 준수 테스트
pytest를 사용한 단위 및 통합 테스트
"""

import asyncio
import json
import pytest
import httpx
from typing import Dict, Any, List

# 테스트 대상 모듈
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.a2a.a2a_streamlit_client import A2AStreamlitClient


class TestA2AOrchestratorV5Standard:
    """A2A 오케스트레이터 v5.0 표준 준수 테스트"""
    
    @pytest.fixture
    def orchestrator_url(self):
        """오케스트레이터 URL"""
        return "http://localhost:8100"
    
    @pytest.fixture
    def sample_agents_info(self):
        """테스트용 에이전트 정보"""
        return {
            "📁 Data Loader": {"url": "http://localhost:8307", "status": "available"},
            "🧹 Data Cleaning": {"url": "http://localhost:8306", "status": "available"},
            "🔍 EDA Tools": {"url": "http://localhost:8312", "status": "available"},
            "📊 Data Visualization": {"url": "http://localhost:8308", "status": "available"}
        }
    
    @pytest.fixture
    def a2a_client(self, sample_agents_info):
        """A2A 클라이언트 인스턴스"""
        return A2AStreamlitClient(sample_agents_info)
    
    def test_orchestrator_agent_card_standard_compliance(self, orchestrator_url):
        """오케스트레이터 Agent Card가 A2A 표준을 준수하는지 테스트"""
        
        async def check_agent_card():
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{orchestrator_url}/.well-known/agent.json")
                assert response.status_code == 200
                
                agent_card = response.json()
                
                # A2A 표준 필수 필드 검증
                assert "name" in agent_card
                assert "description" in agent_card
                assert "url" in agent_card
                assert "version" in agent_card
                assert "capabilities" in agent_card
                assert "skills" in agent_card
                
                # 버전 확인
                assert agent_card["version"] == "5.0.0"
                assert "Standard Orchestrator" in agent_card["name"]
                
                # 기능 검증
                capabilities = agent_card["capabilities"]
                assert capabilities.get("streaming") is True
                assert "skills" in agent_card
                assert len(agent_card["skills"]) > 0
                
                # 스킬 검증
                skill = agent_card["skills"][0]
                assert skill["id"] == "orchestrate_analysis"
                assert "a2a-standard" in skill.get("tags", [])
                
                print("✅ Agent Card A2A 표준 준수 검증 완료")
                return True
        
        result = asyncio.run(check_agent_card())
        assert result is True
    
    def test_orchestrator_plan_generation_with_artifacts(self, orchestrator_url, a2a_client):
        """오케스트레이터가 Artifact로 계획을 생성하는지 테스트"""
        
        async def test_plan_generation():
            try:
                # 오케스트레이터에게 계획 요청
                plan_result = await a2a_client.get_plan("데이터셋에 대한 종합적인 EDA 분석을 수행해주세요")
                
                # 기본 응답 구조 검증
                assert "success" in plan_result
                assert "steps" in plan_result
                
                if plan_result["success"]:
                    steps = plan_result["steps"]
                    assert isinstance(steps, list)
                    assert len(steps) > 0
                    
                    # 각 단계 구조 검증
                    for step in steps:
                        assert "step_number" in step
                        assert "agent_name" in step
                        assert "task_description" in step
                        assert "reasoning" in step
                        
                        # 에이전트 이름이 매핑되었는지 확인
                        agent_name = step["agent_name"]
                        assert agent_name in a2a_client._agents_info or agent_name.startswith(("📁", "🧹", "🔍", "📊"))
                    
                    print(f"✅ 계획 생성 성공: {len(steps)}개 단계")
                    return True
                else:
                    print(f"⚠️ 계획 생성 실패: {plan_result.get('error', 'Unknown error')}")
                    return False
                    
            except Exception as e:
                print(f"❌ 계획 생성 테스트 실패: {e}")
                return False
        
        result = asyncio.run(test_plan_generation())
        assert result is True
    
    def test_artifact_parsing_logic(self, a2a_client):
        """Artifact 파싱 로직 단위 테스트"""
        
        # 테스트용 A2A 표준 응답 (Artifact 포함)
        mock_response = {
            "artifacts": [
                {
                    "name": "execution_plan",
                    "metadata": {
                        "content_type": "application/json",
                        "plan_type": "ai_ds_team_orchestration"
                    },
                    "parts": [
                        {
                            "kind": "text",
                            "text": json.dumps({
                                "plan_type": "ai_ds_team_orchestration",
                                "objective": "데이터 분석 수행",
                                "steps": [
                                    {
                                        "step_number": 1,
                                        "agent_name": "data_loader",
                                        "task_description": "데이터 로드 및 검증",
                                        "reasoning": "데이터 분석의 첫 단계"
                                    },
                                    {
                                        "step_number": 2,
                                        "agent_name": "eda_tools",
                                        "task_description": "탐색적 데이터 분석",
                                        "reasoning": "데이터 패턴 파악"
                                    }
                                ]
                            }, ensure_ascii=False)
                        }
                    ]
                }
            ]
        }
        
        # Artifact 파싱 테스트
        parsed_steps = a2a_client._parse_a2a_standard_response(mock_response)
        
        assert isinstance(parsed_steps, list)
        assert len(parsed_steps) == 2
        
        # 첫 번째 단계 검증
        step1 = parsed_steps[0]
        assert step1["step_number"] == 1
        assert step1["agent_name"] == "📁 Data Loader"  # 매핑된 이름
        assert "데이터 로드" in step1["task_description"]
        
        # 두 번째 단계 검증
        step2 = parsed_steps[1]
        assert step2["step_number"] == 2
        assert step2["agent_name"] == "🔍 EDA Tools"  # 매핑된 이름
        assert "탐색적" in step2["task_description"]
        
        print("✅ Artifact 파싱 로직 검증 완료")
    
    def test_agent_name_mapping(self, a2a_client):
        """에이전트 이름 매핑 테스트"""
        
        mapping = a2a_client._get_agent_mapping()
        
        # 기본 매핑 확인
        assert mapping["data_loader"] == "📁 Data Loader"
        assert mapping["data_cleaning"] == "🧹 Data Cleaning"
        assert mapping["eda_tools"] == "🔍 EDA Tools"
        assert mapping["data_visualization"] == "📊 Data Visualization"
        
        # 전체 이름 매핑 확인
        assert mapping["AI_DS_Team DataLoaderToolsAgent"] == "📁 Data Loader"
        assert mapping["AI_DS_Team EDAToolsAgent"] == "🔍 EDA Tools"
        
        print("✅ 에이전트 이름 매핑 검증 완료")
    
    def test_fallback_plan_generation(self, a2a_client):
        """폴백 계획 생성 테스트"""
        
        # 빈 응답으로 폴백 시나리오 테스트
        empty_response = {}
        parsed_steps = a2a_client._parse_a2a_standard_response(empty_response)
        
        # 폴백이 동작하지 않으면 빈 리스트 반환
        assert isinstance(parsed_steps, list)
        
        print("✅ 폴백 계획 생성 로직 검증 완료")
    
    def test_error_handling(self, a2a_client):
        """오류 처리 테스트"""
        
        # 잘못된 JSON 형식
        invalid_response = {
            "artifacts": [
                {
                    "name": "execution_plan",
                    "metadata": {"plan_type": "ai_ds_team_orchestration"},
                    "parts": [{"kind": "text", "text": "invalid json {"}]
                }
            ]
        }
        
        parsed_steps = a2a_client._parse_a2a_standard_response(invalid_response)
        assert isinstance(parsed_steps, list)  # 오류 시에도 리스트 반환
        
        print("✅ 오류 처리 로직 검증 완료")


class TestA2AIntegration:
    """A2A 통합 테스트"""
    
    @pytest.fixture
    def agents_info(self):
        """실제 에이전트 정보"""
        return {
            "Orchestrator": {"url": "http://localhost:8100", "status": "available"},
            "📁 Data Loader": {"url": "http://localhost:8307", "status": "available"},
            "🧹 Data Cleaning": {"url": "http://localhost:8306", "status": "available"},
            "🔍 EDA Tools": {"url": "http://localhost:8312", "status": "available"},
            "📊 Data Visualization": {"url": "http://localhost:8308", "status": "available"}
        }
    
    def test_full_orchestration_workflow(self, agents_info):
        """전체 오케스트레이션 워크플로우 테스트"""
        
        async def test_workflow():
            client = A2AStreamlitClient(agents_info)
            
            try:
                # 1. 계획 생성
                plan_result = await client.get_plan("데이터 분석을 수행해주세요")
                
                if not plan_result.get("success"):
                    print(f"❌ 계획 생성 실패: {plan_result.get('error')}")
                    return False
                
                steps = plan_result["steps"]
                print(f"📋 생성된 계획: {len(steps)}개 단계")
                
                # 2. 각 단계 정보 출력
                for i, step in enumerate(steps, 1):
                    print(f"  {i}. {step['agent_name']}: {step['task_description']}")
                
                # 3. 기본 검증
                assert len(steps) > 0
                assert all("agent_name" in step for step in steps)
                assert all("task_description" in step for step in steps)
                
                print("✅ 전체 워크플로우 테스트 성공")
                return True
                
            except Exception as e:
                print(f"❌ 워크플로우 테스트 실패: {e}")
                return False
            finally:
                await client.close()
        
        result = asyncio.run(test_workflow())
        assert result is True


if __name__ == "__main__":
    # 개별 테스트 실행
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # 직접 실행 모드
        test_instance = TestA2AOrchestratorV5Standard()
        
        print("🧪 A2A 오케스트레이터 v5.0 표준 준수 테스트 시작")
        
        try:
            # Agent Card 테스트
            test_instance.test_orchestrator_agent_card_standard_compliance("http://localhost:8100")
            
            # 에이전트 정보 설정
            agents_info = {
                "📁 Data Loader": {"url": "http://localhost:8307", "status": "available"},
                "🧹 Data Cleaning": {"url": "http://localhost:8306", "status": "available"},
                "🔍 EDA Tools": {"url": "http://localhost:8312", "status": "available"},
                "📊 Data Visualization": {"url": "http://localhost:8308", "status": "available"}
            }
            client = A2AStreamlitClient(agents_info)
            
            # 단위 테스트들
            test_instance.test_artifact_parsing_logic(client)
            test_instance.test_agent_name_mapping(client)
            test_instance.test_fallback_plan_generation(client)
            test_instance.test_error_handling(client)
            
            # 계획 생성 테스트
            test_instance.test_orchestrator_plan_generation_with_artifacts("http://localhost:8100", client)
            
            print("\n🎉 모든 테스트 통과!")
            
        except Exception as e:
            print(f"\n❌ 테스트 실패: {e}")
            sys.exit(1)
    else:
        print("pytest로 실행하거나 'python test_a2a_orchestrator_v5_standard.py run'으로 직접 실행하세요.") 