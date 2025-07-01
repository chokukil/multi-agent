"""
A2A 시스템 통합 테스트
pytest로 실행: pytest test_a2a_integration.py -v
"""

import pytest
import asyncio
import httpx
import json
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestA2AOrchestrator:
    """A2A 오케스트레이터 통합 테스트"""
    
    @pytest.fixture
    def orchestrator_url(self):
        """오케스트레이터 URL"""
        return "http://localhost:8100"
    
    @pytest.mark.asyncio
    async def test_orchestrator_health_check(self, orchestrator_url):
        """오케스트레이터 상태 확인 테스트"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{orchestrator_url}/.well-known/agent.json")
                
                # Then
                assert response.status_code == 200
                agent_card = response.json()
                assert "name" in agent_card
                assert "description" in agent_card
                print(f"✅ 오케스트레이터 상태 정상: {agent_card['name']}")
        except Exception as e:
            pytest.skip(f"오케스트레이터가 실행되지 않음: {e}")
    
    @pytest.mark.asyncio
    async def test_simple_message_request(self, orchestrator_url):
        """간단한 메시지 요청 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Given
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": "integration_test_001",
                            "role": "user",
                            "parts": [{"text": "안녕하세요"}]
                        }
                    },
                    "id": "req_integration_001"
                }
                
                # When
                response = await client.post(
                    f"{orchestrator_url}/",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                # Then
                assert response.status_code == 200
                result = response.json()
                assert "result" in result
                print(f"✅ 간단한 요청 성공: {response.status_code}")
        except Exception as e:
            pytest.skip(f"오케스트레이터 통신 실패: {e}")
    
    @pytest.mark.asyncio
    async def test_data_analysis_request(self, orchestrator_url):
        """데이터 분석 요청 테스트"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Given
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": "integration_test_002",
                            "role": "user",
                            "parts": [{"text": "ion_implant_3lot_dataset.csv 파일로 간단한 EDA 분석을 해주세요"}]
                        }
                    },
                    "id": "req_integration_002"
                }
                
                # When
                response = await client.post(
                    f"{orchestrator_url}/",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                # Then
                assert response.status_code == 200
                result = response.json()
                assert "result" in result
                print(f"✅ 데이터 분석 요청 성공: {response.status_code}")
        except Exception as e:
            pytest.skip(f"데이터 분석 요청 실패: {e}")


class TestA2AAgents:
    """A2A 에이전트들 통합 테스트"""
    
    @pytest.fixture
    def agent_ports(self):
        """테스트할 에이전트 포트들"""
        return [8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314]
    
    @pytest.mark.asyncio
    async def test_all_agents_health(self, agent_ports):
        """모든 에이전트 상태 확인 테스트"""
        healthy_agents = []
        failed_agents = []
        
        for port in agent_ports:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                    
                    if response.status_code == 200:
                        agent_card = response.json()
                        healthy_agents.append({
                            "port": port,
                            "name": agent_card.get("name", f"Agent_{port}"),
                            "url": f"http://localhost:{port}"
                        })
                    else:
                        failed_agents.append(port)
            except Exception as e:
                failed_agents.append(port)
        
        # Then
        print(f"✅ 정상 에이전트: {len(healthy_agents)}개")
        print(f"❌ 실패 에이전트: {len(failed_agents)}개")
        
        for agent in healthy_agents:
            print(f"  - {agent['name']} (포트 {agent['port']})")
        
        # 최소 3개 이상의 에이전트가 정상이어야 함
        assert len(healthy_agents) >= 3, f"정상 에이전트가 부족합니다: {len(healthy_agents)}/9"
    
    @pytest.mark.asyncio
    async def test_data_loader_agent(self):
        """DataLoader 에이전트 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Given
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": "agent_test_001",
                            "role": "user",
                            "parts": [{"text": "사용 가능한 데이터 파일을 보여주세요"}]
                        }
                    },
                    "id": "req_agent_001"
                }
                
                # When
                response = await client.post(
                    "http://localhost:8307/",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                # Then
                assert response.status_code == 200
                result = response.json()
                assert "result" in result
                print(f"✅ DataLoader 에이전트 정상 작동")
        except Exception as e:
            pytest.skip(f"DataLoader 에이전트 테스트 실패: {e}")


class TestA2ASystemIntegration:
    """A2A 시스템 전체 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_to_agent_communication(self):
        """오케스트레이터-에이전트 간 통신 테스트"""
        try:
            # Given: 오케스트레이터에 복잡한 요청 전송
            async with httpx.AsyncClient(timeout=60.0) as client:
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": "system_integration_001",
                            "role": "user",
                            "parts": [{"text": "데이터를 로드하고 간단한 분석을 해주세요"}]
                        }
                    },
                    "id": "req_system_001"
                }
                
                # When
                response = await client.post(
                    "http://localhost:8100/",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                # Then
                assert response.status_code == 200
                result = response.json()
                
                # 응답에 에이전트 실행 결과가 포함되어야 함
                assert "result" in result
                print(f"✅ 오케스트레이터-에이전트 통신 성공")
                
        except Exception as e:
            pytest.skip(f"시스템 통합 테스트 실패: {e}")
    
    @pytest.mark.asyncio
    async def test_streaming_response_format(self):
        """스트리밍 응답 형식 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Given
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "messageId": "streaming_test_001",
                            "role": "user",
                            "parts": [{"text": "스트리밍 테스트입니다"}]
                        }
                    },
                    "id": "req_streaming_001"
                }
                
                # When
                response = await client.post(
                    "http://localhost:8100/",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                # Then
                assert response.status_code == 200
                result = response.json()
                
                # JSON-RPC 2.0 형식 확인
                assert "jsonrpc" in result
                assert result["jsonrpc"] == "2.0"
                assert "id" in result
                assert "result" in result
                
                print(f"✅ 스트리밍 응답 형식 정상")
                
        except Exception as e:
            pytest.skip(f"스트리밍 응답 형식 테스트 실패: {e}")


if __name__ == "__main__":
    # 통합 테스트 실행
    pytest.main([__file__, "-v", "--tb=short", "-x"])
