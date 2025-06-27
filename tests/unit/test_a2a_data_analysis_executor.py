"""
A2A Data Analysis Executor 단위 테스트
"""

import pytest
import asyncio
import httpx
from unittest.mock import Mock, patch, AsyncMock
import uuid
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.a2a_data_analysis_executor import A2ADataAnalysisExecutor


class TestA2ADataAnalysisExecutor:
    """A2A Data Analysis Executor 테스트 클래스"""

    @pytest.fixture
    def executor(self):
        """테스트용 executor 인스턴스"""
        return A2ADataAnalysisExecutor()

    @pytest.fixture
    def sample_plan_state(self):
        """테스트용 계획 상태"""
        return {
            "plan": [
                {
                    "agent_name": "pandas_data_analyst",
                    "skill_name": "analyze_data",
                    "parameters": {
                        "data_id": "test_dataset",
                        "user_instructions": "Analyze the data structure"
                    }
                },
                {
                    "agent_name": "data_visualization", 
                    "skill_name": "analyze_data",
                    "parameters": {
                        "data_id": "test_dataset",
                        "user_instructions": "Create visualizations"
                    }
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_executor_initialization(self, executor):
        """실행기 초기화 테스트"""
        assert executor.timeout == 300
        assert len(executor.agent_configs) == 6
        assert "pandas_data_analyst" in executor.agent_configs
        assert "data_visualization" in executor.agent_configs

    @pytest.mark.asyncio
    async def test_successful_plan_execution(self, executor, sample_plan_state):
        """성공적인 계획 실행 테스트"""
        
        # Mock 응답 설정
        mock_responses = [
            {
                "result": {
                    "status": "completed",
                    "output": "Data analysis completed successfully"
                }
            },
            {
                "result": {
                    "status": "completed", 
                    "output": "Visualizations created successfully"
                }
            }
        ]

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            
            # 각 호출에 대해 다른 응답 반환
            mock_response.json.side_effect = mock_responses
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # progress_stream_manager 모킹
            with patch('core.a2a_data_analysis_executor.progress_stream_manager.stream_update') as mock_stream:
                mock_stream.return_value = None
                
                result = await executor.execute(sample_plan_state)

                # 결과 검증
                assert result["total_steps"] == 2
                assert result["successful_steps"] == 2
                assert "execution_time" in result
                assert len(result["step_outputs"]) == 2

    @pytest.mark.asyncio
    async def test_agent_call_success(self, executor):
        """성공적인 에이전트 호출 테스트"""
        
        agent_config = {"url": "http://localhost:8200"}
        step = {
            "agent_name": "pandas_data_analyst",
            "skill_name": "analyze_data",
            "parameters": {
                "data_id": "test_dataset",
                "user_instructions": "Test analysis"
            }
        }

        mock_response_data = {
            "result": {
                "status": "completed",
                "output": "Analysis completed"
            }
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await executor.call_agent(agent_config, step, 1)

            # 결과 검증
            assert result["success"] is True
            assert result["content"] == mock_response_data["result"]
            assert result["agent"] == "pandas_data_analyst"
            assert result["step"] == 1

    @pytest.mark.asyncio
    async def test_agent_call_timeout(self, executor):
        """에이전트 호출 타임아웃 테스트"""
        
        agent_config = {"url": "http://localhost:8200"}
        step = {
            "agent_name": "pandas_data_analyst",
            "parameters": {
                "data_id": "test_dataset",
                "user_instructions": "Test analysis"
            }
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            result = await executor.call_agent(agent_config, step, 1)

            # 결과 검증
            assert result["success"] is False
            assert "Timeout" in result["error"]
            assert result["agent"] == "pandas_data_analyst"

    @pytest.mark.asyncio
    async def test_agent_call_connection_error(self, executor):
        """에이전트 호출 연결 오류 테스트"""
        
        agent_config = {"url": "http://localhost:8200"}
        step = {
            "agent_name": "pandas_data_analyst",
            "parameters": {
                "data_id": "test_dataset",
                "user_instructions": "Test analysis"
            }
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            result = await executor.call_agent(agent_config, step, 1)

            # 결과 검증
            assert result["success"] is False
            assert "Connection error" in result["error"]

    @pytest.mark.asyncio
    async def test_agent_call_rpc_error(self, executor):
        """에이전트 호출 RPC 오류 테스트"""
        
        agent_config = {"url": "http://localhost:8200"}
        step = {
            "agent_name": "pandas_data_analyst",
            "parameters": {
                "data_id": "test_dataset",
                "user_instructions": "Test analysis"
            }
        }

        mock_response_data = {
            "error": {
                "code": -32603,
                "message": "Internal error"
            }
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await executor.call_agent(agent_config, step, 1)

            # 결과 검증
            assert result["success"] is False
            assert result["error"] == "Internal error"

    @pytest.mark.asyncio
    async def test_check_agent_health_success(self, executor):
        """에이전트 상태 확인 성공 테스트"""
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await executor.check_agent_health("http://localhost:8200")
            assert result is True

    @pytest.mark.asyncio
    async def test_check_agent_health_failure(self, executor):
        """에이전트 상태 확인 실패 테스트"""
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            result = await executor.check_agent_health("http://localhost:8200")
            assert result is False

    @pytest.mark.asyncio
    async def test_get_available_agents(self, executor):
        """사용 가능한 에이전트 목록 조회 테스트"""
        
        # 일부 에이전트만 사용 가능하도록 설정
        with patch.object(executor, 'check_agent_health') as mock_health:
            mock_health.side_effect = lambda url: url in [
                "http://localhost:8200",  # pandas_data_analyst
                "http://localhost:8202"   # data_visualization
            ]

            available_agents = await executor.get_available_agents()
            
            assert "pandas_data_analyst" in available_agents
            assert "data_visualization" in available_agents
            assert len(available_agents) == 2

    @pytest.mark.asyncio
    async def test_unknown_agent_handling(self, executor):
        """알 수 없는 에이전트 처리 테스트"""
        
        plan_state = {
            "plan": [
                {
                    "agent_name": "unknown_agent",
                    "parameters": {
                        "data_id": "test_dataset",
                        "user_instructions": "Test"
                    }
                }
            ]
        }

        with patch('core.a2a_data_analysis_executor.progress_stream_manager.stream_update') as mock_stream:
            mock_stream.return_value = None
            
            result = await executor.execute(plan_state)

            # 결과 검증
            assert result["total_steps"] == 1
            assert result["successful_steps"] == 0
            assert len(result["step_outputs"]) == 0

    @pytest.mark.asyncio
    async def test_emit_error(self, executor):
        """에러 발생 테스트"""
        
        with patch('core.a2a_data_analysis_executor.progress_stream_manager.stream_update') as mock_stream:
            mock_stream.return_value = None
            
            await executor.emit_error(1, "test_agent", "Test error message")
            
            # stream_update가 호출되었는지 확인
            mock_stream.assert_called_once()
            call_args = mock_stream.call_args[0][0]
            assert call_args["event_type"] == "agent_error"
            assert call_args["data"]["step"] == 1
            assert call_args["data"]["agent_name"] == "test_agent"
            assert call_args["data"]["error"] == "Test error message"

if __name__ == "__main__":
    # 개별 테스트 실행을 위한 코드
    pytest.main([__file__, "-v"]) 