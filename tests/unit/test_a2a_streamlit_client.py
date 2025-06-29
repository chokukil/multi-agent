"""
A2A Streamlit Client 단위 테스트
"""
import pytest
import json
from unittest.mock import Mock, patch
from core.a2a.a2a_streamlit_client import A2AStreamlitClient


class TestA2AStreamlitClient:
    """A2A Streamlit Client 테스트 클래스"""
    
    @pytest.fixture
    def mock_agents_info(self):
        """테스트용 에이전트 정보"""
        return {
            "📁 Data Loader": {"url": "http://localhost:8307", "status": "active"},
            "🧹 Data Cleaning": {"url": "http://localhost:8306", "status": "active"},
            "🔍 EDA Tools": {"url": "http://localhost:8312", "status": "active"},
            "📊 Data Visualization": {"url": "http://localhost:8308", "status": "active"},
            "🔧 Data Wrangling": {"url": "http://localhost:8309", "status": "active"},
            "⚙️ Feature Engineering": {"url": "http://localhost:8310", "status": "active"},
            "🗄️ SQL Database": {"url": "http://localhost:8311", "status": "active"},
            "🤖 H2O ML": {"url": "http://localhost:8313", "status": "active"},
            "📈 MLflow Tools": {"url": "http://localhost:8314", "status": "active"},
            "Orchestrator": {"url": "http://localhost:8100", "status": "active"}
        }
    
    @pytest.fixture
    def client(self, mock_agents_info):
        """테스트용 A2A 클라이언트"""
        return A2AStreamlitClient(mock_agents_info)
    
    def test_parse_a2a_standard_response_with_history(self, client):
        """A2A 표준 응답 구조 (history) 파싱 테스트"""
        # Given: A2A 표준 응답 구조
        orchestrator_response = {
            "jsonrpc": "2.0",
            "id": "test_request",
            "result": {
                "contextId": "test-context",
                "history": [
                    {
                        "role": "agent",
                        "message": {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": json.dumps({
                                        "steps": [
                                            {
                                                "step_number": 1,
                                                "agent_name": "data_loader",
                                                "task_description": "Load and analyze data",
                                                "reasoning": "Data loading required"
                                            },
                                            {
                                                "step_number": 2,
                                                "agent_name": "eda_tools",
                                                "task_description": "Perform EDA analysis",
                                                "reasoning": "Exploratory analysis needed"
                                            }
                                        ]
                                    })
                                }
                            ]
                        }
                    }
                ],
                "status": {
                    "state": "completed",
                    "message": {}
                }
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 올바른 파싱 결과 확인
        assert len(result) == 2
        assert result[0]["step_number"] == 1
        assert result[0]["agent_name"] == "📁 Data Loader"
        assert result[0]["task_description"] == "Load and analyze data"
        assert result[1]["step_number"] == 2
        assert result[1]["agent_name"] == "🔍 EDA Tools"
        assert result[1]["task_description"] == "Perform EDA analysis"
    
    def test_parse_a2a_standard_response_with_status(self, client):
        """A2A 표준 응답 구조 (status) 파싱 테스트"""
        # Given: status에 계획이 포함된 A2A 응답
        orchestrator_response = {
            "jsonrpc": "2.0",
            "id": "test_request",
            "result": {
                "contextId": "test-context",
                "history": [],
                "status": {
                    "state": "completed",
                    "message": {
                        "parts": [
                            {
                                "kind": "text",
                                "text": json.dumps({
                                    "plan_executed": [
                                        {
                                            "step": 1,
                                            "agent": "DataVisualizationAgent",
                                            "description": "Create visualizations"
                                        }
                                    ]
                                })
                            }
                        ]
                    }
                }
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 올바른 파싱 결과 확인
        assert len(result) == 1
        assert result[0]["step_number"] == 1
        assert result[0]["agent_name"] == "📊 Data Visualization"
        assert result[0]["task_description"] == "Create visualizations"
    
    def test_parse_direct_message_response(self, client):
        """직접 메시지 응답 파싱 테스트"""
        # Given: 직접 메시지 응답 구조
        orchestrator_response = {
            "result": {
                "message": {
                    "parts": [
                        {
                            "kind": "text",
                            "text": json.dumps({
                                "steps": [
                                    {
                                        "step_number": 1,
                                        "agent_name": "h2o_ml",
                                        "task_description": "Train ML model"
                                    }
                                ]
                            })
                        }
                    ]
                }
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 올바른 파싱 결과 확인
        assert len(result) == 1
        assert result[0]["agent_name"] == "🤖 H2O ML"
        assert result[0]["task_description"] == "Train ML model"
    
    def test_parse_direct_response_steps_format(self, client):
        """직접 응답 (steps 형식) 파싱 테스트"""
        # Given: 직접 steps 형식 응답
        orchestrator_response = {
            "result": {
                "steps": [
                    {
                        "step_number": 1,
                        "agent_name": "feature_engineering",
                        "task_description": "Engineer features"
                    },
                    {
                        "step": 2,
                        "agent": "mlflow_tools",
                        "description": "Track experiments"
                    }
                ]
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 올바른 파싱 결과 확인
        assert len(result) == 2
        assert result[0]["agent_name"] == "⚙️ Feature Engineering"
        assert result[1]["agent_name"] == "📈 MLflow Tools"
    
    def test_parse_direct_response_plan_executed_format(self, client):
        """직접 응답 (plan_executed 형식) 파싱 테스트"""
        # Given: plan_executed 형식 응답
        orchestrator_response = {
            "result": {
                "plan_executed": [
                    {
                        "step": 1,
                        "agent": "sql_database",
                        "task": "Query database"
                    }
                ]
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 올바른 파싱 결과 확인
        assert len(result) == 1
        assert result[0]["agent_name"] == "🗄️ SQL Database"
        assert result[0]["task_description"] == "Query database"
    
    def test_extract_plan_from_json_block(self, client):
        """JSON 블록에서 계획 추출 테스트"""
        # Given: JSON 블록이 포함된 텍스트
        text_with_json_block = """
        Here is the execution plan:
        
        ```json
        {
            "steps": [
                {
                    "step_number": 1,
                    "agent_name": "data_cleaning",
                    "task_description": "Clean data"
                }
            ]
        }
        ```
        
        This plan will be executed.
        """
        
        # When: JSON 블록 추출
        result = client._extract_plan_from_text(text_with_json_block)
        
        # Then: 올바른 추출 결과 확인
        assert len(result) == 1
        assert result[0]["agent_name"] == "🧹 Data Cleaning"
        assert result[0]["task_description"] == "Clean data"
    
    def test_agent_name_mapping(self, client):
        """에이전트 이름 매핑 테스트"""
        # Given: 다양한 에이전트 이름 형식
        test_cases = [
            ("data_loader", "📁 Data Loader"),
            ("AI_DS_Team DataLoaderToolsAgent", "📁 Data Loader"),
            ("DataLoaderToolsAgent", "📁 Data Loader"),
            ("eda_tools", "🔍 EDA Tools"),
            ("AI_DS_Team EDAToolsAgent", "🔍 EDA Tools"),
            ("unknown_agent", "unknown_agent")  # 매핑되지 않은 경우
        ]
        
        agent_mapping = client._get_agent_mapping()
        
        # When & Then: 각 매핑 테스트
        for input_name, expected_output in test_cases:
            actual_output = agent_mapping.get(input_name, input_name)
            assert actual_output == expected_output, f"Failed for {input_name}"
    
    def test_agent_fallback_mechanism(self, client):
        """사용 불가능한 에이전트에 대한 폴백 메커니즘 테스트"""
        # Given: 사용 불가능한 에이전트가 포함된 계획
        steps = [
            {
                "step_number": 1,
                "agent_name": "nonexistent_agent",
                "task_description": "Some task"
            }
        ]
        
        # When: 단계 처리
        result = client._process_steps(steps)
        
        # Then: 폴백 에이전트 사용 확인
        assert len(result) == 1
        assert result[0]["agent_name"] in client._agents_info
        assert result[0]["agent_name"] != "Orchestrator"
        assert "원래: nonexistent_agent" in result[0]["task_description"]
    
    def test_empty_response_handling(self, client):
        """빈 응답 처리 테스트"""
        # Given: 빈 응답들
        empty_responses = [
            {},
            {"result": {}},
            {"result": {"history": [], "status": {}}},
            {"result": {"steps": []}},
            None
        ]
        
        # When & Then: 각 빈 응답에 대해 빈 리스트 반환 확인
        for empty_response in empty_responses:
            if empty_response is None:
                continue
            result = client.parse_orchestration_plan(empty_response)
            assert result == [], f"Failed for {empty_response}"
    
    def test_malformed_json_handling(self, client):
        """잘못된 JSON 처리 테스트"""
        # Given: 잘못된 JSON이 포함된 응답
        orchestrator_response = {
            "result": {
                "history": [
                    {
                        "role": "agent",
                        "message": {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": "{ invalid json content"
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 빈 결과 반환 확인
        assert result == []
    
    def test_non_dict_response_handling(self, client):
        """딕셔너리가 아닌 응답 처리 테스트"""
        # Given: 딕셔너리가 아닌 응답들
        non_dict_responses = [
            "string response",
            123,
            [],
            None
        ]
        
        # When & Then: 각 응답에 대해 빈 리스트 반환 확인
        for response in non_dict_responses:
            result = client.parse_orchestration_plan(response)
            assert result == []
    
    def test_step_standardization(self, client):
        """단계 표준화 테스트"""
        # Given: 다양한 형식의 단계들
        steps = [
            {
                "step_number": 1,
                "agent_name": "data_loader",
                "task_description": "Load data",
                "reasoning": "Need to load data first"
            },
            {
                "step": 2,
                "agent": "eda_tools",
                "description": "Analyze data",
                "priority": "high"
            },
            {
                "agent_name": "data_visualization",
                "task": "Create charts"
            }
        ]
        
        # When: 단계 처리
        result = client._process_steps(steps)
        
        # Then: 표준화된 형식 확인
        assert len(result) == 3
        
        # 첫 번째 단계
        assert result[0]["step_number"] == 1
        assert result[0]["agent_name"] == "📁 Data Loader"
        assert result[0]["task_description"] == "Load data"
        assert result[0]["reasoning"] == "Need to load data first"
        
        # 두 번째 단계 (다른 필드명 사용)
        assert result[1]["step_number"] == 2
        assert result[1]["agent_name"] == "🔍 EDA Tools"
        assert result[1]["task_description"] == "Analyze data"
        assert result[1]["parameters"]["priority"] == "high"
        
        # 세 번째 단계 (기본값 사용)
        assert result[2]["step_number"] == 3  # 자동 할당
        assert result[2]["agent_name"] == "📊 Data Visualization"
        assert result[2]["task_description"] == "Create charts"

    @patch('core.a2a.a2a_streamlit_client.A2AStreamlitClient._debug_log')
    def test_debug_logging(self, mock_debug_log, client):
        """디버그 로깅 테스트"""
        # Given: 정상적인 응답
        orchestrator_response = {
            "result": {
                "steps": [
                    {
                        "step_number": 1,
                        "agent_name": "data_loader",
                        "task_description": "Load data"
                    }
                ]
            }
        }
        
        # When: 파싱 실행
        client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 디버그 로그 호출 확인
        mock_debug_log.assert_called()
        
        # 특정 로그 메시지 확인
        log_calls = [call.args[0] for call in mock_debug_log.call_args_list]
        assert any("A2A 표준 기반 계획 파싱 시작" in call for call in log_calls)
        assert any("직접 응답 구조 파싱 중" in call for call in log_calls) 