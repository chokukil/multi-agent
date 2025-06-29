"""
실제 A2A 오케스트레이터 응답 시나리오 테스트
"""
import pytest
import json
from core.a2a.a2a_streamlit_client import A2AStreamlitClient


class TestA2ARealScenarios:
    """실제 A2A 시나리오 테스트 클래스"""
    
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
    
    def test_real_orchestrator_response_semiconductor(self, client):
        """실제 반도체 이온주입 공정 분석 응답 파싱 테스트"""
        # Given: 실제 오케스트레이터 응답 구조 (로그에서 확인된 형태)
        orchestrator_response = {
            "jsonrpc": "2.0",
            "id": "plan_request_1751179619",
            "result": {
                "contextId": "309f9d92-87f1-4f5b-a796-39629dd9b3d7",
                "history": [
                    {
                        "role": "user",
                        "message": {
                            "messageId": "msg_1751179619",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": "당신은 20년 경력의 반도체 이온주입 공정(Process) 엔지니어입니다..."
                                }
                            ]
                        }
                    },
                    {
                        "role": "agent", 
                        "message": {
                            "messageId": "response_1751179619",
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": json.dumps({
                                        "objective": "반도체 이온주입 공정 데이터 분석 및 이상 진단",
                                        "reasoning": "반도체 공정 데이터의 특성상 다단계 분석이 필요함",
                                        "steps": [
                                            {
                                                "step_number": 1,
                                                "agent_name": "AI_DS_Team DataLoaderToolsAgent",
                                                "task_description": "반도체 이온주입 공정 데이터 로드 및 기본 검증",
                                                "reasoning": "공정 데이터의 무결성 확인 필요"
                                            },
                                            {
                                                "step_number": 2,
                                                "agent_name": "AI_DS_Team EDAToolsAgent", 
                                                "task_description": "TW, RS 등 핵심 공정 지표 분석",
                                                "reasoning": "공정 이상 여부 판단을 위한 통계 분석"
                                            },
                                            {
                                                "step_number": 3,
                                                "agent_name": "AI_DS_Team DataVisualizationAgent",
                                                "task_description": "공정 트렌드 및 이상 패턴 시각화",
                                                "reasoning": "공정 엔지니어가 직관적으로 이해할 수 있는 차트 제공"
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
                    "message": {
                        "messageId": "status_1751179619",
                        "parts": [
                            {
                                "kind": "text",
                                "text": "반도체 이온주입 공정 분석 계획이 성공적으로 수립되었습니다."
                            }
                        ]
                    }
                },
                "kind": "task"
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 올바른 파싱 결과 확인
        assert len(result) == 3
        
        # 첫 번째 단계: Data Loader
        assert result[0]["step_number"] == 1
        assert result[0]["agent_name"] == "📁 Data Loader"
        assert "반도체 이온주입 공정 데이터 로드" in result[0]["task_description"]
        
        # 두 번째 단계: EDA Tools
        assert result[1]["step_number"] == 2
        assert result[1]["agent_name"] == "🔍 EDA Tools"
        assert "TW, RS 등 핵심 공정 지표" in result[1]["task_description"]
        
        # 세 번째 단계: Data Visualization
        assert result[2]["step_number"] == 3
        assert result[2]["agent_name"] == "📊 Data Visualization"
        assert "공정 트렌드 및 이상 패턴" in result[2]["task_description"]
    
    def test_orchestrator_response_with_markdown_json_block(self, client):
        """마크다운 JSON 블록이 포함된 오케스트레이터 응답 테스트"""
        # Given: 마크다운 형식의 JSON 블록이 포함된 응답
        orchestrator_response = {
            "result": {
                "history": [
                    {
                        "role": "agent",
                        "message": {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": """
# 🎯 Universal AI Orchestration Plan

**User Request**: 데이터 분석을 수행해주세요

## 🚀 Execution Plan

```json
{
    "objective": "포괄적인 데이터 분석 수행",
    "steps": [
        {
            "step_number": 1,
            "agent_name": "data_loader",
            "task_description": "데이터 로드 및 전처리",
            "reasoning": "분석을 위한 데이터 준비"
        },
        {
            "step_number": 2,
            "agent_name": "eda_tools",
            "task_description": "탐색적 데이터 분석 수행",
            "reasoning": "데이터의 패턴과 특성 파악"
        }
    ]
}
```

이 계획에 따라 분석을 진행합니다.
                                    """
                                }
                            ]
                        }
                    }
                ],
                "status": {"state": "completed"}
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 마크다운 블록에서 JSON 추출 확인
        assert len(result) == 2
        assert result[0]["agent_name"] == "📁 Data Loader"
        assert result[1]["agent_name"] == "🔍 EDA Tools"
    
    def test_orchestrator_response_with_plan_executed_format(self, client):
        """plan_executed 형식의 실제 응답 테스트"""
        # Given: plan_executed 형식의 응답
        orchestrator_response = {
            "result": {
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
                                            "agent": "DataLoaderToolsAgent",
                                            "description": "데이터 로딩 및 검증",
                                            "status": "completed",
                                            "execution_time": "2.3s"
                                        },
                                        {
                                            "step": 2,
                                            "agent": "EDAToolsAgent",
                                            "description": "통계 분석 수행",
                                            "status": "completed", 
                                            "execution_time": "5.7s"
                                        },
                                        {
                                            "step": 3,
                                            "agent": "DataVisualizationAgent",
                                            "description": "시각화 생성",
                                            "status": "completed",
                                            "execution_time": "3.2s"
                                        }
                                    ],
                                    "total_execution_time": "11.2s",
                                    "success_rate": "100%"
                                })
                            }
                        ]
                    }
                }
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: plan_executed 형식 파싱 확인
        assert len(result) == 3
        assert result[0]["step_number"] == 1
        assert result[0]["agent_name"] == "📁 Data Loader"
        assert result[1]["step_number"] == 2
        assert result[1]["agent_name"] == "🔍 EDA Tools"
        assert result[2]["step_number"] == 3
        assert result[2]["agent_name"] == "📊 Data Visualization"
    
    def test_complex_agent_name_variations(self, client):
        """다양한 에이전트 이름 변형 처리 테스트"""
        # Given: 다양한 에이전트 이름 형식
        orchestrator_response = {
            "result": {
                "steps": [
                    {"step": 1, "agent_name": "data_loader", "task": "Load data"},
                    {"step": 2, "agent": "AI_DS_Team DataCleaningAgent", "description": "Clean data"},
                    {"step": 3, "agent_name": "EDAToolsAgent", "task_description": "Analyze data"},
                    {"step": 4, "agent": "feature_engineering", "description": "Engineer features"},
                    {"step": 5, "agent_name": "AI_DS_Team H2OMLAgent", "task": "Train model"},
                    {"step": 6, "agent": "mlflow_tools", "description": "Track experiments"}
                ]
            }
        }
        
        # When: 파싱 실행
        result = client.parse_orchestration_plan(orchestrator_response)
        
        # Then: 모든 에이전트 이름이 올바르게 매핑되었는지 확인
        expected_agents = [
            "📁 Data Loader",
            "🧹 Data Cleaning", 
            "🔍 EDA Tools",
            "⚙️ Feature Engineering",
            "🤖 H2O ML",
            "📈 MLflow Tools"
        ]
        
        assert len(result) == 6
        for i, expected_agent in enumerate(expected_agents):
            assert result[i]["agent_name"] == expected_agent
    
    def test_empty_and_malformed_responses(self, client):
        """빈 응답과 잘못된 형식 응답 처리 테스트"""
        # Given: 다양한 문제가 있는 응답들
        problematic_responses = [
            # 빈 result
            {"result": {}},
            
            # history는 있지만 빈 배열
            {"result": {"history": [], "status": {"state": "completed"}}},
            
            # 잘못된 JSON이 포함된 응답
            {
                "result": {
                    "history": [
                        {
                            "role": "agent",
                            "message": {
                                "parts": [{"kind": "text", "text": "{ invalid json }"}]
                            }
                        }
                    ]
                }
            },
            
            # steps가 있지만 빈 배열
            {"result": {"steps": []}},
            
            # 에이전트 정보가 없는 단계
            {"result": {"steps": [{"step": 1, "description": "Some task"}]}},
        ]
        
        # When & Then: 각 응답에 대해 빈 결과 또는 적절한 처리 확인
        for response in problematic_responses:
            result = client.parse_orchestration_plan(response)
            # 모든 문제가 있는 응답은 빈 리스트를 반환하거나 적절히 처리되어야 함
            assert isinstance(result, list)
    
    def test_large_response_performance(self, client):
        """대용량 응답 처리 성능 테스트"""
        # Given: 많은 단계가 포함된 응답
        large_steps = []
        for i in range(50):  # 50개 단계
            large_steps.append({
                "step_number": i + 1,
                "agent_name": f"data_loader" if i % 2 == 0 else "eda_tools",
                "task_description": f"Task {i + 1}: Process data batch {i + 1}",
                "reasoning": f"Step {i + 1} is required for comprehensive analysis"
            })
        
        orchestrator_response = {
            "result": {
                "history": [
                    {
                        "role": "agent",
                        "message": {
                            "parts": [
                                {
                                    "kind": "text",
                                    "text": json.dumps({"steps": large_steps})
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        # When: 파싱 실행 (성능 측정)
        import time
        start_time = time.time()
        result = client.parse_orchestration_plan(orchestrator_response)
        end_time = time.time()
        
        # Then: 성능 및 정확성 확인
        assert len(result) == 50
        assert end_time - start_time < 1.0  # 1초 이내 처리
        
        # 모든 단계가 올바르게 매핑되었는지 확인
        for i, step in enumerate(result):
            assert step["step_number"] == i + 1
            expected_agent = "📁 Data Loader" if i % 2 == 0 else "🔍 EDA Tools"
            assert step["agent_name"] == expected_agent 