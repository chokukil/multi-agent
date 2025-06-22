# tests/integration/test_message_system_integration.py
"""
메시지 시스템 통합 테스트
"""
import pytest
import asyncio
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from core.schemas.messages import MessageType, AgentType, ToolType
from core.schemas.message_factory import MessageFactory
from core.streaming.typed_chat_stream import TypedChatStreamCallback
from core.execution.timeout_manager import TimeoutManager, TaskComplexity

class TestMessageSystemIntegration:
    """메시지 시스템 전체 통합 테스트"""
    
    @pytest.fixture
    def mock_streamlit_container(self):
        """모의 Streamlit 컨테이너"""
        container = Mock()
        container.progress = Mock()
        container.status = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
        container.expander = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
        container.container = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
        container.markdown = Mock()
        return container
    
    @pytest.fixture
    def timeout_manager(self):
        """타임아웃 매니저 인스턴스"""
        return TimeoutManager()
    
    @pytest.fixture
    def callback_with_container(self, mock_streamlit_container):
        """컨테이너가 있는 콜백 인스턴스"""
        return TypedChatStreamCallback(mock_streamlit_container)
    
    def test_full_workflow_simulation(self, callback_with_container, mock_streamlit_container):
        """전체 워크플로우 시뮬레이션 테스트"""
        # 1. 에이전트 시작
        agent_start = MessageFactory.create_agent_start(
            AgentType.EDA_SPECIALIST,
            "데이터 탐색 분석 시작",
            expected_duration=60
        )
        callback_with_container(agent_start)
        
        # 2. 진행 상황 업데이트
        for i in range(1, 4):
            progress = MessageFactory.create_progress(i, 3, f"단계 {i} 처리 중")
            callback_with_container(progress)
        
        # 3. 도구 호출
        tool_call = MessageFactory.create_tool_call(
            "python_repl",
            ToolType.PYTHON_REPL,
            {"code": "df.describe()"}
        )
        callback_with_container(tool_call)
        
        # 4. 도구 결과
        tool_result = MessageFactory.create_tool_result(
            tool_call.call_id,
            "python_repl",
            ToolType.PYTHON_REPL,
            success=True,
            result="       count    mean     std     min     25%\nage  1000.0   35.2    12.1    18.0    25.0\n",
            execution_time=0.125
        )
        callback_with_container(tool_result)
        
        # 5. 코드 실행
        code_execution = MessageFactory.create_code_execution(
            code="import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
            output="   name  age  city\n0  Alice   25  Seoul\n1  Bob     30  Busan",
            execution_time=0.05
        )
        callback_with_container(code_execution)
        
        # 6. 에이전트 완료
        agent_end = MessageFactory.create_agent_end(
            AgentType.EDA_SPECIALIST,
            success=True,
            duration=58.7,
            summary="데이터 탐색 분석 완료: 1000개 행, 3개 컬럼"
        )
        callback_with_container(agent_end)
        
        # 7. 최종 응답
        final_response = MessageFactory.create_response(
            "## 데이터 탐색 분석 결과\n\n데이터셋은 1000개의 행과 3개의 컬럼으로 구성되어 있습니다.",
            MessageType.FINAL_RESPONSE,
            artifacts=["eda_report.html", "summary_stats.csv"]
        )
        callback_with_container(final_response)
        
        # 검증: 모든 UI 컴포넌트가 호출되었는지 확인
        assert mock_streamlit_container.progress.called
        assert mock_streamlit_container.status.called
        assert mock_streamlit_container.expander.called
        assert mock_streamlit_container.container.called
        
        # 최종 응답이 저장되었는지 확인
        assert callback_with_container.final_response is not None
        assert "데이터 탐색 분석 결과" in callback_with_container.final_response
    
    def test_error_recovery_workflow(self, callback_with_container, mock_streamlit_container):
        """에러 복구 워크플로우 테스트"""
        # 1. 에이전트 시작
        agent_start = MessageFactory.create_agent_start(
            AgentType.STATISTICAL_ANALYST,
            "통계 분석 수행"
        )
        callback_with_container(agent_start)
        
        # 2. 도구 실행 실패
        failed_tool = MessageFactory.create_tool_result(
            "call-123",
            "statistical_analysis",
            ToolType.MCP_TOOL,
            success=False,
            result=None,
            error="FileNotFoundError: 데이터 파일을 찾을 수 없습니다",
            execution_time=1.2
        )
        callback_with_container(failed_tool)
        
        # 3. 에러 메시지
        error_msg = MessageFactory.create_error(
            "FileNotFoundError",
            "데이터 파일을 찾을 수 없습니다",
            traceback="Traceback (most recent call last):\n  File...",
            context={"step": "data_loading", "file": "missing_data.csv"}
        )
        callback_with_container(error_msg)
        
        # 4. 복구 시도 - 대체 방법으로 처리
        recovery_code = MessageFactory.create_code_execution(
            code="# 샘플 데이터 생성\nimport numpy as np\ndata = np.random.randn(100)",
            output="샘플 데이터 생성 완료",
            execution_time=0.03
        )
        callback_with_container(recovery_code)
        
        # 5. 에이전트 부분 성공으로 완료
        agent_end = MessageFactory.create_agent_end(
            AgentType.STATISTICAL_ANALYST,
            success=True,  # 복구 성공
            duration=15.3,
            summary="원본 데이터 없음, 샘플 데이터로 분석 완료"
        )
        callback_with_container(agent_end)
        
        # 검증: 에러가 올바르게 표시되었는지 확인
        assert mock_streamlit_container.container.called
        assert callback_with_container.final_response is not None
    
    def test_legacy_message_compatibility(self, callback_with_container):
        """기존 메시지 형식과의 호환성 테스트"""
        # 기존 딕셔너리 형태 메시지들
        legacy_messages = [
            {
                "node": "direct_response",
                "content": "간단한 질문에 대한 직접 응답입니다."
            },
            {
                "node": "final_responder",
                "content": "복잡한 분석의 최종 결과입니다."
            },
            {
                "node": "planner",
                "content": "계획 수립 중입니다..."
            }
        ]
        
        for msg in legacy_messages:
            # 예외 없이 처리되어야 함
            callback_with_container(msg)
        
        # 최종 응답이 설정되었는지 확인 (direct_response나 final_responder에서)
        assert callback_with_container.final_response is not None
    
    def test_timeout_manager_integration(self, timeout_manager):
        """타임아웃 매니저 통합 테스트"""
        # 다양한 복잡도에 대한 타임아웃 확인
        simple_timeout = timeout_manager.get_timeout_by_query_type("simple")
        complex_timeout = timeout_manager.get_timeout_by_query_type("complex")
        
        # 복잡한 작업이 더 긴 타임아웃을 가져야 함
        assert simple_timeout < complex_timeout
        
        # 에이전트별 가중치 적용 확인
        eda_timeout = timeout_manager.get_timeout(TaskComplexity.SIMPLE, "eda_specialist")
        viz_timeout = timeout_manager.get_timeout(TaskComplexity.SIMPLE, "visualization_expert")
        
        # 시각화 전문가가 더 긴 타임아웃을 가져야 함
        assert eda_timeout < viz_timeout
    
    def test_concurrent_message_handling(self, callback_with_container, mock_streamlit_container):
        """동시 메시지 처리 테스트"""
        messages = []
        
        # 여러 에이전트가 동시에 작업하는 상황 시뮬레이션
        for i in range(3):
            agent_start = MessageFactory.create_agent_start(
                [AgentType.EDA_SPECIALIST, AgentType.VISUALIZATION_EXPERT, AgentType.STATISTICAL_ANALYST][i],
                f"작업 {i+1} 수행 중"
            )
            messages.append(agent_start)
            
            progress = MessageFactory.create_progress(1, 2, f"작업 {i+1} 진행 중")
            messages.append(progress)
        
        # 모든 메시지 처리
        for msg in messages:
            callback_with_container(msg)
        
        # UI가 여러 번 호출되었는지 확인
        assert mock_streamlit_container.status.call_count >= 3
        assert mock_streamlit_container.progress.call_count >= 3
    
    def test_visualization_message_integration(self, callback_with_container, mock_streamlit_container):
        """시각화 메시지 통합 테스트"""
        # 시각화 메시지 생성
        viz_message = MessageFactory.create_visualization(
            title="데이터 분포 히스토그램",
            image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
            artifact_id="histogram_001"
        )
        
        callback_with_container(viz_message)
        
        # 컨테이너가 호출되었는지 확인
        assert mock_streamlit_container.container.called

class TestEndToEndWorkflow:
    """End-to-End 워크플로우 테스트"""
    
    def test_simple_query_workflow(self):
        """단순 쿼리 처리 워크플로우"""
        from core.query_router import classify_query_complexity
        
        simple_query = "안녕하세요"
        classification = classify_query_complexity(simple_query)
        
        assert classification.complexity.value == "simple"
        assert classification.confidence >= 0.5  # API 키 없을 때 fallback 고려
    
    def test_complex_query_workflow(self):
        """복잡한 쿼리 처리 워크플로우"""
        from core.query_router import classify_query_complexity
        
        complex_query = "고객 구매 데이터를 분석하여 RFM 세분화를 수행하고, 각 세그먼트별 특성을 시각화한 후 머신러닝 모델로 이탈 고객을 예측해주세요"
        classification = classify_query_complexity(complex_query)
        
        assert classification.complexity.value in ["complex", "moderate"]
        assert classification.confidence >= 0.5  # API 키 없을 때 fallback 고려

if __name__ == "__main__":
    pytest.main([__file__])