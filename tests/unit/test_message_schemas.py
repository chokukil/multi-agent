# tests/unit/test_message_schemas.py
"""
Pydantic v2 메시지 스키마 단위 테스트
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from core.schemas.messages import (
    MessageType, AgentType, ToolType, 
    ProgressMessage, AgentStartMessage, AgentEndMessage,
    ToolCallMessage, ToolResultMessage, CodeExecutionMessage,
    VisualizationMessage, ErrorMessage, ResponseMessage
)
from core.schemas.message_factory import MessageFactory

class TestMessageTypes:
    """메시지 타입 열거형 테스트"""
    
    def test_message_type_values(self):
        """MessageType 값들이 올바른지 확인"""
        assert MessageType.PROGRESS == "progress"
        assert MessageType.AGENT_START == "agent_start"
        assert MessageType.FINAL_RESPONSE == "final_response"
        assert MessageType.DIRECT_RESPONSE == "direct_response"
    
    def test_agent_type_values(self):
        """AgentType 값들이 올바른지 확인"""
        assert AgentType.ROUTER == "router"
        assert AgentType.EDA_SPECIALIST == "eda_specialist"
        assert AgentType.FINAL_RESPONDER == "final_responder"
    
    def test_tool_type_values(self):
        """ToolType 값들이 올바른지 확인"""
        assert ToolType.PYTHON_REPL == "python_repl_ast"
        assert ToolType.MCP_TOOL == "mcp_tool"
        assert ToolType.LLM == "llm"

class TestProgressMessage:
    """ProgressMessage 테스트"""
    
    def test_valid_progress_message(self):
        """올바른 진행 메시지 생성"""
        msg = ProgressMessage(
            current_step=1,
            total_steps=5,
            step_description="테스트 단계",
            percentage=20.0
        )
        
        assert msg.message_type == MessageType.PROGRESS
        assert msg.current_step == 1
        assert msg.total_steps == 5
        assert msg.step_description == "테스트 단계"
        assert msg.percentage == 20.0
        assert isinstance(msg.timestamp, datetime)
    
    def test_invalid_percentage(self):
        """잘못된 퍼센트 값 검증"""
        with pytest.raises(ValidationError):
            ProgressMessage(
                current_step=1,
                total_steps=5,
                step_description="테스트 단계",
                percentage=150.0  # 100 초과
            )

class TestAgentMessages:
    """에이전트 관련 메시지 테스트"""
    
    def test_agent_start_message(self):
        """에이전트 시작 메시지 테스트"""
        msg = AgentStartMessage(
            agent_type=AgentType.EDA_SPECIALIST,
            task_description="데이터 분석 시작",
            expected_duration=30
        )
        
        assert msg.message_type == MessageType.AGENT_START
        assert msg.agent_type == AgentType.EDA_SPECIALIST
        assert msg.task_description == "데이터 분석 시작"
        assert msg.expected_duration == 30
    
    def test_agent_end_message(self):
        """에이전트 완료 메시지 테스트"""
        msg = AgentEndMessage(
            agent_type=AgentType.EDA_SPECIALIST,
            success=True,
            duration=25.5,
            output_summary="분석 완료"
        )
        
        assert msg.message_type == MessageType.AGENT_END
        assert msg.agent_type == AgentType.EDA_SPECIALIST
        assert msg.success is True
        assert msg.duration == 25.5
        assert msg.output_summary == "분석 완료"

class TestToolMessages:
    """도구 관련 메시지 테스트"""
    
    def test_tool_call_message(self):
        """도구 호출 메시지 테스트"""
        input_data = {"code": "print('hello')"}
        msg = ToolCallMessage(
            tool_name="python_repl",
            tool_type=ToolType.PYTHON_REPL,
            input_data=input_data
        )
        
        assert msg.message_type == MessageType.TOOL_CALL
        assert msg.tool_name == "python_repl"
        assert msg.tool_type == ToolType.PYTHON_REPL
        assert msg.input_data == input_data
        assert msg.call_id is not None
    
    def test_tool_result_message(self):
        """도구 결과 메시지 테스트"""
        msg = ToolResultMessage(
            tool_name="python_repl",
            tool_type=ToolType.PYTHON_REPL,
            call_id="test-id",
            success=True,
            result="hello",
            execution_time=0.1
        )
        
        assert msg.message_type == MessageType.TOOL_RESULT
        assert msg.success is True
        assert msg.result == "hello"
        assert msg.execution_time == 0.1

class TestCodeExecutionMessage:
    """코드 실행 메시지 테스트"""
    
    def test_code_execution_success(self):
        """성공적인 코드 실행 메시지"""
        msg = CodeExecutionMessage(
            code="print('hello')",
            output="hello\n",
            execution_time=0.1,
            has_visualization=False
        )
        
        assert msg.message_type == MessageType.CODE_EXECUTION
        assert msg.code == "print('hello')"
        assert msg.output == "hello"  # Pydantic이 문자열을 strip함
        assert msg.language == "python"  # 기본값
        assert msg.has_visualization is False
    
    def test_code_execution_with_error(self):
        """에러가 있는 코드 실행 메시지"""
        msg = CodeExecutionMessage(
            code="1/0",
            error="ZeroDivisionError: division by zero",
            execution_time=0.05
        )
        
        assert msg.error == "ZeroDivisionError: division by zero"
        assert msg.output is None

class TestVisualizationMessage:
    """시각화 메시지 테스트"""
    
    def test_visualization_message(self):
        """시각화 메시지 생성"""
        msg = VisualizationMessage(
            title="테스트 플롯",
            image_base64="iVBORw0KGgoAAAANSUhEUgA...",
            artifact_id="plot_001"
        )
        
        assert msg.message_type == MessageType.VISUALIZATION
        assert msg.title == "테스트 플롯"
        assert msg.image_format == "png"  # 기본값
        assert msg.artifact_id == "plot_001"

class TestErrorMessage:
    """에러 메시지 테스트"""
    
    def test_error_message(self):
        """에러 메시지 생성"""
        msg = ErrorMessage(
            error_type="ValidationError",
            error_message="잘못된 입력",
            traceback="Traceback...",
            context={"step": 1}
        )
        
        assert msg.message_type == MessageType.ERROR
        assert msg.error_type == "ValidationError"
        assert msg.error_message == "잘못된 입력"
        assert msg.context == {"step": 1}

class TestResponseMessage:
    """응답 메시지 테스트"""
    
    def test_final_response_message(self):
        """최종 응답 메시지"""
        msg = ResponseMessage(
            message_type=MessageType.FINAL_RESPONSE,
            content="분석이 완료되었습니다.",
            artifacts=["plot1.png", "data.csv"]
        )
        
        assert msg.message_type == MessageType.FINAL_RESPONSE
        assert msg.content == "분석이 완료되었습니다."
        assert len(msg.artifacts) == 2
    
    def test_direct_response_message(self):
        """직접 응답 메시지"""
        msg = ResponseMessage(
            message_type=MessageType.DIRECT_RESPONSE,
            content="안녕하세요!"
        )
        
        assert msg.message_type == MessageType.DIRECT_RESPONSE
        assert msg.content == "안녕하세요!"
        assert msg.artifacts == []  # 기본값

class TestMessageFactory:
    """MessageFactory 테스트"""
    
    def test_create_progress(self):
        """진행 메시지 팩토리 테스트"""
        msg = MessageFactory.create_progress(2, 5, "처리 중")
        
        assert isinstance(msg, ProgressMessage)
        assert msg.current_step == 2
        assert msg.total_steps == 5
        assert msg.percentage == 40.0
    
    def test_create_agent_start(self):
        """에이전트 시작 메시지 팩토리 테스트"""
        msg = MessageFactory.create_agent_start(
            AgentType.EDA_SPECIALIST,
            "EDA 분석 시작"
        )
        
        assert isinstance(msg, AgentStartMessage)
        assert msg.agent_type == AgentType.EDA_SPECIALIST
        assert msg.task_description == "EDA 분석 시작"
    
    def test_create_response(self):
        """응답 메시지 팩토리 테스트"""
        msg = MessageFactory.create_response(
            "완료!",
            MessageType.FINAL_RESPONSE,
            artifacts=["result.png"]
        )
        
        assert isinstance(msg, ResponseMessage)
        assert msg.content == "완료!"
        assert msg.message_type == MessageType.FINAL_RESPONSE
        assert msg.artifacts == ["result.png"]
    
    def test_from_legacy_dict_direct_response(self):
        """레거시 딕셔너리에서 직접 응답 변환"""
        legacy_dict = {
            "node": "direct_response",
            "content": "안녕하세요!"
        }
        
        msg = MessageFactory.from_legacy_dict(legacy_dict)
        
        assert isinstance(msg, ResponseMessage)
        assert msg.message_type == MessageType.DIRECT_RESPONSE
        assert msg.content == "안녕하세요!"
        assert msg.node == "direct_response"
    
    def test_from_legacy_dict_final_response(self):
        """레거시 딕셔너리에서 최종 응답 변환"""
        legacy_dict = {
            "node": "final_responder",
            "content": "분석 완료!"
        }
        
        msg = MessageFactory.from_legacy_dict(legacy_dict)
        
        assert isinstance(msg, ResponseMessage)
        assert msg.message_type == MessageType.FINAL_RESPONSE
        assert msg.content == "분석 완료!"
        assert msg.node == "final_responder"

if __name__ == "__main__":
    pytest.main([__file__])