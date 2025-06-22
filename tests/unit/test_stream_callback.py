# tests/unit/test_stream_callback.py
"""
스트림 콜백 단위 테스트
"""
import pytest
from unittest.mock import Mock, MagicMock

from core.streaming.typed_chat_stream import TypedChatStreamCallback
from core.schemas.messages import MessageType, AgentType, ToolType
from core.schemas.message_factory import MessageFactory

class TestTypedChatStreamCallback:
    """TypedChatStreamCallback 단위 테스트"""
    
    @pytest.fixture
    def mock_ui_container(self):
        """모의 UI 컨테이너 생성"""
        container = Mock()
        container.progress = Mock()
        container.status = Mock()
        container.expander = Mock()
        container.container = Mock()
        container.markdown = Mock()
        return container
    
    @pytest.fixture
    def callback(self, mock_ui_container):
        """테스트용 콜백 인스턴스 생성"""
        return TypedChatStreamCallback(mock_ui_container)
    
    def test_progress_message_handling(self, callback, mock_ui_container):
        """진행 메시지 처리 테스트"""
        msg = MessageFactory.create_progress(2, 5, "테스트 진행 중")
        
        callback(msg)
        
        # UI 업데이트 호출 확인
        mock_ui_container.progress.assert_called_once()
        args = mock_ui_container.progress.call_args
        assert args[0][0] == 0.4  # 40%
        assert "Step 2/5" in args[1]["text"]
    
    def test_agent_start_handling(self, callback, mock_ui_container):
        """에이전트 시작 메시지 처리 테스트"""
        msg = MessageFactory.create_agent_start(
            AgentType.EDA_SPECIALIST,
            "데이터 분석 시작",
            expected_duration=30
        )
        
        callback(msg)
        
        # status UI 호출 확인
        mock_ui_container.status.assert_called_once()
        args = mock_ui_container.status.call_args
        assert "eda_specialist" in args[0][0]
    
    def test_agent_end_handling(self, callback, mock_ui_container):
        """에이전트 완료 메시지 처리 테스트"""
        msg = MessageFactory.create_agent_end(
            AgentType.EDA_SPECIALIST,
            success=True,
            duration=25.5,
            summary="분석 완료"
        )
        
        callback(msg)
        
        # container UI 호출 확인
        mock_ui_container.container.assert_called()
    
    def test_code_execution_handling(self, callback, mock_ui_container):
        """코드 실행 메시지 처리 테스트"""
        msg = MessageFactory.create_code_execution(
            code="print('hello')",
            output="hello\n",
            execution_time=0.1
        )
        
        callback(msg)
        
        # expander UI 호출 확인
        mock_ui_container.expander.assert_called()
        args = mock_ui_container.expander.call_args
        assert "Python Code Execution" in args[0][0]
    
    def test_direct_response_handling(self, callback, mock_ui_container):
        """직접 응답 메시지 처리 테스트"""
        msg = MessageFactory.create_response(
            "안녕하세요!",
            MessageType.DIRECT_RESPONSE
        )
        
        callback(msg)
        
        # 응답이 버퍼에 저장되는지 확인
        assert callback.final_response == "안녕하세요!"
        assert callback.is_final_responder is True
        
        # UI 업데이트 확인
        mock_ui_container.container.assert_called()
    
    def test_final_response_handling(self, callback, mock_ui_container):
        """최종 응답 메시지 처리 테스트"""
        msg = MessageFactory.create_response(
            "분석이 완료되었습니다.",
            MessageType.FINAL_RESPONSE,
            artifacts=["plot1.png", "data.csv"]
        )
        
        callback(msg)
        
        # 응답이 저장되는지 확인
        assert callback.final_response == "분석이 완료되었습니다."
        assert callback.is_final_responder is True
        
        # UI 업데이트 확인
        mock_ui_container.container.assert_called()
    
    def test_error_handling(self, callback, mock_ui_container):
        """에러 메시지 처리 테스트"""
        msg = MessageFactory.create_error(
            "ValidationError",
            "잘못된 입력값",
            traceback="Traceback..."
        )
        
        callback(msg)
        
        # container UI 호출 확인
        mock_ui_container.container.assert_called()
    
    def test_legacy_dict_conversion(self, callback, mock_ui_container):
        """기존 딕셔너리 메시지 변환 테스트"""
        legacy_msg = {
            "node": "direct_response",
            "content": "테스트 응답"
        }
        
        callback(legacy_msg)
        
        # 변환된 메시지가 올바르게 처리되는지 확인
        assert callback.final_response == "테스트 응답"
        assert callback.is_final_responder is True
    
    def test_unknown_message_handling(self, callback, mock_ui_container):
        """알 수 없는 메시지 타입 처리 테스트"""
        # 알 수 없는 메시지는 딕셔너리 형태로 테스트
        unknown_msg = {
            "node": "unknown_node",
            "content": "unknown content"
        }
        
        callback(unknown_msg)
        
        # 기본 진행 메시지로 변환되어 처리되는지 확인
        # 이 테스트는 예외가 발생하지 않는지만 확인
    
    def test_callback_with_error_callback(self, mock_ui_container):
        """에러 콜백이 있는 경우 테스트"""
        error_callback = Mock()
        callback = TypedChatStreamCallback(
            mock_ui_container,
            error_callback=error_callback
        )
        
        # 처리할 수 없는 메시지 타입으로 에러 유발
        invalid_data = "invalid string data"  # 딕셔너리도 BaseMessage도 아닌 데이터
        
        callback(invalid_data)
        
        # 에러 콜백이 호출되었는지 확인
        error_callback.assert_called()
    
    def test_flush_method(self, callback, mock_ui_container):
        """flush 메서드 테스트"""
        # 버퍼에 데이터 추가
        callback.buffer = ["첫 번째 ", "두 번째"]
        callback.is_final_responder = True
        
        callback.flush()
        
        # UI 업데이트 확인
        mock_ui_container.markdown.assert_called_once_with("첫 번째 두 번째")
        
        # 버퍼가 클리어되었는지 확인
        assert len(callback.buffer) == 0
    
    def test_get_final_response(self, callback):
        """최종 응답 반환 메서드 테스트"""
        callback.final_response = "테스트 응답"
        
        assert callback.get_final_response() == "테스트 응답"

class TestLegacyCompatibility:
    """기존 시스템과의 호환성 테스트"""
    
    @pytest.fixture
    def mock_ui_container(self):
        """모의 UI 컨테이너 생성"""
        container = Mock()
        container.markdown = Mock()
        container.container = Mock()
        return container
    
    @pytest.fixture
    def callback(self, mock_ui_container):
        """테스트용 콜백 인스턴스 생성"""
        return TypedChatStreamCallback(mock_ui_container)
    
    def test_legacy_direct_response_dict(self, callback, mock_ui_container):
        """기존 direct_response 딕셔너리 형태 처리"""
        legacy_msg = {
            "node": "direct_response",
            "content": "기존 응답 형태"
        }
        
        callback(legacy_msg)
        
        assert callback.final_response == "기존 응답 형태"
        assert callback.is_final_responder is True
        mock_ui_container.container.assert_called()
    
    def test_legacy_final_responder_dict(self, callback, mock_ui_container):
        """기존 final_responder 딕셔너리 형태 처리"""
        legacy_msg = {
            "node": "final_responder",
            "content": "최종 응답"
        }
        
        callback(legacy_msg)
        
        assert callback.final_response == "최종 응답"
        assert callback.is_final_responder is True
        mock_ui_container.container.assert_called()
    
    def test_legacy_complex_content_dict(self, callback, mock_ui_container):
        """복잡한 형태의 기존 딕셔너리 처리"""
        legacy_msg = {
            "node": "direct_response",
            "content": {
                "final_response": "중첩된 응답",
                "metadata": {"type": "test"}
            }
        }
        
        callback(legacy_msg)
        
        # 딕셔너리 content는 문자열로 변환되어 처리됨
        assert "중첩된 응답" in callback.final_response or str(legacy_msg["content"]) == callback.final_response

if __name__ == "__main__":
    pytest.main([__file__])