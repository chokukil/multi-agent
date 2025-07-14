#!/usr/bin/env python3
"""
💬 채팅 인터페이스 단위 테스트

ChatGPT/Claude 스타일 채팅 인터페이스의 모든 기능을 pytest로 검증

Test Coverage:
- 메시지 생성 및 관리
- Enter키 실행, Shift+Enter 멀티라인 입력
- 메시지 히스토리 저장/불러오기
- 타이핑 효과 및 스트리밍
- 세션 상태 관리
- 에러 처리 및 복구
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

# 테스트 대상 임포트
from ui.components.chat_interface import (
    ChatInterface, ChatMessage, MessageRole, MessageStatus,
    get_chat_interface, initialize_chat_interface
)

class TestChatMessage:
    """ChatMessage 데이터 클래스 테스트"""
    
    def test_chat_message_creation(self):
        """메시지 생성 테스트"""
        message_id = str(uuid.uuid4())
        content = "안녕하세요!"
        timestamp = datetime.now()
        
        message = ChatMessage(
            id=message_id,
            role=MessageRole.USER,
            content=content,
            timestamp=timestamp
        )
        
        assert message.id == message_id
        assert message.role == MessageRole.USER
        assert message.content == content
        assert message.timestamp == timestamp
        assert message.status == MessageStatus.COMPLETED
        assert message.metadata == {}
    
    def test_chat_message_to_dict(self):
        """메시지 딕셔너리 변환 테스트"""
        message = ChatMessage(
            id="test-id",
            role=MessageRole.ASSISTANT,
            content="테스트 응답",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            status=MessageStatus.STREAMING,
            metadata={"test": "data"}
        )
        
        expected = {
            "id": "test-id",
            "role": "assistant",
            "content": "테스트 응답",
            "timestamp": "2024-01-01T12:00:00",
            "status": "streaming",
            "metadata": {"test": "data"}
        }
        
        assert message.to_dict() == expected
    
    def test_chat_message_from_dict(self):
        """딕셔너리에서 메시지 생성 테스트"""
        data = {
            "id": "test-id",
            "role": "user",
            "content": "테스트 메시지",
            "timestamp": "2024-01-01T12:00:00",
            "status": "completed",
            "metadata": {"source": "test"}
        }
        
        message = ChatMessage.from_dict(data)
        
        assert message.id == "test-id"
        assert message.role == MessageRole.USER
        assert message.content == "테스트 메시지"
        assert message.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert message.status == MessageStatus.COMPLETED
        assert message.metadata == {"source": "test"}

class TestChatInterface:
    """ChatInterface 클래스 테스트"""
    
    @pytest.fixture
    def chat_interface(self):
        """테스트용 채팅 인터페이스 픽스처"""
        with patch('streamlit.session_state', {}):
            return ChatInterface()
    
    def test_chat_interface_initialization(self, chat_interface):
        """채팅 인터페이스 초기화 테스트"""
        assert chat_interface.session_key == "cherry_chat_messages"
        assert chat_interface.input_key == "cherry_chat_input"
        assert chat_interface.editing_key == "cherry_chat_editing"
        assert chat_interface.typing_speed == 0.03
        assert chat_interface.chunk_size == 3
    
    @patch('streamlit.session_state')
    def test_add_message(self, mock_session_state, chat_interface):
        """메시지 추가 테스트"""
        mock_session_state.__getitem__.return_value = []
        mock_session_state.__setitem__ = Mock()
        
        message = chat_interface.add_message(MessageRole.USER, "테스트 메시지")
        
        assert isinstance(message, ChatMessage)
        assert message.role == MessageRole.USER
        assert message.content == "테스트 메시지"
        assert message.status == MessageStatus.COMPLETED
        
        # 세션 상태 업데이트 확인
        mock_session_state.__setitem__.assert_called()
    
    @patch('streamlit.session_state')
    def test_add_streaming_message(self, mock_session_state, chat_interface):
        """스트리밍 메시지 시작 테스트"""
        mock_session_state.__getitem__.return_value = []
        mock_session_state.__setitem__ = Mock()
        
        message_id = chat_interface.add_streaming_message(MessageRole.ASSISTANT)
        
        assert isinstance(message_id, str)
        # UUID 형식 확인
        uuid.UUID(message_id)  # 유효한 UUID가 아니면 예외 발생
        
        # 세션 상태 업데이트 확인
        mock_session_state.__setitem__.assert_called()
    
    @patch('streamlit.session_state')
    def test_update_streaming_message(self, mock_session_state, chat_interface):
        """스트리밍 메시지 업데이트 테스트"""
        # 기존 메시지 설정
        message_id = "test-message-id"
        existing_messages = [{
            "id": message_id,
            "role": "assistant",
            "content": "안녕",
            "timestamp": "2024-01-01T12:00:00",
            "status": "streaming",
            "metadata": {}
        }]
        
        mock_session_state.__getitem__.return_value = existing_messages
        mock_session_state.__setitem__ = Mock()
        
        chat_interface.update_streaming_message(message_id, "하세요!")
        
        # 메시지 업데이트 확인
        mock_session_state.__setitem__.assert_called()
        updated_messages = mock_session_state.__setitem__.call_args[0][1]
        assert updated_messages[0]["content"] == "안녕하세요!"
    
    @patch('streamlit.session_state')
    def test_complete_streaming_message(self, mock_session_state, chat_interface):
        """스트리밍 메시지 완료 테스트"""
        message_id = "test-message-id"
        existing_messages = [{
            "id": message_id,
            "role": "assistant",
            "content": "완성된 메시지",
            "timestamp": "2024-01-01T12:00:00",
            "status": "streaming",
            "metadata": {}
        }]
        
        mock_session_state.__getitem__.return_value = existing_messages
        mock_session_state.__setitem__ = Mock()
        
        chat_interface.complete_streaming_message(message_id)
        
        # 상태 업데이트 확인
        mock_session_state.__setitem__.assert_called()
        updated_messages = mock_session_state.__setitem__.call_args[0][1]
        assert updated_messages[0]["status"] == "completed"
    
    @patch('streamlit.session_state')
    def test_get_messages(self, mock_session_state, chat_interface):
        """메시지 목록 조회 테스트"""
        test_messages = [
            {
                "id": "msg1",
                "role": "user",
                "content": "안녕하세요",
                "timestamp": "2024-01-01T12:00:00",
                "status": "completed",
                "metadata": {}
            },
            {
                "id": "msg2",
                "role": "assistant",
                "content": "안녕하세요!",
                "timestamp": "2024-01-01T12:01:00",
                "status": "completed",
                "metadata": {}
            }
        ]
        
        mock_session_state.get.return_value = test_messages
        
        messages = chat_interface.get_messages()
        
        assert len(messages) == 2
        assert all(isinstance(msg, ChatMessage) for msg in messages)
        assert messages[0].content == "안녕하세요"
        assert messages[1].content == "안녕하세요!"
    
    @patch('streamlit.session_state')
    def test_clear_messages(self, mock_session_state, chat_interface):
        """메시지 히스토리 클리어 테스트"""
        mock_session_state.__setitem__ = Mock()
        
        chat_interface.clear_messages()
        
        mock_session_state.__setitem__.assert_called_with(chat_interface.session_key, [])
    
    @patch('streamlit.session_state')
    def test_get_conversation_context(self, mock_session_state, chat_interface):
        """대화 컨텍스트 추출 테스트"""
        test_messages = [
            {
                "id": "msg1",
                "role": "user",
                "content": "파이썬이 뭔가요?",
                "timestamp": "2024-01-01T12:00:00",
                "status": "completed",
                "metadata": {}
            },
            {
                "id": "msg2",
                "role": "assistant",
                "content": "파이썬은 프로그래밍 언어입니다.",
                "timestamp": "2024-01-01T12:01:00",
                "status": "completed",
                "metadata": {}
            }
        ]
        
        mock_session_state.get.return_value = test_messages
        
        context = chat_interface.get_conversation_context(max_messages=2)
        
        expected = "사용자: 파이썬이 뭔가요?\nAI: 파이썬은 프로그래밍 언어입니다."
        assert context == expected
    
    def test_format_message_content(self, chat_interface):
        """메시지 내용 포맷팅 테스트"""
        # HTML 이스케이프 테스트
        content = "<script>alert('test')</script>"
        formatted = chat_interface._format_message_content(content)
        assert "&lt;script&gt;" in formatted
        assert "&lt;/script&gt;" in formatted
        
        # 줄바꿈 처리 테스트
        content = "첫 번째 줄\n두 번째 줄"
        formatted = chat_interface._format_message_content(content)
        assert "<br>" in formatted
        
        # 인라인 코드 처리 테스트
        content = "`print('hello')`"
        formatted = chat_interface._format_message_content(content)
        assert "<code" in formatted
    
    @patch('streamlit.session_state')
    @patch('streamlit.markdown')
    def test_render_message_user(self, mock_markdown, mock_session_state, chat_interface):
        """사용자 메시지 렌더링 테스트"""
        message = ChatMessage(
            id="test-id",
            role=MessageRole.USER,
            content="테스트 메시지",
            timestamp=datetime.now()
        )
        
        chat_interface.render_message(message)
        
        # markdown 호출 확인
        mock_markdown.assert_called()
        call_args = mock_markdown.call_args[1]
        assert call_args["unsafe_allow_html"] is True
        
        # 사용자 메시지 HTML 확인
        html_content = mock_markdown.call_args[0][0]
        assert "user-message" in html_content
        assert "👤" in html_content
        assert "테스트 메시지" in html_content
    
    @patch('streamlit.session_state')
    @patch('streamlit.markdown')
    def test_render_message_assistant(self, mock_markdown, mock_session_state, chat_interface):
        """AI 메시지 렌더링 테스트"""
        message = ChatMessage(
            id="test-id",
            role=MessageRole.ASSISTANT,
            content="AI 응답",
            timestamp=datetime.now()
        )
        
        chat_interface.render_message(message)
        
        # markdown 호출 확인
        mock_markdown.assert_called()
        
        # AI 메시지 HTML 확인
        html_content = mock_markdown.call_args[0][0]
        assert "ai-message" in html_content
        assert "🍒" in html_content
        assert "AI 응답" in html_content
    
    @patch('streamlit.session_state')
    @patch('streamlit.markdown')
    def test_render_typing_indicator(self, mock_markdown, mock_session_state, chat_interface):
        """타이핑 인디케이터 렌더링 테스트"""
        chat_interface.render_typing_indicator()
        
        mock_markdown.assert_called()
        html_content = mock_markdown.call_args[0][0]
        assert "typing-indicator" in html_content
        assert "CherryAI가 응답하고 있습니다" in html_content
        assert "typing-dots" in html_content
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.markdown')
    def test_render_chat_history_empty(self, mock_markdown, chat_interface):
        """빈 채팅 히스토리 렌더링 테스트"""
        with patch.object(chat_interface, 'get_messages', return_value=[]):
            chat_interface.render_chat_history()
            
            # 환영 메시지 확인
            mock_markdown.assert_called()
            html_content = mock_markdown.call_args[0][0]
            assert "안녕하세요! 저는 CherryAI입니다" in html_content
    
    @patch('streamlit.session_state', {})
    @patch('streamlit.markdown')
    @patch('streamlit.chat_input')
    def test_render_input_area(self, mock_chat_input, mock_markdown, chat_interface):
        """입력 영역 렌더링 테스트"""
        mock_chat_input.return_value = "사용자 입력"
        
        result = chat_interface.render_input_area()
        
        # chat_input 호출 확인
        mock_chat_input.assert_called_once()
        call_kwargs = mock_chat_input.call_args[1]
        assert "placeholder" in call_kwargs
        assert "Enter: 전송, Shift+Enter: 줄바꿈" in call_kwargs["placeholder"]
        
        # JavaScript 삽입 확인
        mock_markdown.assert_called()
        
        assert result == "사용자 입력"

class TestChatInterfaceIntegration:
    """채팅 인터페이스 통합 테스트"""
    
    @patch('streamlit.session_state', {})
    def test_complete_message_flow(self):
        """완전한 메시지 플로우 테스트"""
        chat = ChatInterface()
        
        # 1. 사용자 메시지 추가
        user_msg = chat.add_message(MessageRole.USER, "안녕하세요")
        assert user_msg.role == MessageRole.USER
        assert user_msg.content == "안녕하세요"
        
        # 2. AI 스트리밍 메시지 시작
        ai_msg_id = chat.add_streaming_message(MessageRole.ASSISTANT)
        assert ai_msg_id is not None
        
        # 3. 스트리밍 업데이트
        chat.update_streaming_message(ai_msg_id, "안녕")
        chat.update_streaming_message(ai_msg_id, "하세요!")
        
        # 4. 스트리밍 완료
        chat.complete_streaming_message(ai_msg_id)
        
        # 5. 메시지 확인
        messages = chat.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "안녕하세요"
        assert messages[1].content == "안녕하세요!"
        assert messages[1].status == MessageStatus.COMPLETED
    
    @patch('streamlit.session_state', {})
    def test_conversation_context_generation(self):
        """대화 컨텍스트 생성 테스트"""
        chat = ChatInterface()
        
        # 여러 메시지 추가
        chat.add_message(MessageRole.USER, "첫 번째 질문")
        chat.add_message(MessageRole.ASSISTANT, "첫 번째 답변")
        chat.add_message(MessageRole.USER, "두 번째 질문")
        chat.add_message(MessageRole.ASSISTANT, "두 번째 답변")
        
        # 컨텍스트 추출 (최대 3개)
        context = chat.get_conversation_context(max_messages=3)
        
        # 최근 3개 메시지만 포함되어야 함
        lines = context.split('\n')
        assert len(lines) == 3
        assert "첫 번째 질문" not in context
        assert "첫 번째 답변" in context
        assert "두 번째 질문" in context
        assert "두 번째 답변" in context

class TestChatInterfaceGlobalFunctions:
    """전역 함수 테스트"""
    
    @patch('ui.components.chat_interface._chat_interface_instance', None)
    def test_get_chat_interface_singleton(self):
        """싱글톤 인스턴스 테스트"""
        interface1 = get_chat_interface()
        interface2 = get_chat_interface()
        
        assert interface1 is interface2
        assert isinstance(interface1, ChatInterface)
    
    @patch('ui.components.chat_interface._chat_interface_instance', None)
    def test_initialize_chat_interface(self):
        """채팅 인터페이스 초기화 테스트"""
        interface = initialize_chat_interface()
        
        assert isinstance(interface, ChatInterface)
        
        # 같은 인스턴스 반환 확인
        interface2 = get_chat_interface()
        assert interface is interface2

class TestChatInterfaceErrorHandling:
    """채팅 인터페이스 에러 처리 테스트"""
    
    def test_invalid_message_role(self):
        """잘못된 메시지 역할 처리 테스트"""
        with pytest.raises(ValueError):
            MessageRole("invalid_role")
    
    def test_invalid_message_status(self):
        """잘못된 메시지 상태 처리 테스트"""
        with pytest.raises(ValueError):
            MessageStatus("invalid_status")
    
    @patch('streamlit.session_state', {})
    def test_update_nonexistent_message(self):
        """존재하지 않는 메시지 업데이트 테스트"""
        chat = ChatInterface()
        
        # 존재하지 않는 메시지 업데이트 시도
        chat.update_streaming_message("nonexistent-id", "content")
        
        # 에러 없이 처리되어야 함
        messages = chat.get_messages()
        assert len(messages) == 0
    
    def test_malformed_message_dict(self):
        """잘못된 형식의 메시지 딕셔너리 테스트"""
        with pytest.raises((KeyError, ValueError)):
            ChatMessage.from_dict({
                "id": "test",
                # role 누락
                "content": "test",
                "timestamp": "invalid-timestamp",
                "status": "completed",
                "metadata": {}
            })

# 테스트 실행을 위한 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 