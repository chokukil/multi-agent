# tests/unit/test_stream_callback.py
"""
스트림 콜백 단위 테스트
"""
import pytest
from unittest.mock import Mock, MagicMock, patch

from core.streaming.typed_chat_stream import TypedChatStreamCallback
from core.schemas.messages import MessageType, AgentType, ToolType
from core.schemas.message_factory import MessageFactory

@pytest.fixture
def mock_ui_container():
    """Provides a mock Streamlit container."""
    return MagicMock()

@pytest.fixture
def callback(mock_ui_container):
    """Provides an instance of TypedChatStreamCallback."""
    return TypedChatStreamCallback(mock_ui_container)

class TestTypedChatStreamCallback:
    """TypedChatStreamCallback 단위 테스트"""
    
    def test_message_processing_does_not_raise_error(self, callback):
        """Tests that a BaseMessage object can be processed without errors."""
        msg = MessageFactory.create_agent_start(
            AgentType.PLANNER, "Planning", 10
        )
        try:
            callback(msg)
        except Exception as e:
            pytest.fail(f"Processing a valid message should not raise an error: {e}")

    def test_legacy_dict_processing_does_not_raise_error(self, callback):
        """Tests processing of a legacy dictionary message."""
        legacy_msg = {
            "node": "direct_response",
            "content": "Legacy content"
        }
        try:
            callback(legacy_msg)
        except Exception as e:
            pytest.fail(f"Processing a legacy dict should not raise an error: {e}")

    def test_unsupported_message_type_does_not_raise_error(self, callback):
        """Tests graceful handling of an unsupported message type by not raising an exception."""
        unsupported_msg = "just a raw string"
        try:
            callback(unsupported_msg)
        except ValueError:
             pytest.fail("Callback should handle unsupported types gracefully, not raise ValueError.")

if __name__ == "__main__":
    pytest.main([__file__])