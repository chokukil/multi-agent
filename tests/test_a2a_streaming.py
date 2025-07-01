"""
A2A 스트리밍 기능 단위 테스트
pytest로 실행: pytest tests/test_a2a_streaming.py -v
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRealTimeStreamingTaskUpdater:
    """RealTimeStreamingTaskUpdater 단위 테스트"""
    
    @pytest.mark.asyncio
    async def test_streaming_buffer_management(self):
        """스트리밍 버퍼 관리 테스트"""
        # Given
        class MockStreamingTaskUpdater:
            def __init__(self, event_queue, task_id, context_id):
                self.event_queue = event_queue
                self.task_id = task_id
                self.context_id = context_id
                self._buffer = ""
            
            async def stream_character(self, char: str):
                """문자 단위 스트리밍"""
                self._buffer += char
                
            async def stream_chunk(self, chunk: str):
                """청크 단위 스트리밍"""
                self._buffer += chunk
                
            async def get_buffer(self):
                return self._buffer
        
        updater = MockStreamingTaskUpdater(None, "test_task", "test_context")
        
        # When
        await updater.stream_character("H")
        await updater.stream_character("e")
        await updater.stream_chunk("llo World")
        
        # Then
        buffer = await updater.get_buffer()
        assert buffer == "Hello World"


class TestA2AStreamingProtocol:
    """A2A 스트리밍 프로토콜 테스트"""
    
    @pytest.mark.asyncio
    async def test_chunked_response_structure(self):
        """청크 응답 구조 테스트"""
        # Given
        class MockA2AResponse:
            def __init__(self, final: bool, content: str):
                self.final = final
                self.content = content
                self.message_id = "test_msg_123"
                
            def to_dict(self):
                return {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "final": self.final,
                        "messageId": self.message_id,
                        "parts": [{"text": self.content, "type": "text"}],
                        "role": "agent"
                    }
                }
        
        # When
        chunk1 = MockA2AResponse(False, "데이터 로딩 중...")
        chunk2 = MockA2AResponse(False, "분석 진행 중...")
        final_chunk = MockA2AResponse(True, "분석 완료!")
        
        # Then
        assert chunk1.to_dict()["result"]["final"] == False
        assert chunk2.to_dict()["result"]["final"] == False
        assert final_chunk.to_dict()["result"]["final"] == True
        assert final_chunk.to_dict()["result"]["parts"][0]["text"] == "분석 완료!"


class TestAgentStreamingCommunication:
    """에이전트 스트리밍 통신 테스트"""
    
    @pytest.mark.asyncio
    async def test_agent_streaming_response(self):
        """에이전트 스트리밍 응답 테스트"""
        # Given
        class MockA2AAgent:
            def __init__(self, agent_name: str):
                self.agent_name = agent_name
                self.response_chunks = [
                    f"📁 {agent_name} 작업을 시작합니다...",
                    f"🔍 데이터를 분석하고 있습니다...",
                    f"📊 결과를 생성하고 있습니다...",
                    f"✅ {agent_name} 작업이 완료되었습니다!"
                ]
                self.current_chunk = 0
            
            async def stream_response(self):
                """스트리밍 응답 제너레이터"""
                for chunk in self.response_chunks:
                    yield {
                        "agent": self.agent_name,
                        "content": chunk,
                        "final": chunk == self.response_chunks[-1],
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                    await asyncio.sleep(0.1)  # 스트리밍 지연 시뮬레이션
        
        agent = MockA2AAgent("DataLoader")
        
        # When
        collected_chunks = []
        async for chunk in agent.stream_response():
            collected_chunks.append(chunk)
        
        # Then
        assert len(collected_chunks) == 4
        assert collected_chunks[0]["final"] == False
        assert collected_chunks[-1]["final"] == True
        assert "DataLoader" in collected_chunks[0]["content"]
        assert "완료" in collected_chunks[-1]["content"]


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "--tb=short"])
