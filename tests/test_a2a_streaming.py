"""
A2A 스트리밍 시스템 단위 테스트
A2A SDK 0.2.9 표준 스트리밍 프로토콜 검증
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from a2a_ds_servers.a2a_orchestrator import RealTimeStreamingTaskUpdater
from core.a2a.a2a_streamlit_client import A2AStreamlitClient


class TestRealTimeStreamingTaskUpdater:
    """RealTimeStreamingTaskUpdater 단위 테스트"""
    
    @pytest.fixture
    def mock_event_queue(self):
        """EventQueue 모킹"""
        return AsyncMock()
    
    @pytest.fixture
    def streaming_updater(self, mock_event_queue):
        """RealTimeStreamingTaskUpdater 인스턴스"""
        updater = RealTimeStreamingTaskUpdater(
            event_queue=mock_event_queue,
            task_id="test_task_123",
            context_id="test_context_456"
        )
        # update_status 메서드를 AsyncMock으로 패치
        updater.update_status = AsyncMock()
        return updater
    
    @pytest.mark.asyncio
    async def test_stream_chunk_with_final_flag(self, streaming_updater):
        """청크 스트리밍에서 final 플래그 테스트"""
        # Given
        test_chunk = "테스트 청크 메시지"
        
        # When - 중간 청크 (final=False)
        await streaming_updater.stream_chunk(test_chunk, final=False)
        
        # Then
        streaming_updater.update_status.assert_called()
        call_args = streaming_updater.update_status.call_args
        
        # TaskState.working이 호출되었는지 확인
        from a2a.types import TaskState
        assert call_args[0][0] == TaskState.working
        
        # 메시지 내용 확인
        message = call_args[1]['message']
        assert hasattr(message, 'parts')
        assert len(message.parts) > 0
        assert message.parts[0].root.text == test_chunk
    
    @pytest.mark.asyncio
    async def test_stream_chunk_final_completion(self, streaming_updater):
        """최종 청크에서 완료 상태 테스트"""
        # Given
        final_chunk = "최종 응답 메시지"
        
        # When - 최종 청크 (final=True)
        await streaming_updater.stream_chunk(final_chunk, final=True)
        
        # Then
        call_args = streaming_updater.update_status.call_args
        from a2a.types import TaskState
        assert call_args[0][0] == TaskState.completed
    
    @pytest.mark.asyncio
    async def test_character_buffering_and_flush(self, streaming_updater):
        """문자 단위 버퍼링 및 플러시 테스트"""
        # Given
        streaming_updater.buffer_size = 10  # 작은 버퍼 크기로 테스트
        
        # When - 여러 문자 입력
        for char in "안녕하세요! 테스트입니다.":
            await streaming_updater.stream_character(char)
        
        # Then - 버퍼 플러시가 여러 번 호출되었는지 확인
        assert streaming_updater.update_status.call_count > 1
    
    @pytest.mark.asyncio
    async def test_stream_final_response_chunking(self, streaming_updater):
        """최종 응답의 청크 분할 테스트"""
        # Given
        multiline_response = """## 분석 결과
        
첫 번째 줄입니다.
두 번째 줄입니다.
세 번째 줄입니다."""
        
        # When
        await streaming_updater.stream_final_response(multiline_response)
        
        # Then - 여러 청크로 분할되어 전송되었는지 확인
        assert streaming_updater.update_status.call_count > 1
        
        # 마지막 호출이 completed 상태인지 확인
        last_call = streaming_updater.update_status.call_args_list[-1]
        from a2a.types import TaskState
        assert last_call[0][0] == TaskState.completed
    
    @pytest.mark.asyncio
    async def test_add_artifact_with_streaming_confirmation(self, streaming_updater):
        """아티팩트 추가 시 스트리밍 확인 메시지 테스트"""
        # Given
        from a2a.types import TextPart
        test_parts = [TextPart(text="테스트 아티팩트 내용")]
        artifact_name = "test_artifact"
        
        # When
        await streaming_updater.add_artifact(
            parts=test_parts,
            name=artifact_name,
            metadata={"content_type": "text/plain"}
        )
        
        # Then - 부모 클래스 메서드 호출 + 스트리밍 확인 메시지
        assert streaming_updater.update_status.call_count >= 1


class TestA2AStreamlitClient:
    """A2AStreamlitClient 스트리밍 기능 테스트"""
    
    @pytest.fixture
    def mock_agents_info(self):
        """테스트용 에이전트 정보"""
        return {
            "Data Loader": {
                "port": 8307,
                "description": "데이터 로딩 전문 에이전트"
            },
            "Orchestrator": {
                "port": 8100,
                "description": "오케스트레이터"
            }
        }
    
    @pytest.fixture
    def a2a_client(self, mock_agents_info):
        """A2AStreamlitClient 인스턴스"""
        return A2AStreamlitClient(mock_agents_info, timeout=30.0)
    
    @pytest.mark.asyncio
    async def test_stream_task_message_chunking(self, a2a_client):
        """스트리밍 태스크의 메시지 청킹 테스트"""
        # Given
        mock_response = {
            "result": {
                "message": {
                    "parts": [
                        {
                            "kind": "text",
                            "text": "이것은 긴 메시지입니다. 여러 청크로 분할되어야 합니다. 스트리밍 효과를 테스트하기 위한 메시지입니다."
                        }
                    ]
                }
            }
        }
        
        with patch.object(a2a_client._client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_post.return_value = mock_response_obj
            
            # When
            chunks = []
            async for chunk in a2a_client.stream_task("Data Loader", "테스트 요청"):
                chunks.append(chunk)
            
            # Then
            assert len(chunks) > 1  # 여러 청크로 분할됨
            
            # 메시지 타입 청크들 확인
            message_chunks = [c for c in chunks if c.get('type') == 'message']
            assert len(message_chunks) > 1
            
            # final 플래그 확인
            final_chunks = [c for c in chunks if c.get('final') == True]
            assert len(final_chunks) == 1  # 마지막 청크만 final=True
    
    @pytest.mark.asyncio
    async def test_stream_task_artifact_handling(self, a2a_client):
        """스트리밍 태스크의 아티팩트 처리 테스트"""
        # Given
        mock_response = {
            "result": {
                "message": {
                    "parts": [{"kind": "text", "text": "데이터 로딩 완료"}]
                },
                "artifacts": [
                    {
                        "name": "data_summary",
                        "metadata": {"content_type": "text/markdown"},
                        "parts": [
                            {"text": "## 데이터 요약\n- 행: 1000\n- 열: 10"}
                        ]
                    }
                ]
            }
        }
        
        with patch.object(a2a_client._client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_post.return_value = mock_response_obj
            
            # When
            chunks = []
            async for chunk in a2a_client.stream_task("Data Loader", "데이터 로드 요청"):
                chunks.append(chunk)
            
            # Then
            artifact_chunks = [c for c in chunks if c.get('type') == 'artifact']
            assert len(artifact_chunks) == 1
            
            artifact = artifact_chunks[0]['content']
            assert artifact['name'] == 'data_summary'
            assert artifact['contentType'] == 'text/markdown'
    
    @pytest.mark.asyncio
    async def test_stream_task_error_handling(self, a2a_client):
        """스트리밍 태스크의 오류 처리 테스트"""
        # Given
        mock_error_response = {
            "error": {
                "code": -32000,
                "message": "에이전트 연결 실패"
            }
        }
        
        with patch.object(a2a_client._client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_error_response
            mock_response_obj.raise_for_status.return_value = None
            mock_post.return_value = mock_response_obj
            
            # When
            chunks = []
            async for chunk in a2a_client.stream_task("Data Loader", "테스트 요청"):
                chunks.append(chunk)
            
            # Then
            assert len(chunks) == 1
            error_chunk = chunks[0]
            assert error_chunk['type'] == 'message'
            assert '오류' in error_chunk['content']['text']
            assert error_chunk['final'] == True


class TestA2AStreamingProtocol:
    """A2A 스트리밍 프로토콜 표준 준수 테스트"""
    
    @pytest.mark.asyncio
    async def test_jsonrpc_message_format(self):
        """JSON-RPC 2.0 메시지 형식 테스트"""
        # Given
        agents_info = {"Orchestrator": {"port": 8100}}
        client = A2AStreamlitClient(agents_info)
        
        with patch.object(client._client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": {"message": {"parts": []}}}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # When
            async for _ in client.stream_task("Orchestrator", "테스트"):
                break
            
            # Then
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            
            # JSON-RPC 2.0 표준 확인
            assert payload['jsonrpc'] == '2.0'
            assert payload['method'] == 'message/send'
            assert 'params' in payload
            assert 'id' in payload
            
            # 메시지 구조 확인
            message = payload['params']['message']
            assert 'messageId' in message
            assert 'role' in message
            assert 'parts' in message
            assert message['role'] == 'user'
    
    def test_message_parts_structure(self):
        """메시지 parts 구조 테스트"""
        # Given
        agents_info = {"Test Agent": {"port": 8000}}
        client = A2AStreamlitClient(agents_info)
        
        # When
        with patch.object(client._client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": {"message": {"parts": []}}}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # 데이터 참조와 함께 테스트
            async def run_test():
                async for _ in client.stream_task("Test Agent", "테스트", data_id="test_data"):
                    break
            
            asyncio.run(run_test())
            
            # Then
            payload = mock_post.call_args[1]['json']
            parts = payload['params']['message']['parts']
            
            # 텍스트 part 확인
            text_parts = [p for p in parts if p.get('kind') == 'text']
            assert len(text_parts) >= 1
            assert text_parts[0]['text'] == '테스트'
    
    @pytest.mark.asyncio
    async def test_final_flag_in_streaming(self):
        """스트리밍에서 final 플래그 테스트"""
        # Given
        agents_info = {"Test Agent": {"port": 8000}}
        client = A2AStreamlitClient(agents_info)
        
        mock_response = {
            "result": {
                "message": {
                    "parts": [{"kind": "text", "text": "첫 번째 메시지"}]
                }
            }
        }
        
        with patch.object(client._client, 'post') as mock_post:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status.return_value = None
            mock_post.return_value = mock_response_obj
            
            # When
            chunks = []
            async for chunk in client.stream_task("Test Agent", "테스트"):
                chunks.append(chunk)
            
            # Then
            # 중간 청크들은 final=False
            intermediate_chunks = chunks[:-1]
            for chunk in intermediate_chunks:
                assert chunk.get('final') == False
            
            # 마지막 청크는 final=True
            final_chunk = chunks[-1]
            assert final_chunk.get('final') == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
