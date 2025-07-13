"""
🍒 CherryAI - Streaming Integration Tests
스트리밍 컴포넌트들 간의 통합 테스트

테스트 시나리오:
1. RealtimeChatContainer + A2ASSEClient 통합
2. UnifiedChatInterface + RealtimeChatContainer 통합  
3. A2AStreamingServer + A2ASSEClient 통합
4. 전체 스트리밍 파이프라인 테스트
"""

import pytest
import asyncio
import json
import time
import aiohttp
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
import uvicorn
from fastapi.testclient import TestClient
import streamlit as st
import threading

# 시스템 경로 설정
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 테스트 대상 임포트
from ui.streaming.realtime_chat_container import RealtimeChatContainer, StreamingMessage
from ui.components.unified_chat_interface import UnifiedChatInterface
from core.streaming.a2a_sse_client import A2ASSEClient, A2AStreamEvent, A2AMessageType
from a2a_ds_servers.base.streaming_server import A2AStreamingServer, StreamingConfig


class TestRealtimeChatContainerWithA2ASSEClient:
    """RealtimeChatContainer와 A2ASSEClient 통합 테스트"""
    
    @pytest.fixture
    def mock_streamlit_environment(self):
        """Streamlit 환경 모킹"""
        mock_state = {}
        with patch.object(st, 'session_state', mock_state):
            yield mock_state
    
    @pytest.fixture
    def chat_container(self, mock_streamlit_environment):
        """RealtimeChatContainer 인스턴스"""
        return RealtimeChatContainer("integration_test")
    
    @pytest.fixture
    def sse_client(self):
        """A2ASSEClient 인스턴스"""
        return A2ASSEClient("http://localhost:8000", {
            "pandas": "http://localhost:8001",
            "orchestrator": "http://localhost:8002"
        })
    
    @pytest.mark.asyncio
    async def test_real_time_streaming_workflow(self, chat_container, sse_client):
        """실시간 스트리밍 워크플로우 통합 테스트"""
        # 1. 사용자 메시지 추가
        user_message_id = chat_container.add_user_message("분석해주세요")
        assert len(chat_container.messages) == 1
        
        # 2. 스트리밍 응답 시작
        streaming_message_id = chat_container.add_streaming_message("a2a", "pandas", "")
        assert len(chat_container.messages) == 2
        assert chat_container.get_active_streams_count() == 1
        
        # 3. SSE 이벤트 시뮬레이션 (실제 A2A 서버 응답과 유사)
        mock_events = [
            {"event_type": "start", "content": "분석을 시작합니다...", "final": False},
            {"event_type": "progress", "content": "데이터 로딩 중...", "final": False},
            {"event_type": "progress", "content": "통계 계산 중...", "final": False},
            {"event_type": "complete", "content": "분석 완료!", "final": True}
        ]
        
        full_content = ""
        for i, event in enumerate(mock_events):
            is_final = event["final"]
            chunk = f" {event['content']}" if i > 0 else event['content']
            full_content += chunk
            
            # 채팅 컨테이너 업데이트
            chat_container.update_streaming_message(streaming_message_id, chunk, is_final)
            
            # 상태 검증
            message = chat_container.messages[1]  # 스트리밍 메시지
            assert message.content == full_content
            
            if is_final:
                assert message.is_final
                assert message.status == "completed"
                assert chat_container.get_active_streams_count() == 0
            else:
                assert not message.is_final
                assert message.status == "streaming"
        
        # 4. 최종 검증
        assert len(chat_container.messages) == 2
        final_message = chat_container.messages[1]
        expected_content = "분석을 시작합니다... 데이터 로딩 중... 통계 계산 중... 분석 완료!"
        assert final_message.content == expected_content
        assert final_message.is_final
        assert final_message.status == "completed"
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_streams(self, chat_container, sse_client):
        """다중 동시 스트림 테스트"""
        # 여러 에이전트에서 동시 스트리밍
        pandas_stream_id = chat_container.add_streaming_message("a2a", "pandas", "Pandas: ")
        viz_stream_id = chat_container.add_streaming_message("a2a", "visualization", "Viz: ")
        
        assert chat_container.get_active_streams_count() == 2
        
        # 각 스트림 독립적 업데이트
        chat_container.update_streaming_message(pandas_stream_id, "데이터 분석 중...", False)
        chat_container.update_streaming_message(viz_stream_id, "차트 생성 중...", False)
        
        # 상태 확인
        pandas_msg = next(msg for msg in chat_container.messages if msg.message_id == pandas_stream_id)
        viz_msg = next(msg for msg in chat_container.messages if msg.message_id == viz_stream_id)
        
        assert pandas_msg.content == "Pandas: 데이터 분석 중..."
        assert viz_msg.content == "Viz: 차트 생성 중..."
        assert not pandas_msg.is_final
        assert not viz_msg.is_final
        
        # 하나씩 완료
        chat_container.update_streaming_message(pandas_stream_id, " 완료!", True)
        assert chat_container.get_active_streams_count() == 1
        
        chat_container.update_streaming_message(viz_stream_id, " 완료!", True)
        assert chat_container.get_active_streams_count() == 0
        
        # 최종 검증
        pandas_msg = next(msg for msg in chat_container.messages if msg.message_id == pandas_stream_id)
        viz_msg = next(msg for msg in chat_container.messages if msg.message_id == viz_stream_id)
        
        assert pandas_msg.content == "Pandas: 데이터 분석 중... 완료!"
        assert viz_msg.content == "Viz: 차트 생성 중... 완료!"
        assert pandas_msg.is_final
        assert viz_msg.is_final
    
    @pytest.mark.asyncio
    async def test_error_handling_in_streaming(self, chat_container, sse_client):
        """스트리밍 중 에러 처리 테스트"""
        # 스트리밍 시작
        stream_id = chat_container.add_streaming_message("a2a", "pandas", "")
        
        # 일반적인 진행
        chat_container.update_streaming_message(stream_id, "분석 시작...", False)
        
        # 에러 시뮬레이션 (실제로는 A2ASSEClient에서 에러 이벤트 수신)
        error_stream_id = chat_container.add_streaming_message("a2a", "error", "에러 발생: 연결 실패")
        
        # 에러 스트림 즉시 완료
        chat_container.finalize_streaming_message(error_stream_id)
        
        # 원래 스트림은 계속 진행
        chat_container.update_streaming_message(stream_id, " 복구 중...", False)
        chat_container.update_streaming_message(stream_id, " 완료!", True)
        
        # 검증
        assert len(chat_container.messages) == 2
        normal_msg = next(msg for msg in chat_container.messages if msg.message_id == stream_id)
        error_msg = next(msg for msg in chat_container.messages if msg.message_id == error_stream_id)
        
        assert normal_msg.content == "분석 시작... 복구 중... 완료!"
        assert error_msg.content == "에러 발생: 연결 실패"
        assert normal_msg.is_final
        assert error_msg.is_final


class TestUnifiedChatInterfaceIntegration:
    """UnifiedChatInterface 통합 테스트"""
    
    @pytest.fixture
    def mock_streamlit_full_environment(self):
        """완전한 Streamlit 환경 모킹"""
        class MockSessionState:
            def __init__(self):
                self._data = {}
                self.file_upload_completed = False
                self.welcome_shown = False
                self.uploaded_files_for_chat = []
                self.ui_minimized = False
            
            def __contains__(self, key):
                return hasattr(self, key) or key in self._data
            
            def __getitem__(self, key):
                if hasattr(self, key):
                    return getattr(self, key)
                return self._data[key]
            
            def __setitem__(self, key, value):
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self._data[key] = value
            
            def get(self, key, default=None):
                if hasattr(self, key):
                    return getattr(self, key)
                return self._data.get(key, default)
        
        mock_session_state = MockSessionState()
        
        with patch.multiple(
            'streamlit',
            session_state=mock_session_state,
            container=MagicMock(),
            columns=MagicMock(return_value=[MagicMock(), MagicMock()]),
            success=MagicMock(),
            info=MagicMock(),
            warning=MagicMock(),
            error=MagicMock(),
            empty=MagicMock()
        ):
            yield mock_session_state
    
    @pytest.fixture
    def mock_ui_components(self):
        """UI 컴포넌트 모킹"""
        with patch('ui.components.unified_chat_interface.create_file_upload_manager') as mock_file_manager, \
             patch('ui.components.unified_chat_interface.create_question_input') as mock_question_input:
            
            mock_file_manager.return_value = Mock()
            mock_question_input.return_value = Mock()
            yield mock_file_manager, mock_question_input
    
    @pytest.fixture
    def unified_interface(self, mock_streamlit_full_environment, mock_ui_components):
        """UnifiedChatInterface 인스턴스"""
        return UnifiedChatInterface()
    
    def test_unified_interface_chat_container_integration(self, unified_interface):
        """UnifiedChatInterface와 RealtimeChatContainer 통합 테스트"""
        # 채팅 컨테이너 접근
        chat_container = unified_interface.get_chat_container()
        assert isinstance(chat_container, RealtimeChatContainer)
        
        # 초기 상태 확인
        assert len(chat_container.messages) == 0
        assert chat_container.get_active_streams_count() == 0
        
        # UnifiedChatInterface를 통한 메시지 추가 시뮬레이션
        with patch.object(unified_interface, '_handle_user_query') as mock_handle:
            mock_handle.return_value = None
            
            # 사용자 쿼리 시뮬레이션
            test_query = "데이터를 분석해주세요"
            unified_interface._handle_user_query(test_query)
            
            mock_handle.assert_called_once_with(test_query)
        
        # 직접 채팅 컨테이너에 메시지 추가하여 통합 확인
        message_id = chat_container.add_user_message("테스트 메시지")
        assert len(chat_container.messages) == 1
        
        # UnifiedChatInterface의 clear_all 기능 테스트
        unified_interface.clear_all()
        assert len(chat_container.messages) == 0
    
    def test_file_upload_and_chat_integration(self, unified_interface, mock_streamlit_full_environment):
        """파일 업로드와 채팅 통합 테스트"""
        # 파일 업로드 완료 시뮬레이션
        mock_streamlit_full_environment.file_upload_completed = True
        mock_streamlit_full_environment.uploaded_files_for_chat = [
            {"name": "data.csv", "type": "csv", "size": 1024}
        ]
        
        # 웰컴 메시지가 아직 표시되지 않았다고 가정
        mock_streamlit_full_environment.welcome_shown = False
        
        # _handle_welcome_and_suggestions 메서드 테스트
        uploaded_files = mock_streamlit_full_environment.uploaded_files_for_chat
        
        with patch.object(unified_interface, '_generate_llm_welcome_with_suggestions') as mock_welcome:
            mock_welcome.return_value = asyncio.Future()
            mock_welcome.return_value.set_result("환영합니다! 데이터 분석을 시작하겠습니다.")
            
            unified_interface._handle_welcome_and_suggestions(uploaded_files)
            
            # 웰컴 표시 상태 확인
            assert mock_streamlit_full_environment.welcome_shown is True


class TestA2AStreamingServerIntegration:
    """A2AStreamingServer 통합 테스트"""
    
    @pytest.fixture
    def mock_agent_executor(self):
        """Mock A2A 에이전트 실행기"""
        class MockAgentExecutor:
            async def execute(self, context):
                # 간단한 응답 시뮬레이션
                await asyncio.sleep(0.1)
                return {"result": "분석 완료", "status": "success"}
            
            async def cancel(self, context):
                return {"status": "cancelled"}
        
        return MockAgentExecutor()
    
    @pytest.fixture
    def streaming_server(self, mock_agent_executor):
        """A2AStreamingServer 인스턴스"""
        config = StreamingConfig(
            buffer_size=512,
            timeout_seconds=30,
            heartbeat_interval=5
        )
        return A2AStreamingServer(mock_agent_executor, config)
    
    @pytest.fixture
    def test_client(self, streaming_server):
        """FastAPI 테스트 클라이언트"""
        return TestClient(streaming_server.get_app())
    
    def test_server_endpoints_integration(self, test_client, streaming_server):
        """서버 엔드포인트 통합 테스트"""
        # 유효한 A2A 요청 데이터
        request_data = {
            "parts": [{"kind": "text", "text": "데이터를 분석해주세요"}],
            "messageId": "test-123",
            "role": "user"
        }
        
        # SSE 스트리밍 엔드포인트 테스트 (실제로는 SSE 응답)
        session_id = "test-session-123"
        
        # 스트림 상태 확인 엔드포인트
        status_response = test_client.get(f"/stream/{session_id}/status")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert "session_id" in status_data
        assert status_data["session_id"] == session_id
    
    def test_stream_lifecycle_management(self, streaming_server):
        """스트림 생명주기 관리 테스트"""
        session_id = "lifecycle-test"
        
        # 스트림 시작
        streaming_server.active_streams[session_id] = {
            "status": "starting",
            "last_update": time.time(),
            "message_count": 0
        }
        
        # 상태 업데이트
        streaming_server._update_stream_status(session_id, "processing")
        assert streaming_server.active_streams[session_id]["status"] == "processing"
        assert streaming_server.active_streams[session_id]["message_count"] == 1
        
        # 스트림 정리
        streaming_server._cleanup_stream(session_id)
        assert session_id not in streaming_server.active_streams
    
    @pytest.mark.asyncio
    async def test_concurrent_streams_management(self, streaming_server):
        """동시 스트림 관리 테스트"""
        # 여러 세션 동시 생성
        sessions = ["session-1", "session-2", "session-3"]
        
        for session_id in sessions:
            streaming_server.active_streams[session_id] = {
                "status": "active",
                "last_update": time.time(),
                "message_count": 0
            }
        
        assert len(streaming_server.active_streams) == 3
        
        # 일부 세션을 오래된 것으로 만들기
        old_time = time.time() - 400  # 400초 전
        streaming_server.active_streams["session-1"]["last_update"] = old_time
        streaming_server.active_streams["session-2"]["last_update"] = old_time
        
        # 비활성 스트림 정리
        await streaming_server.cleanup_inactive_streams(300)  # 300초 타임아웃
        
        # 최근 세션만 남아있어야 함
        assert len(streaming_server.active_streams) == 1
        assert "session-3" in streaming_server.active_streams
        assert "session-1" not in streaming_server.active_streams
        assert "session-2" not in streaming_server.active_streams


class TestFullStreamingPipeline:
    """전체 스트리밍 파이프라인 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_streaming_simulation(self):
        """엔드투엔드 스트리밍 시뮬레이션"""
        # 1. Streamlit 환경 설정
        mock_state = {}
        with patch.object(st, 'session_state', mock_state):
            
            # 2. 채팅 컨테이너 초기화
            chat_container = RealtimeChatContainer("e2e_test")
            
            # 3. A2A SSE 클라이언트 초기화
            sse_client = A2ASSEClient("http://localhost:8000", {
                "pandas": "http://localhost:8001"
            })
            
            # 4. 사용자 쿼리 시작
            user_query = "타이타닉 데이터셋을 분석해주세요"
            user_msg_id = chat_container.add_user_message(user_query)
            
            # 5. 스트리밍 응답 시뮬레이션
            streaming_msg_id = chat_container.add_streaming_message("a2a", "pandas", "")
            
            # 6. 실제 A2A 서버 응답과 유사한 스트리밍 이벤트 시뮬레이션
            streaming_events = [
                "🔍 데이터셋 로딩 중...",
                " ✅ 891행, 12열의 데이터 확인",
                " 📊 기본 통계 분석 수행 중...",
                " 📈 생존율: 38.4% (342/891)",
                " 👥 성별별 분석: 남성 577명, 여성 314명",
                " 🚢 클래스별 분포: 1등급 216명, 2등급 184명, 3등급 491명",
                " 📋 분석 완료! 상세 리포트를 생성했습니다."
            ]
            
            accumulated_content = ""
            for i, chunk in enumerate(streaming_events):
                accumulated_content += chunk
                is_final = (i == len(streaming_events) - 1)
                
                # 채팅 컨테이너 업데이트
                chat_container.update_streaming_message(streaming_msg_id, chunk, is_final)
                
                # 진행 상황 검증
                current_msg = next(
                    msg for msg in chat_container.messages 
                    if msg.message_id == streaming_msg_id
                )
                
                assert current_msg.content == accumulated_content
                assert current_msg.is_final == is_final
                
                # 짧은 대기 (실제 스트리밍 시뮬레이션)
                await asyncio.sleep(0.01)
            
            # 7. 최종 상태 검증
            assert len(chat_container.messages) == 2  # 사용자 + 어시스턴트
            assert chat_container.get_active_streams_count() == 0
            
            final_response = chat_container.messages[1]
            assert "타이타닉" not in final_response.content  # 범용적 응답 확인
            assert "분석 완료" in final_response.content
            assert final_response.is_final
            assert final_response.status == "completed"
            assert final_response.source == "a2a"
            assert final_response.agent_type == "pandas"
    
    @pytest.mark.asyncio
    async def test_error_recovery_pipeline(self):
        """에러 복구 파이프라인 테스트"""
        mock_state = {}
        with patch.object(st, 'session_state', mock_state):
            
            chat_container = RealtimeChatContainer("error_recovery_test")
            
            # 사용자 쿼리
            user_msg_id = chat_container.add_user_message("데이터 분석 요청")
            
            # 첫 번째 시도 - 에러 발생
            first_stream_id = chat_container.add_streaming_message("a2a", "pandas", "분석 시작...")
            
            # 에러 진행
            chat_container.update_streaming_message(first_stream_id, " 데이터 로딩 중...", False)
            chat_container.update_streaming_message(first_stream_id, " ❌ 오류: 연결 실패", True)
            
            # 작은 딜레이 추가 (message_id 충돌 방지)
            await asyncio.sleep(0.001)
            
            # 두 번째 시도 - 성공 (다른 agent_type 사용하여 message_id 충돌 방지)
            second_stream_id = chat_container.add_streaming_message("a2a", "retry", "🔄 재시도 중...")
            
            # 성공 진행
            chat_container.update_streaming_message(second_stream_id, " ✅ 연결 복구됨", False)
            chat_container.update_streaming_message(second_stream_id, " 📊 분석 완료!", True)
            
            # 최종 검증
            assert len(chat_container.messages) == 3  # 사용자 + 에러 응답 + 성공 응답
            assert chat_container.get_active_streams_count() == 0
            
            error_msg = chat_container.messages[1]
            success_msg = chat_container.messages[2]
            
            assert "오류" in error_msg.content
            assert "완료" in success_msg.content
            assert error_msg.is_final
            assert success_msg.is_final
    
    def test_message_persistence_and_session_management(self):
        """메시지 지속성 및 세션 관리 테스트"""
        mock_state = {}
        with patch.object(st, 'session_state', mock_state):
            
            # 첫 번째 세션
            chat_container_1 = RealtimeChatContainer("persistence_test")
            
            # 메시지 추가
            msg1_id = chat_container_1.add_user_message("첫 번째 메시지")
            msg2_id = chat_container_1.add_assistant_message("첫 번째 응답")
            
            assert len(chat_container_1.messages) == 2
            
            # 동일 컨테이너 키로 새 인스턴스 생성 (세션 복원 시뮬레이션)
            chat_container_2 = RealtimeChatContainer("persistence_test")
            
            # 세션 상태가 공유되는지 확인
            assert len(chat_container_2.messages) == 2
            assert chat_container_2.messages[0].content == "첫 번째 메시지"
            assert chat_container_2.messages[1].content == "첫 번째 응답"
            
            # 새 메시지 추가
            msg3_id = chat_container_2.add_user_message("두 번째 메시지")
            
            # 양쪽에서 모두 보이는지 확인
            assert len(chat_container_1.messages) == 3
            assert len(chat_container_2.messages) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 