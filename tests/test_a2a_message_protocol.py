#!/usr/bin/env python3
"""
A2A 메시지 프로토콜 통일화 테스트

A2A SDK 0.2.9 표준 메시지 프로토콜 유틸리티의 기능을 검증합니다.
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

# A2A SDK 표준 임포트
from a2a.types import Message, TextPart, TaskState
from a2a.server.agent_execution import RequestContext
from a2a.server.tasks.task_updater import TaskUpdater

# 테스트 대상
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'a2a_ds_servers'))
from utils.a2a_message_protocol import A2AMessageProtocol, create_user_message, extract_user_input

class TestA2AMessageProtocol:
    """A2A 메시지 프로토콜 통일화 테스트 클래스"""
    
    def test_create_user_message(self):
        """사용자 메시지 생성 테스트"""
        text = "테스트 메시지입니다"
        message = A2AMessageProtocol.create_user_message(text)
        
        # 기본 구조 검증
        assert isinstance(message, Message)
        assert message.role == "user"
        assert len(message.parts) == 1
        assert isinstance(message.parts[0], TextPart)
        assert message.parts[0].text == text
        assert message.messageId is not None
        assert len(message.messageId) > 0
    
    def test_create_user_message_with_custom_id(self):
        """커스텀 ID로 사용자 메시지 생성 테스트"""
        text = "커스텀 ID 테스트"
        custom_id = "custom_test_123"
        message = A2AMessageProtocol.create_user_message(text, custom_id)
        
        assert message.messageId == custom_id
        assert message.role == "user"
        assert message.parts[0].text == text
    
    def test_create_agent_message(self):
        """에이전트 메시지 생성 테스트"""
        text = "에이전트 응답입니다"
        message = A2AMessageProtocol.create_agent_message(text)
        
        assert isinstance(message, Message)
        assert message.role == "agent"
        assert message.parts[0].text == text
        assert message.messageId.startswith("agent_msg_")
    
    def test_create_send_request(self):
        """SendMessageRequest 생성 테스트"""
        message = A2AMessageProtocol.create_user_message("테스트")
        request = A2AMessageProtocol.create_send_request(message)
        
        assert request.jsonrpc == "2.0"
        assert request.method == "message/send"
        assert request.params.message == message
        assert request.id is not None
    
    def test_message_validation_success(self):
        """올바른 메시지 유효성 검증 테스트"""
        message = A2AMessageProtocol.create_user_message("유효한 메시지")
        is_valid, errors = A2AMessageProtocol.validate_message(message)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_message_validation_missing_fields(self):
        """필수 필드 누락 메시지 유효성 검증 테스트"""
        # messageId 누락
        message = Message(role="user", parts=[TextPart(text="test")])
        is_valid, errors = A2AMessageProtocol.validate_message(message)
        
        assert is_valid is False
        assert any("messageId" in error for error in errors)
    
    def test_message_validation_invalid_role(self):
        """잘못된 role 유효성 검증 테스트"""
        message = Message(
            messageId="test_123",
            role="invalid_role",
            parts=[TextPart(text="test")]
        )
        is_valid, errors = A2AMessageProtocol.validate_message(message)
        
        assert is_valid is False
        assert any("role" in error for error in errors)
    
    def test_extract_user_input_basic(self):
        """기본 사용자 입력 추출 테스트"""
        # Mock RequestContext 생성
        context = Mock(spec=RequestContext)
        message = A2AMessageProtocol.create_user_message("사용자 입력 테스트")
        context.message = message
        
        extracted = A2AMessageProtocol.extract_user_input(context)
        assert extracted == "사용자 입력 테스트"
    
    def test_extract_user_input_empty_context(self):
        """빈 컨텍스트에서 사용자 입력 추출 테스트"""
        context = Mock(spec=RequestContext)
        context.message = None
        
        extracted = A2AMessageProtocol.extract_user_input(context)
        assert extracted == ""
    
    def test_extract_user_input_multiple_parts(self):
        """여러 Part가 있는 메시지에서 입력 추출 테스트"""
        context = Mock(spec=RequestContext)
        
        # 여러 TextPart를 가진 메시지 생성
        message = Message(
            messageId="multi_part_test",
            role="user",
            parts=[
                TextPart(text="첫 번째 부분"),
                TextPart(text="두 번째 부분")
            ]
        )
        context.message = message
        
        extracted = A2AMessageProtocol.extract_user_input(context)
        assert "첫 번째 부분" in extracted
        assert "두 번째 부분" in extracted
    
    @pytest.mark.asyncio
    async def test_safe_task_update(self):
        """안전한 TaskUpdater 메시지 전송 테스트"""
        # Mock TaskUpdater
        task_updater = AsyncMock(spec=TaskUpdater)
        task_updater.update_status = AsyncMock()
        
        # 메시지 전송 테스트
        success = await A2AMessageProtocol.safe_task_update(
            task_updater, 
            TaskState.working, 
            "작업 진행 중입니다"
        )
        
        assert success is True
        task_updater.update_status.assert_called_once()
        
        # 호출된 인자 검증
        call_args = task_updater.update_status.call_args
        assert call_args[0][0] == TaskState.working  # state 인자
        assert hasattr(call_args[1]['message'], 'messageId')  # message 인자
        assert call_args[1]['message'].role == "agent"
    
    @pytest.mark.asyncio
    async def test_safe_task_update_with_exception(self):
        """TaskUpdater 예외 발생 시 안전 처리 테스트"""
        # 예외 발생 Mock
        task_updater = AsyncMock(spec=TaskUpdater)
        task_updater.update_status = AsyncMock(side_effect=Exception("테스트 예외"))
        
        success = await A2AMessageProtocol.safe_task_update(
            task_updater, 
            TaskState.failed, 
            "실패 메시지"
        )
        
        assert success is False
    
    def test_create_a2a_payload(self):
        """레거시 A2A payload 생성 테스트"""
        message = "테스트 메시지"
        payload = A2AMessageProtocol.create_a2a_payload(message)
        
        # 기본 구조 검증
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "message/send"
        assert "params" in payload
        assert "message" in payload["params"]
        
        # 메시지 구조 검증
        msg = payload["params"]["message"]
        assert msg["role"] == "user"
        assert len(msg["parts"]) == 1
        assert msg["parts"][0]["kind"] == "text"
        assert msg["parts"][0]["text"] == message
        assert "messageId" in msg
    
    def test_create_a2a_payload_with_context(self):
        """컨텍스트 포함 레거시 payload 생성 테스트"""
        message = "컨텍스트 테스트"
        context = "추가 컨텍스트 정보"
        payload = A2AMessageProtocol.create_a2a_payload(message, context)
        
        assert "metadata" in payload["params"]
        assert payload["params"]["metadata"]["context"] == context
        assert "timestamp" in payload["params"]["metadata"]
    
    def test_parse_legacy_response_dict(self):
        """딕셔너리 형태 레거시 응답 파싱 테스트"""
        response_data = {
            "result": "성공적인 응답입니다"
        }
        
        result = A2AMessageProtocol.parse_legacy_response(response_data)
        
        assert result["success"] is True
        assert result["text"] == "성공적인 응답입니다"
        assert result["error"] is None
    
    def test_parse_legacy_response_error(self):
        """오류 포함 레거시 응답 파싱 테스트"""
        response_data = {
            "error": "처리 중 오류 발생"
        }
        
        result = A2AMessageProtocol.parse_legacy_response(response_data)
        
        assert result["success"] is False
        assert result["error"] == "처리 중 오류 발생"
    
    def test_global_functions(self):
        """전역 함수들 동작 테스트"""
        # create_user_message 전역 함수
        message = create_user_message("전역 함수 테스트")
        assert message.role == "user"
        assert message.parts[0].text == "전역 함수 테스트"
        
        # extract_user_input 전역 함수
        context = Mock(spec=RequestContext)
        context.message = message
        extracted = extract_user_input(context)
        assert extracted == "전역 함수 테스트"
    
    def test_compatibility_with_existing_code(self):
        """기존 코드와의 호환성 테스트"""
        # 기존에 사용되던 메시지 형식 지원 확인
        legacy_formats = [
            {"kind": "text", "text": "kind 필드 사용"},
            {"type": "text", "text": "type 필드 사용"}
        ]
        
        for fmt in legacy_formats:
            # 메시지 생성 및 검증 (호환성 확인)
            message = A2AMessageProtocol.create_user_message(fmt["text"])
            assert message.parts[0].text == fmt["text"]
    
    def test_error_handling_robustness(self):
        """오류 처리 견고성 테스트"""
        # 잘못된 응답 데이터 처리
        invalid_responses = [
            None,
            {},
            {"invalid": "structure"},
            "string_response"
        ]
        
        for invalid_response in invalid_responses:
            result = A2AMessageProtocol.parse_legacy_response(invalid_response)
            # 오류가 발생해도 기본 구조는 유지되어야 함
            assert "success" in result
            assert "text" in result
            assert "error" in result
            assert "raw_response" in result

if __name__ == "__main__":
    # 단독 실행 시 간단한 테스트
    pytest.main([__file__, "-v"]) 