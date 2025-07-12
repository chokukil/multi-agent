#!/usr/bin/env python3
"""
수정된 A2A 메시지 프로토콜 테스트

Part.root 구조를 올바르게 처리하는 A2A 메시지 프로토콜 유틸리티 테스트
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock

# A2A SDK 표준 임포트
from a2a.types import Message, TextPart, Part, TaskState
from a2a.server.agent_execution import RequestContext

# 테스트 대상
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'a2a_ds_servers'))
from utils.a2a_message_protocol_fixed import (
    A2AMessageProtocolFixed, 
    create_user_message_fixed,
    extract_user_input_fixed
)

class TestA2AMessageProtocolFixed:
    """수정된 A2A 메시지 프로토콜 테스트 클래스"""
    
    def test_create_user_message_with_part_root(self):
        """Part.root 구조로 사용자 메시지 생성 테스트"""
        text = "Part.root 구조 테스트"
        message = A2AMessageProtocolFixed.create_user_message(text)
        
        # 기본 구조 검증
        assert isinstance(message, Message)
        assert message.role == "user"
        assert len(message.parts) == 1
        
        # Part.root 구조 검증
        part = message.parts[0]
        assert isinstance(part, Part)
        assert hasattr(part, 'root')
        assert isinstance(part.root, TextPart)
        assert part.root.text == text
        
        # messageId 검증
        assert message.messageId is not None
        assert message.messageId.startswith("user_msg_")
    
    def test_create_agent_message_with_part_root(self):
        """Part.root 구조로 에이전트 메시지 생성 테스트"""
        text = "에이전트 응답 테스트"
        custom_id = "agent_test_123"
        message = A2AMessageProtocolFixed.create_agent_message(text, custom_id)
        
        assert message.role == "agent"
        assert message.messageId == custom_id
        
        # Part.root 구조 검증
        part = message.parts[0]
        assert part.root.text == text
        assert part.root.kind == "text"
    
    def test_extract_text_from_part(self):
        """Part 객체에서 텍스트 추출 테스트"""
        text = "추출할 텍스트"
        text_part = TextPart(text=text)
        part = Part(root=text_part)
        
        extracted = A2AMessageProtocolFixed.extract_text_from_part(part)
        assert extracted == text
        
        # 직접 TextPart인 경우도 테스트
        extracted_direct = A2AMessageProtocolFixed.extract_text_from_part(text_part)
        assert extracted_direct == text
    
    def test_extract_user_input_from_context(self):
        """RequestContext에서 사용자 입력 추출 테스트"""
        # Mock RequestContext 생성
        context = Mock(spec=RequestContext)
        message = A2AMessageProtocolFixed.create_user_message("컨텍스트 테스트")
        context.message = message
        
        extracted = A2AMessageProtocolFixed.extract_user_input(context)
        assert extracted == "컨텍스트 테스트"
    
    def test_message_validation_success(self):
        """올바른 메시지 유효성 검증 테스트"""
        message = A2AMessageProtocolFixed.create_user_message("유효한 메시지")
        is_valid, errors = A2AMessageProtocolFixed.validate_message(message)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_create_a2a_payload_legacy(self):
        """레거시 A2A payload 생성 테스트"""
        message = "레거시 테스트 메시지"
        payload = A2AMessageProtocolFixed.create_a2a_payload(message)
        
        # 기본 구조 검증
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "message/send"
        assert "params" in payload
        
        # 메시지 구조 검증
        msg = payload["params"]["message"]
        assert msg["role"] == "user"
        assert msg["parts"][0]["kind"] == "text"
        assert msg["parts"][0]["text"] == message
        assert "messageId" in msg
    
    def test_global_functions_fixed(self):
        """수정된 전역 함수들 테스트"""
        # create_user_message_fixed
        message = create_user_message_fixed("전역 함수 테스트")
        assert message.role == "user"
        assert message.parts[0].root.text == "전역 함수 테스트"
        
        # extract_user_input_fixed
        context = Mock(spec=RequestContext)
        context.message = message
        extracted = extract_user_input_fixed(context)
        assert extracted == "전역 함수 테스트"
    
    def test_extract_response_text_empty(self):
        """빈 응답에서 텍스트 추출 테스트"""
        empty_response = None
        extracted = A2AMessageProtocolFixed.extract_response_text(empty_response)
        assert extracted == ""
        
        # 빈 객체
        empty_object = Mock()
        extracted_empty = A2AMessageProtocolFixed.extract_response_text(empty_object)
        assert extracted_empty == ""
    
    @pytest.mark.asyncio
    async def test_safe_task_update_success(self):
        """안전한 TaskUpdater 메시지 전송 성공 테스트"""
        # Mock TaskUpdater
        task_updater = AsyncMock()
        task_updater.update_status = AsyncMock()
        
        success = await A2AMessageProtocolFixed.safe_task_update(
            task_updater, 
            TaskState.working, 
            "작업 진행 중"
        )
        
        assert success is True
        task_updater.update_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_safe_task_update_failure(self):
        """TaskUpdater 실패 시 안전 처리 테스트"""
        # 실패하는 Mock
        task_updater = AsyncMock()
        task_updater.update_status = AsyncMock(side_effect=Exception("Mock 오류"))
        
        success = await A2AMessageProtocolFixed.safe_task_update(
            task_updater, 
            TaskState.failed, 
            "실패 메시지"
        )
        
        assert success is False
    
    def test_compatibility_summary(self):
        """호환성 요약 테스트"""
        # A2A SDK 0.2.9 Part.root 구조 준수 확인
        message = A2AMessageProtocolFixed.create_user_message("호환성 확인")
        
        # 필수 구조 검증
        assert hasattr(message, 'messageId')
        assert hasattr(message, 'role')
        assert hasattr(message, 'parts')
        assert len(message.parts) > 0
        assert hasattr(message.parts[0], 'root')
        assert hasattr(message.parts[0].root, 'text')
        
        # 텍스트 추출 가능 확인
        extracted = A2AMessageProtocolFixed.extract_text_from_part(message.parts[0])
        assert extracted == "호환성 확인"
        
        print("✅ A2A SDK 0.2.9 Part.root 구조 호환성 확인 완료")

if __name__ == "__main__":
    # 단독 실행 시 간단한 테스트
    test_instance = TestA2AMessageProtocolFixed()
    test_instance.test_compatibility_summary()
    print("🎉 수정된 A2A 메시지 프로토콜 테스트 완료!") 