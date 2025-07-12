#!/usr/bin/env python3
"""
A2A 메시지 프로토콜 통일화 유틸리티 (수정된 버전)

A2A SDK 0.2.9의 Part.root 구조를 올바르게 처리하는 표준 유틸리티입니다.
"""

import uuid
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# A2A SDK 0.2.9 표준 임포트
from a2a.types import (
    Message, 
    TextPart, 
    Part,
    SendMessageRequest, 
    MessageSendParams,
    TaskState,
    Role
)
from a2a.server.agent_execution import RequestContext
from a2a.server.tasks.task_updater import TaskUpdater

logger = logging.getLogger(__name__)

class A2AMessageProtocolFixed:
    """A2A SDK 0.2.9 Part.root 구조를 올바르게 처리하는 메시지 프로토콜 클래스"""
    
    @staticmethod
    def create_user_message(text: str, message_id: Optional[str] = None) -> Message:
        """
        표준 사용자 메시지 생성 (Part.root 구조 적용)
        
        Args:
            text: 메시지 텍스트
            message_id: 메시지 ID (생략 시 자동 생성)
            
        Returns:
            Message: A2A 표준 Message 객체
        """
        if not message_id:
            message_id = f"user_msg_{uuid.uuid4().hex[:12]}"
            
        # A2A SDK 0.2.9: Part 객체로 TextPart를 감싸야 함
        text_part = TextPart(text=text)
        part = Part(root=text_part)
            
        return Message(
            messageId=message_id,
            role="user",
            parts=[part]
        )
    
    @staticmethod
    def create_agent_message(text: str, message_id: Optional[str] = None) -> Message:
        """
        표준 에이전트 메시지 생성 (Part.root 구조 적용)
        
        Args:
            text: 메시지 텍스트
            message_id: 메시지 ID (생략 시 자동 생성)
            
        Returns:
            Message: A2A 표준 Message 객체
        """
        if not message_id:
            message_id = f"agent_msg_{uuid.uuid4().hex[:12]}"
            
        # A2A SDK 0.2.9: Part 객체로 TextPart를 감싸야 함
        text_part = TextPart(text=text)
        part = Part(root=text_part)
            
        return Message(
            messageId=message_id,
            role="agent", 
            parts=[part]
        )
    
    @staticmethod
    def extract_user_input(context: RequestContext) -> str:
        """
        RequestContext에서 사용자 입력 추출 (Part.root 방식)
        
        Args:
            context: A2A RequestContext 객체
            
        Returns:
            str: 추출된 사용자 입력 텍스트
        """
        if not context.message or not context.message.parts:
            logger.warning("Context에 메시지가 없음")
            return ""
        
        user_text = ""
        
        for part in context.message.parts:
            try:
                # A2A SDK 0.2.9 표준: part.root를 통한 접근
                if hasattr(part, 'root'):
                    actual_part = part.root
                    if hasattr(actual_part, 'kind') and actual_part.kind == "text":
                        user_text += actual_part.text + " "
                    elif hasattr(actual_part, 'text'):
                        user_text += actual_part.text + " "
                # Fallback: 직접 TextPart인 경우
                elif hasattr(part, 'text'):
                    user_text += part.text + " "
            except Exception as e:
                logger.warning(f"Part 파싱 실패: {e}")
                continue
                
        return user_text.strip()
    
    @staticmethod
    def extract_text_from_part(part: Union[Part, TextPart]) -> str:
        """
        Part 또는 TextPart에서 안전하게 텍스트 추출
        
        Args:
            part: Part 또는 TextPart 객체
            
        Returns:
            str: 추출된 텍스트
        """
        try:
            # Part 객체인 경우 root 접근
            if hasattr(part, 'root'):
                actual_part = part.root
                if hasattr(actual_part, 'text'):
                    return actual_part.text
            # 직접 TextPart인 경우
            elif hasattr(part, 'text'):
                return part.text
        except Exception as e:
            logger.warning(f"텍스트 추출 실패: {e}")
        
        return ""
    
    @staticmethod
    def extract_response_text(response: Any) -> str:
        """
        A2A 응답에서 텍스트 추출 (Part.root 방식)
        
        Args:
            response: A2A 응답 객체
            
        Returns:
            str: 추출된 응답 텍스트
        """
        text_content = ""
        
        try:
            # A2A 응답 구조 파싱
            if hasattr(response, 'root') and response.root:
                result = response.root.result if hasattr(response.root, 'result') else response.root
                
                # TaskStatusUpdateEvent 타입 처리
                if hasattr(result, 'message') and result.message and hasattr(result.message, 'parts'):
                    for part in result.message.parts:
                        text_content += A2AMessageProtocolFixed.extract_text_from_part(part)
                            
                # 직접 parts가 있는 경우
                elif hasattr(result, 'parts') and result.parts:
                    for part in result.parts:
                        text_content += A2AMessageProtocolFixed.extract_text_from_part(part)
                            
                # history 구조가 있는 경우 (일부 에이전트)
                elif hasattr(result, 'history') and result.history:
                    for msg in result.history:
                        if hasattr(msg, 'role') and msg.role == 'agent' and hasattr(msg, 'parts'):
                            for part in msg.parts:
                                text_content += A2AMessageProtocolFixed.extract_text_from_part(part)
                                    
        except Exception as e:
            logger.error(f"응답 텍스트 추출 실패: {e}")
            
        return text_content.strip()
    
    @staticmethod
    def validate_message(message: Message) -> tuple[bool, List[str]]:
        """
        A2A 메시지 유효성 검증
        
        Args:
            message: 검증할 Message 객체
            
        Returns:
            tuple[bool, List[str]]: (유효성, 오류 메시지 리스트)
        """
        errors = []
        
        # messageId 필수 검증
        if not hasattr(message, 'messageId') or not message.messageId:
            errors.append("messageId 필드가 필수입니다")
        
        # role 필수 검증
        if not hasattr(message, 'role') or not message.role:
            errors.append("role 필드가 필수입니다")
        elif message.role not in ["user", "agent"]:
            errors.append(f"role은 'user' 또는 'agent'여야 합니다: {message.role}")
        
        # parts 필수 검증
        if not hasattr(message, 'parts') or not message.parts:
            errors.append("parts 필드가 필수입니다")
        else:
            for i, part in enumerate(message.parts):
                # Part.root 구조 확인
                text_found = False
                try:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        text_found = True
                    elif hasattr(part, 'text'):
                        text_found = True
                except:
                    pass
                
                if not text_found:
                    errors.append(f"Part {i}: text 내용이 없습니다")
        
        return len(errors) == 0, errors
    
    @staticmethod
    async def safe_task_update(task_updater: TaskUpdater, state: TaskState, 
                             text: str, message_id: Optional[str] = None) -> bool:
        """
        안전한 TaskUpdater 메시지 전송 (Part.root 구조 적용)
        
        Args:
            task_updater: TaskUpdater 인스턴스
            state: TaskState (working, completed, failed)
            text: 메시지 텍스트
            message_id: 메시지 ID (생략 시 자동 생성)
            
        Returns:
            bool: 전송 성공 여부
        """
        try:
            message = A2AMessageProtocolFixed.create_agent_message(text, message_id)
            
            # 메시지 유효성 검증
            is_valid, errors = A2AMessageProtocolFixed.validate_message(message)
            if not is_valid:
                logger.error(f"TaskUpdate 메시지 유효성 검증 실패: {errors}")
                return False
            
            await task_updater.update_status(state, message=message)
            return True
            
        except Exception as e:
            logger.error(f"TaskUpdate 전송 실패: {e}")
            return False
    
    @staticmethod
    def create_a2a_payload(message: str, agent_context: Optional[str] = None) -> Dict[str, Any]:
        """
        레거시 A2A payload 형식 생성 (하위 호환성)
        
        Args:
            message: 메시지 텍스트
            agent_context: 추가 컨텍스트 (선택적)
            
        Returns:
            Dict[str, Any]: A2A payload 딕셔너리
        """
        message_id = f"legacy_msg_{uuid.uuid4().hex[:12]}"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",  # "kind"와 "type" 모두 지원되지만 "kind" 권장
                            "text": message
                        }
                    ]
                }
            },
            "id": message_id
        }
        
        if agent_context:
            payload["params"]["metadata"] = {
                "context": agent_context,
                "timestamp": datetime.now().isoformat()
            }
        
        return payload

# 편의를 위한 전역 함수들 (수정된 버전)
def create_user_message_fixed(text: str, message_id: Optional[str] = None) -> Message:
    """전역 함수: 사용자 메시지 생성 (Part.root 구조 적용)"""
    return A2AMessageProtocolFixed.create_user_message(text, message_id)

def create_agent_message_fixed(text: str, message_id: Optional[str] = None) -> Message:
    """전역 함수: 에이전트 메시지 생성 (Part.root 구조 적용)"""
    return A2AMessageProtocolFixed.create_agent_message(text, message_id)

def extract_user_input_fixed(context: RequestContext) -> str:
    """전역 함수: 사용자 입력 추출 (Part.root 구조 적용)"""
    return A2AMessageProtocolFixed.extract_user_input(context)

def extract_response_text_fixed(response: Any) -> str:
    """전역 함수: 응답 텍스트 추출 (Part.root 구조 적용)"""
    return A2AMessageProtocolFixed.extract_response_text(response)

async def safe_task_update_fixed(task_updater: TaskUpdater, state: TaskState, 
                                text: str, message_id: Optional[str] = None) -> bool:
    """전역 함수: 안전한 TaskUpdater 메시지 전송 (Part.root 구조 적용)"""
    return await A2AMessageProtocolFixed.safe_task_update(task_updater, state, text, message_id)

# 호환성을 위한 기본 유틸리티
def get_user_input_from_context(context: RequestContext) -> str:
    """
    호환성 함수: RequestContext에서 사용자 입력 추출
    기존 코드에서 사용하던 함수명과 호환
    """
    return A2AMessageProtocolFixed.extract_user_input(context) 