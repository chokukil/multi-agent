"""
AI Data Science Team A2A Wrapper Utilities

A2A SDK와 AI DS Team 라이브러리 간의 호환성을 위한 유틸리티 함수들
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from uuid import uuid4

from a2a.server.agent_execution.agent_executor import RequestContext
from a2a.types import Part, TextPart, Message, Role

logger = logging.getLogger(__name__)


def extract_user_input(context: RequestContext) -> str:
    """
    RequestContext에서 사용자 입력 텍스트를 추출합니다.
    
    Args:
        context: A2A RequestContext 객체
        
    Returns:
        str: 추출된 사용자 입력 텍스트
    """
    try:
        if not context.message or not context.message.parts:
            return ""
            
        user_text = ""
        for part in context.message.parts:
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                user_text += part.root.text + " "
            elif hasattr(part, 'text'):
                user_text += part.text + " "
                
        return user_text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting user input: {e}")
        return ""


def safe_get_workflow_summary(agent: Any, fallback_message: str = None) -> str:
    """
    AI DS Team 에이전트에서 안전하게 workflow summary를 가져옵니다.
    
    Args:
        agent: AI DS Team 에이전트 인스턴스
        fallback_message: get_workflow_summary 실패 시 사용할 기본 메시지
        
    Returns:
        str: 워크플로우 요약 또는 기본 메시지
    """
    try:
        # get_workflow_summary 메서드가 존재하는지 확인
        if hasattr(agent, 'get_workflow_summary'):
            return agent.get_workflow_summary(markdown=True)
        else:
            logger.warning(f"Agent {type(agent).__name__} does not have get_workflow_summary method")
            return fallback_message or f"✅ {type(agent).__name__} 작업이 완료되었습니다."
            
    except AttributeError as e:
        logger.warning(f"get_workflow_summary AttributeError: {e}")
        return fallback_message or "✅ 데이터 처리 작업이 완료되었습니다."
        
    except Exception as e:
        logger.error(f"Error getting workflow summary: {e}")
        return fallback_message or f"✅ 작업이 완료되었습니다. (오류: {str(e)})"


def create_agent_response(
    content: str,
    response_type: str = "text",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    A2A 응답 형식에 맞는 에이전트 응답을 생성합니다.
    
    Args:
        content: 응답 내용
        response_type: 응답 타입 ("text", "markdown", "json" 등)
        metadata: 추가 메타데이터
        
    Returns:
        Dict[str, Any]: A2A 형식의 응답
    """
    response = {
        "content": content,
        "type": response_type,
        "timestamp": datetime.now().isoformat(),
        "success": True
    }
    
    if metadata:
        response["metadata"] = metadata
        
    return response


def format_streaming_chunk(
    content: str,
    chunk_id: int,
    is_final: bool = False,
    agent_name: str = "AI DS Agent"
) -> Dict[str, Any]:
    """
    스트리밍용 청크 데이터를 포맷합니다.
    
    Args:
        content: 청크 내용
        chunk_id: 청크 ID
        is_final: 마지막 청크 여부
        agent_name: 에이전트 이름
        
    Returns:
        Dict[str, Any]: 포맷된 스트리밍 청크
    """
    return {
        "type": "stream_chunk",
        "chunk_id": chunk_id,
        "content": content,
        "is_final": is_final,
        "agent": agent_name,
        "timestamp": datetime.now().isoformat()
    }


def convert_ai_ds_response_to_a2a(
    ai_ds_response: Any,
    agent_name: str = "AI DS Agent"
) -> Dict[str, Any]:
    """
    AI DS Team 응답을 A2A 형식으로 변환합니다.
    
    Args:
        ai_ds_response: AI DS Team 에이전트 응답
        agent_name: 에이전트 이름
        
    Returns:
        Dict[str, Any]: A2A 형식 응답
    """
    try:
        # AI DS Team 응답이 딕셔너리인 경우
        if isinstance(ai_ds_response, dict):
            content = ai_ds_response.get("messages", [])
            if content and isinstance(content, list) and len(content) > 0:
                # 마지막 메시지 추출
                last_message = content[-1]
                if hasattr(last_message, 'content'):
                    return create_agent_response(
                        content=last_message.content,
                        metadata={"agent": agent_name, "source": "ai_ds_team"}
                    )
                elif isinstance(last_message, str):
                    return create_agent_response(
                        content=last_message,
                        metadata={"agent": agent_name, "source": "ai_ds_team"}
                    )
        
        # 문자열 응답인 경우
        elif isinstance(ai_ds_response, str):
            return create_agent_response(
                content=ai_ds_response,
                metadata={"agent": agent_name, "source": "ai_ds_team"}
            )
            
        # 기타 경우 문자열로 변환
        else:
            return create_agent_response(
                content=str(ai_ds_response),
                metadata={"agent": agent_name, "source": "ai_ds_team"}
            )
            
    except Exception as e:
        logger.error(f"Error converting AI DS response to A2A: {e}")
        return create_agent_response(
            content=f"응답 변환 중 오류가 발생했습니다: {str(e)}",
            metadata={"agent": agent_name, "error": True}
        )


def validate_agent_instance(agent: Any, expected_methods: List[str] = None) -> bool:
    """
    AI DS Team 에이전트 인스턴스가 유효한지 검증합니다.
    
    Args:
        agent: 검증할 에이전트 인스턴스
        expected_methods: 필수 메서드 목록
        
    Returns:
        bool: 유효성 검증 결과
    """
    if expected_methods is None:
        expected_methods = ["invoke"]  # 기본적으로 invoke 메서드는 있어야 함
        
    try:
        for method_name in expected_methods:
            if not hasattr(agent, method_name):
                logger.warning(f"Agent {type(agent).__name__} missing method: {method_name}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating agent instance: {e}")
        return False


def create_error_response(error_message: str, agent_name: str = "AI DS Agent") -> Dict[str, Any]:
    """
    오류 응답을 생성합니다.
    
    Args:
        error_message: 오류 메시지
        agent_name: 에이전트 이름
        
    Returns:
        Dict[str, Any]: 오류 응답
    """
    return {
        "content": f"❌ 오류가 발생했습니다: {error_message}",
        "type": "error",
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "metadata": {
            "agent": agent_name,
            "error": True,
            "error_message": error_message
        }
    } 