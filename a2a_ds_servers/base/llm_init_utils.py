"""LLM 초기화 유틸리티 함수들

Ollama 사용 시 API 키 체크를 건너뛰는 등의 공통 로직을 제공합니다.
"""

import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

def check_llm_requirements() -> bool:
    """
    LLM 사용을 위한 요구사항 체크
    
    Returns:
        bool: LLM 초기화가 가능한지 여부
        
    Raises:
        ValueError: LLM 초기화에 필요한 조건이 충족되지 않은 경우
    """
    # LLM Provider 확인
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    logger.info(f"🔍 LLM Provider: {llm_provider}")
    
    # Ollama 사용 시에는 API 키 체크 건너뛰기
    if llm_provider == 'ollama':
        logger.info("✅ Ollama 사용 - API 키 체크 건너뛰기")
        return True
    
    # 다른 Provider 사용 시 API 키 체크
    api_key = (
        os.getenv('OPENAI_API_KEY') or 
        os.getenv('ANTHROPIC_API_KEY') or 
        os.getenv('GOOGLE_API_KEY')
    )
    
    if not api_key:
        raise ValueError(
            f"No LLM API key found for provider '{llm_provider}'. "
            "Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY environment variable."
        )
    
    logger.info(f"✅ API 키 확인됨 for provider: {llm_provider}")
    return True

def create_llm_with_fallback() -> Any:
    """
    LLM 클라이언트 생성 (폴백 포함)
    
    Returns:
        LLM 클라이언트 인스턴스
        
    Raises:
        RuntimeError: LLM 초기화에 실패한 경우
    """
    try:
        # 요구사항 체크
        check_llm_requirements()
        
        # LLM 클라이언트 생성
        from core.llm_factory import create_llm_instance
        llm = create_llm_instance()
        
        logger.info("✅ LLM 클라이언트 초기화 성공")
        return llm
        
    except Exception as e:
        logger.error(f"❌ LLM 초기화 실패: {e}")
        raise RuntimeError("LLM initialization is required for operation") from e

def safe_llm_init(agent_name: str = "Agent") -> tuple[Any, Any]:
    """
    안전한 LLM 및 에이전트 초기화
    
    Args:
        agent_name: 에이전트 이름 (로깅용)
        
    Returns:
        tuple: (llm_instance, agent_instance) - agent_instance는 None일 수 있음
        
    Raises:
        RuntimeError: LLM 초기화에 실패한 경우
    """
    logger.info(f"🚀 {agent_name} LLM 초기화 시작")
    
    try:
        llm = create_llm_with_fallback()
        logger.info(f"✅ {agent_name} LLM 초기화 완료")
        return llm, None
        
    except Exception as e:
        logger.error(f"❌ {agent_name} LLM 초기화 실패: {e}")
        raise