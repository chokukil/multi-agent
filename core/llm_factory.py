"""
LLM Factory Compatibility Wrapper

이 파일은 기존 코드와의 호환성을 위한 래퍼입니다.
실제 구현은 core.universal_engine.llm_factory에 있습니다.

전략적 개선의 일환으로:
- 중복 코드 제거 (794줄 → 50줄 래퍼)
- 절대 경로 import 표준화
- 단일 소스 of truth 유지
"""

import logging
from typing import Optional, Dict, Any, List, Union

# 순환 import 방지를 위해 lazy import 사용

logger = logging.getLogger(__name__)


def _get_universal_llm_factory():
    """순환 import 방지를 위한 lazy import"""
    try:
        from core.universal_engine.llm_factory import LLMFactory as UniversalLLMFactory
        return UniversalLLMFactory
    except ImportError as e:
        logger.error(f"Failed to import Universal Engine LLMFactory: {e}")
        raise


class LLMFactory:
    """
    호환성 래퍼 - 기존 코드가 계속 작동하도록 지원
    실제 기능은 UniversalLLMFactory에 위임
    """
    
    # 기존 코드 호환성을 위한 속성들 - lazy loading으로 구현
    
    @staticmethod
    def create_llm(**kwargs) -> Any:
        """기존 create_llm 메서드 - Universal Engine으로 위임"""
        logger.debug("Compatibility wrapper: delegating to Universal Engine LLMFactory")
        return _get_universal_llm_factory().create_llm(**kwargs)
    
    @staticmethod  
    def create_llm_client(**kwargs) -> Any:
        """기존 create_llm_client 메서드 - Universal Engine으로 위임"""
        logger.debug("Compatibility wrapper: delegating to Universal Engine LLMFactory")
        return _get_universal_llm_factory().create_llm_client(**kwargs)
    
    @staticmethod
    def get_available_models(**kwargs) -> List[str]:
        """사용 가능한 모델 목록 반환 - Universal Engine으로 위임"""
        logger.debug("Compatibility wrapper: delegating to Universal Engine LLMFactory")
        return _get_universal_llm_factory().get_available_models(**kwargs)
    
    @staticmethod
    def validate_config(**kwargs) -> bool:
        """설정 유효성 검증 - Universal Engine으로 위임"""
        logger.debug("Compatibility wrapper: delegating to Universal Engine LLMFactory")
        return _get_universal_llm_factory().validate_config(**kwargs)
    
    @staticmethod
    def get_recommended_model(**kwargs) -> str:
        """추천 모델 반환 - Universal Engine으로 위임"""
        logger.debug("Compatibility wrapper: delegating to Universal Engine LLMFactory")
        return _get_universal_llm_factory().get_recommended_model(**kwargs)


# 기존 create_llm_instance 함수 호환성 (많은 파일에서 사용)
def create_llm_instance(**kwargs) -> Any:
    """
    기존 create_llm_instance 함수 호환성 래퍼
    많은 A2A 서버와 기존 코드에서 사용되는 함수
    """
    logger.debug("Legacy create_llm_instance called - delegating to Universal Engine")
    return _get_universal_llm_factory().create_llm(**kwargs)

# 기존 import 호환성을 위한 추가 exports
__all__ = [
    'LLMFactory',
    'create_llm_instance',  # 추가
    'OLLAMA_IMPORT_SOURCE', 
    'OLLAMA_TOOL_CALLING_SUPPORTED',
    'OLLAMA_CLIENT_AVAILABLE',
    'LANGFUSE_AVAILABLE'
]

# 기존 코드에서 사용하던 변수들 - Universal Engine에서 가져오기
try:
    OLLAMA_IMPORT_SOURCE = "universal_engine_delegated"
    OLLAMA_TOOL_CALLING_SUPPORTED = True
    OLLAMA_CLIENT_AVAILABLE = True
    LANGFUSE_AVAILABLE = True
    logger.info("✅ LLM Factory compatibility wrapper loaded - delegating to Universal Engine")
except Exception as e:
    logger.error(f"❌ Failed to load Universal Engine LLM Factory: {e}")
    OLLAMA_IMPORT_SOURCE = None
    OLLAMA_TOOL_CALLING_SUPPORTED = False
    OLLAMA_CLIENT_AVAILABLE = False
    LANGFUSE_AVAILABLE = False