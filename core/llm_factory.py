# File: core/llm_factory.py
# Location: ./core/llm_factory.py

import os
import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI

# Ollama import - 새로운 패키지를 우선 시도하고 fallback
try:
    from langchain_ollama import ChatOllama
    OLLAMA_IMPORT_SOURCE = "langchain_ollama"
except ImportError:
    try:
        from langchain_community.chat_models.ollama import ChatOllama
        OLLAMA_IMPORT_SOURCE = "langchain_community"
        logging.warning("Using deprecated ChatOllama from langchain_community. Consider upgrading to langchain-ollama.")
    except ImportError:
        ChatOllama = None
        OLLAMA_IMPORT_SOURCE = None
        logging.error("ChatOllama not available. Please install langchain-ollama or langchain-community.")

# Langfuse imports - 2.60.8 버전에 맞는 올바른 import 경로 사용
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler  # 2.60.8에서는 callback 모듈에서 import
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not available. Install langfuse for advanced tracing.")

from .utils.config import get_config

def create_llm_instance(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    streaming: bool = True,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> Any:
    """
    통합 LLM 인스턴스 생성 팩토리
    
    Args:
        provider: LLM 제공자 (OPENAI, OLLAMA)
        model: 모델 이름
        temperature: 온도 설정
        streaming: 스트리밍 여부
        session_id: 세션 ID (Streamlit session_state에서 전달)
        user_id: 사용자 ID
        **kwargs: 추가 파라미터
    
    Returns:
        LLM 인스턴스
    """
    # Langfuse 콜백 핸들러 초기화 - multi_agent_supervisor.py 패턴 사용
    if LANGFUSE_AVAILABLE:
        langfuse_config = get_config('langfuse')
        if langfuse_config.get('host') and langfuse_config.get('public_key') and langfuse_config.get('secret_key'):
            try:
                # 세션 ID를 파라미터에서 가져오거나 환경변수/기본값 사용
                effective_session_id = session_id or os.getenv("THREAD_ID", "default-session")
                effective_user_id = user_id or os.getenv("EMP_NO", "default_user")
                
                # multi_agent_supervisor.py와 동일한 패턴으로 CallbackHandler 직접 초기화
                handler = CallbackHandler(
                    session_id=effective_session_id,
                    user_id=effective_user_id,
                    metadata={
                        "app_type": "llm_factory",
                        "model": model or "unknown",
                        "provider": provider or "unknown",
                        "temperature": temperature,
                        "session_id": effective_session_id
                    }
                )
                
                # kwargs에 콜백 추가
                if 'callbacks' not in kwargs:
                    kwargs['callbacks'] = []
                kwargs['callbacks'].append(handler)
                logging.info(f"Langfuse callback handler initialized with session_id: {effective_session_id}")
            except Exception as e:
                logging.error(f"Failed to initialize Langfuse: {e}")

    # 환경 변수에서 기본값 읽기
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "OPENAI")
    
    provider = provider.upper()
    
    try:
        if provider == "OPENAI":
            # OpenAI 설정
            api_key = os.getenv("OPENAI_API_KEY", "")
            api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            if model is None:
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            
            # ChatOpenAI 인스턴스 생성
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                streaming=streaming,
                api_key=api_key,
                base_url=api_base,
                **kwargs
            )
            
            logging.info(f"Created OpenAI LLM: model={model}, temperature={temperature}")
            
        elif provider == "OLLAMA":
            # Ollama 설정 - OLLAMA_BASE_URL과 OLLAMA_API_BASE 모두 지원
            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            
            if model is None:
                model = os.getenv("OLLAMA_MODEL", "llama2")
            
            # Ollama는 로컬 LLM이므로 긴 타임아웃 설정 (10분)
            ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))  # 10분 기본값
            
            # ChatOllama 인스턴스 생성
            llm = ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url,
                streaming=streaming,
                request_timeout=ollama_timeout,  # 요청 타임아웃 설정
                **kwargs
            )
            
            logging.info(f"Created Ollama LLM: model={model}, base_url={base_url}, timeout={ollama_timeout}s, temperature={temperature}")
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return llm
        
    except Exception as e:
        logging.error(f"Failed to create LLM instance: {e}")
        raise

def validate_llm_config() -> Dict[str, Any]:
    """
    LLM 설정 검증 및 정보 반환
    
    Returns:
        설정 정보 딕셔너리
    """
    provider = os.getenv("LLM_PROVIDER", "OPENAI")
    
    config = {
        "provider": provider,
        "valid": False,
        "error": None
    }
    
    try:
        if provider == "OPENAI":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                config["error"] = "OPENAI_API_KEY not set"
            else:
                config["valid"] = True
                config["model"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                config["api_base"] = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
                
        elif provider == "OLLAMA":
            config["valid"] = True
            config["model"] = os.getenv("OLLAMA_MODEL", "llama2")
            config["base_url"] = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            
        else:
            config["error"] = f"Unknown provider: {provider}"
            
    except Exception as e:
        config["error"] = str(e)
    
    return config