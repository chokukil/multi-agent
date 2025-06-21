# File: core/llm_factory.py
# Location: ./core/llm_factory.py

import os
import logging
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.ollama import ChatOllama
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from .utils.config import get_config

def create_llm_instance(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    streaming: bool = True,
    **kwargs
) -> Any:
    """
    통합 LLM 인스턴스 생성 팩토리
    
    Args:
        provider: LLM 제공자 (OPENAI, OLLAMA)
        model: 모델 이름
        temperature: 온도 설정
        streaming: 스트리밍 여부
        **kwargs: 추가 파라미터
    
    Returns:
        LLM 인스턴스
    """
    # Langfuse 콜백 핸들러 초기화
    langfuse_config = get_config('langfuse')
    if langfuse_config.get('host') and langfuse_config.get('public_key') and langfuse_config.get('secret_key'):
        try:
            handler = CallbackHandler(
                public_key=langfuse_config.get('public_key'),
                secret_key=langfuse_config.get('secret_key'),
                host=langfuse_config.get('host'),
            )
            # kwargs에 콜백 추가
            if 'callbacks' not in kwargs:
                kwargs['callbacks'] = []
            kwargs['callbacks'].append(handler)
            logging.info("Langfuse callback handler initialized.")
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
            # Ollama 설정
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            if model is None:
                model = os.getenv("OLLAMA_MODEL", "llama2")
            
            # ChatOllama 인스턴스 생성
            llm = ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url,
                streaming=streaming,
                **kwargs
            )
            
            logging.info(f"Created Ollama LLM: model={model}, temperature={temperature}")
            
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
            config["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
        else:
            config["error"] = f"Unknown provider: {provider}"
            
    except Exception as e:
        config["error"] = str(e)
    
    return config