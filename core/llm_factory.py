# File: core/llm_factory.py
# Location: ./core/llm_factory.py

import os
import logging
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI

# Ollama import - 🆕 langchain_ollama 우선 사용 (도구 호출 지원)
try:
    from langchain_ollama import ChatOllama
    OLLAMA_IMPORT_SOURCE = "langchain_ollama"
    OLLAMA_TOOL_CALLING_SUPPORTED = True
    logging.info("✅ Using langchain_ollama - Tool calling supported")
except ImportError:
    try:
        from langchain_community.chat_models.ollama import ChatOllama
        OLLAMA_IMPORT_SOURCE = "langchain_community"
        OLLAMA_TOOL_CALLING_SUPPORTED = False
        logging.warning("⚠️ Using deprecated langchain_community.ChatOllama - Tool calling NOT supported")
    except ImportError:
        ChatOllama = None
        OLLAMA_IMPORT_SOURCE = None
        OLLAMA_TOOL_CALLING_SUPPORTED = False
        logging.error("❌ ChatOllama not available. Please install langchain-ollama")

# Ollama 클라이언트 직접 접근용 (모델 목록 등)
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
    logging.info("✅ Ollama client available")
except ImportError:
    ollama = None
    OLLAMA_CLIENT_AVAILABLE = False
    logging.warning("⚠️ Ollama client not available. Install: uv add ollama")

# Langfuse imports - 2.60.8 버전에 맞는 올바른 import 경로 사용
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler  # 2.60.8에서는 callback 모듈에서 import
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not available. Install langfuse for advanced tracing.")

from core.utils.config import Config

# 🆕 Ollama 모델별 도구 호출 능력 매핑 - 2024년 12월 기준 최신 정보
OLLAMA_TOOL_CALLING_MODELS = {
    # ✅ Llama 계열 - 확실히 지원
    "llama3.1:8b", "llama3.1:70b", "llama3.1:405b",
    "llama3.2:1b", "llama3.2:3b", "llama3.2:11b",
    "llama3.3:70b",  # 새로운 모델
    
    # ✅ Qwen 계열 - 2.5 이상만 지원
    "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", 
    "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
    "qwen2.5-coder:1.5b", "qwen2.5-coder:7b", "qwen2.5-coder:32b",
    "qwen3:8b",  # 사용자 확인: 도구 호출 지원
    
    # ✅ Mistral 계열
    "mistral:7b", "mistral:latest", "mistral-nemo", "mistral-nemo:12b",
    "mixtral:8x7b", "mixtral:8x22b",
    
    # ✅ Google Gemma 계열
    "gemma2:2b", "gemma2:9b", "gemma2:27b",
    
    # ✅ Microsoft Phi 계열
    "phi3:mini", "phi3:medium", "phi3:3.8b", "phi3:14b",
    
    # ✅ Code 전용 모델들
    "codellama:7b", "codellama:13b", "codellama:34b",
    "codegemma:2b", "codegemma:7b",
    "deepseek-coder:1.3b", "deepseek-coder:6.7b", "deepseek-coder:33b",
    "starcoder2:3b", "starcoder2:7b", "starcoder2:15b",
    
    # ✅ 특수 Function Calling 모델들
    "firefunction-v2", "firefunction-v2:70b",
    "nexusraven", "nexusraven:13b",
    
    # ✅ 기타 지원 모델들
    "command-r", "command-r-plus",
    "yi:6b", "yi:9b", "yi:34b",
    "nous-hermes2", "nous-hermes2:10.7b", "nous-hermes2:34b"
}

# 🚨 확실히 지원하지 않는 모델들 (더 상세하게)
OLLAMA_NON_TOOL_CALLING_MODELS = {
    # 구형 Llama 모델들
    "llama2", "llama2:7b", "llama2:13b", "llama2:70b",
    "llama:7b", "llama:13b", "llama:30b", "llama:65b",
    
    # 구형 Qwen 모델들 (2.5 이전)
    "qwen", "qwen:7b", "qwen:14b", "qwen:32b", "qwen:72b",
    "qwen1.5", "qwen1.5:7b", "qwen1.5:14b", "qwen1.5:32b", "qwen1.5:72b",
    
    # 기타 구형/미지원 모델들
    "vicuna", "vicuna:7b", "vicuna:13b", "vicuna:33b",
    "alpaca", "alpaca:7b", "alpaca:13b",
    "orca-mini", "orca-mini:3b", "orca-mini:7b", "orca-mini:13b",
    "wizard-vicuna-uncensored", "wizard-vicuna-uncensored:7b", "wizard-vicuna-uncensored:13b",
    "falcon", "falcon:7b", "falcon:40b", "falcon:180b",
    "mpt", "mpt:7b", "mpt:30b",
    "chatglm3", "chatglm3:6b",
    "zephyr", "zephyr:7b",
    "openchat", "openchat:7b",
    "dolphin-mistral", "dolphin-mistral:7b"
}

# 🎯 권장 모델 목록 (성능/안정성 기준)
RECOMMENDED_OLLAMA_MODELS = {
    "light": {
        "name": "qwen2.5:3b",
        "description": "가벼운 작업용 (3GB RAM)",
        "ram_requirement": "4GB",
        "use_case": "간단한 질의응답, 코드 생성"
    },
    "balanced": {
        "name": "llama3.1:8b", 
        "description": "균형잡힌 성능 (8GB RAM)",
        "ram_requirement": "10GB",
        "use_case": "데이터 분석, 복잡한 추론"
    },
    "powerful": {
        "name": "qwen2.5:14b",
        "description": "고성능 작업용 (14GB RAM)", 
        "ram_requirement": "16GB",
        "use_case": "고급 분석, 복잡한 코딩"
    },
    "coding": {
        "name": "qwen2.5-coder:7b",
        "description": "코딩 전문 모델 (7GB RAM)",
        "ram_requirement": "9GB", 
        "use_case": "코드 생성, 디버깅, 리팩토링"
    }
}

def get_available_ollama_models() -> List[Dict[str, Any]]:
    """Ollama 서버에서 사용 가능한 모델 목록을 가져옵니다"""
    if not OLLAMA_CLIENT_AVAILABLE:
        return []
    
    try:
        models = ollama.list()
        available_models = []
        
        for model in models.get("models", []):
            model_name = model.get("name", "")
            model_info = {
                "name": model_name,
                "size": model.get("size", 0),
                "modified_at": model.get("modified_at", ""),
                "tool_calling_capable": is_ollama_model_tool_capable(model_name),
                "family": model.get("details", {}).get("family", "unknown")
            }
            available_models.append(model_info)
        
        return available_models
    except Exception as e:
        logging.error(f"Failed to get Ollama models: {e}")
        return []

def test_ollama_connection() -> Dict[str, Any]:
    """Ollama 서버 연결을 테스트합니다"""
    if not OLLAMA_CLIENT_AVAILABLE:
        return {
            "connected": False,
            "error": "Ollama client not available. Install: uv add ollama"
        }
    
    try:
        # 간단한 ping 테스트
        models = ollama.list()
        return {
            "connected": True,
            "model_count": len(models.get("models", [])),
            "server_version": getattr(ollama, "__version__", "unknown")
        }
    except Exception as e:
        return {
            "connected": False,
            "error": f"Connection failed: {str(e)}"
        }

def is_ollama_model_tool_capable(model: str) -> bool:
    """Ollama 모델의 도구 호출 능력 확인 - 엄격한 검증"""
    if not model or not OLLAMA_TOOL_CALLING_SUPPORTED:
        return False
    
    model_lower = model.lower().strip()
    
    # 🚨 명시적으로 지원하지 않는 모델들 확인
    for non_capable_model in OLLAMA_NON_TOOL_CALLING_MODELS:
        if model_lower == non_capable_model.lower() or model_lower.startswith(non_capable_model.split(':')[0].lower()):
            return False
    
    # ✅ 명시적으로 지원하는 모델들 확인
    for capable_model in OLLAMA_TOOL_CALLING_MODELS:
        if model_lower == capable_model.lower() or model_lower.startswith(capable_model.split(':')[0].lower()):
            return True
    
    # 🤔 알 수 없는 모델은 False 반환 (보수적 접근)
    logging.warning(f"⚠️ Unknown model '{model}' - assuming no tool calling support")
    return False

def get_model_recommendation(ram_gb: Optional[int] = None) -> Dict[str, Any]:
    """사용자 시스템에 맞는 모델 추천"""
    if ram_gb is None:
        import psutil
        ram_gb = psutil.virtual_memory().total // (1024**3)
    
    if ram_gb >= 16:
        return RECOMMENDED_OLLAMA_MODELS["powerful"]
    elif ram_gb >= 10:
        return RECOMMENDED_OLLAMA_MODELS["balanced"] 
    elif ram_gb >= 6:
        return RECOMMENDED_OLLAMA_MODELS["light"]
    else:
        return {
            "name": "qwen2.5:0.5b",
            "description": "초경량 모델 (0.5GB RAM)",
            "ram_requirement": "2GB",
            "use_case": "기본적인 텍스트 처리만 가능",
            "warning": "성능이 매우 제한적입니다"
        }

def create_llm_instance(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    callbacks: Optional[List[Any]] = None
) -> Any:
    """
    LLM 인스턴스를 생성합니다.
    
    Args:
        provider: LLM 제공자 ("OPENAI" 또는 "OLLAMA")
        model: 모델명
        temperature: 온도 설정
        callbacks: 추가 콜백 리스트
        
    Returns:
        LLM 인스턴스
    """
    config = Config()
    
    # 환경 변수 기반 설정
    provider = provider or os.getenv("LLM_PROVIDER", "OPENAI")
    temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # 콜백 설정 - langfuse 자동 포함
    callback_list = callbacks or []
    
    # Langfuse 콜백 추가 (환경 변수로 활성화된 경우)
    if LANGFUSE_AVAILABLE and os.getenv("LOGGING_PROVIDER") in ["langfuse", "both"]:
        try:
            langfuse_handler = CallbackHandler(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST"),
            )
            callback_list.append(langfuse_handler)
            logging.info("✅ Langfuse callback automatically added to LLM")
        except Exception as e:
            logging.warning(f"⚠️ Failed to add Langfuse callback: {e}")
    
    if provider.upper() == "OPENAI":
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            callbacks=callback_list if callback_list else None
        )
        
    elif provider.upper() == "OLLAMA":
        if not ChatOllama:
            raise ImportError("Ollama not available. Install: uv add langchain-ollama")
        
        model = model or os.getenv("OLLAMA_MODEL", "llama3:8b")
        base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            callbacks=callback_list if callback_list else None
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_llm_capabilities(llm) -> Dict[str, Any]:
    """LLM의 능력 정보 반환"""
    return {
        "tool_calling_capable": getattr(llm, '_tool_calling_capable', True),
        "provider": getattr(llm, '_provider', 'UNKNOWN'),
        "model_name": getattr(llm, '_model_name', 'unknown'),
        "needs_enhanced_prompting": getattr(llm, '_needs_enhanced_prompting', False)
    }

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
                config["tool_calling_capable"] = True
                
        elif provider == "OLLAMA":
            model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # 🆕 기본값 변경
            config["valid"] = OLLAMA_TOOL_CALLING_SUPPORTED
            config["model"] = model
            config["base_url"] = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            config["tool_calling_capable"] = is_ollama_model_tool_capable(model) and OLLAMA_TOOL_CALLING_SUPPORTED
            config["import_source"] = OLLAMA_IMPORT_SOURCE
            
            if not OLLAMA_TOOL_CALLING_SUPPORTED:
                config["error"] = "langchain-ollama package required for tool calling"
                config["warning"] = "Install: pip install langchain-ollama"
            elif not config["tool_calling_capable"]:
                recommended = get_model_recommendation()
                config["warning"] = f"Model '{model}' has limited tool calling. Recommended: {recommended['name']}"
                config["alternatives"] = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "qwen2.5-coder:7b"]
            
            # 🆕 Ollama 연결 상태 추가 확인
            connection_status = test_ollama_connection()
            config["connection"] = connection_status
            config["available_models"] = get_available_ollama_models()
            
        else:
            config["error"] = f"Unknown provider: {provider}"
            
    except Exception as e:
        config["error"] = str(e)
    
    return config

# 🆕 추가 유틸리티 함수들
def get_ollama_status() -> Dict[str, Any]:
    """Ollama 서버의 전체 상태 정보 반환"""
    status = {
        "client_available": OLLAMA_CLIENT_AVAILABLE,
        "langchain_package": OLLAMA_IMPORT_SOURCE,
        "tool_calling_supported": OLLAMA_TOOL_CALLING_SUPPORTED,
        "connection": test_ollama_connection(),
        "available_models": get_available_ollama_models(),
        "recommended_models": RECOMMENDED_OLLAMA_MODELS,
        "current_model": os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    }
    
    # 현재 모델의 도구 호출 능력 확인
    current_model = status["current_model"]
    status["current_model_tool_capable"] = is_ollama_model_tool_capable(current_model)
    
    return status

def suggest_ollama_setup() -> Dict[str, Any]:
    """사용자를 위한 Ollama 설정 제안"""
    status = get_ollama_status()
    suggestions = {
        "steps": [],
        "commands": [],
        "warnings": [],
        "next_actions": []
    }
    
    # 1. 클라이언트 설치 확인
    if not status["client_available"]:
        suggestions["steps"].append("1. Ollama Python 클라이언트 설치")
        suggestions["commands"].append("uv add ollama")
    
    # 2. 서버 연결 확인  
    if not status["connection"]["connected"]:
        suggestions["steps"].append("2. Ollama 서버 시작")
        suggestions["commands"].append("ollama serve")
        suggestions["warnings"].append("Ollama 서버가 실행 중인지 확인하세요")
    
    # 3. 모델 설치 확인
    available_models = status["available_models"]
    if not available_models:
        suggestions["steps"].append("3. 추천 모델 다운로드")
        recommended = get_model_recommendation()
        suggestions["commands"].append(f"ollama pull {recommended['name']}")
        suggestions["next_actions"].append(f"환경변수 설정: OLLAMA_MODEL={recommended['name']}")
    
    # 4. 도구 호출 지원 확인
    if not status["current_model_tool_capable"]:
        suggestions["warnings"].append(f"현재 모델 '{status['current_model']}'은 도구 호출을 지원하지 않습니다")
        recommended = get_model_recommendation()
        suggestions["next_actions"].append(f"권장 모델로 변경: {recommended['name']}")
        suggestions["commands"].append(f"ollama pull {recommended['name']}")
    
    return suggestions