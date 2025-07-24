"""
LLM Factory for Universal Engine - LLM 클라이언트 생성 및 관리

요구사항 5.1에 따른 구현:
- 설정 기반 LLM 클라이언트 생성
- 사용 가능한 모델 목록 반환
- 모델 설정 유효성 검증
- Ollama, OpenAI, Anthropic 지원
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv가 없는 경우 무시
    pass

# 의존성 제거 - 필요한 함수들을 직접 구현

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Universal Engine용 LLM Factory
    - 설정 기반 LLM 클라이언트 생성
    - 모델 검증 및 추천
    - 다중 제공자 지원 (OpenAI, Ollama, Anthropic)
    """
    
    # 지원되는 LLM 제공자
    SUPPORTED_PROVIDERS = ["openai", "ollama", "anthropic"]
    
    # 기본 설정 - .env 환경 변수 우선 사용 (LLM First 원칙)
    @classmethod
    def get_default_configs(cls) -> Dict[str, Dict[str, Any]]:
        """환경 변수를 우선하는 기본 설정 반환"""
        return {
            "openai": {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
                "api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            },
            "ollama": {
                "model": os.getenv("OLLAMA_MODEL", "qwen3-4b-fast:latest"),
                "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            },
            "anthropic": {
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "4000"))
            }
        }
    
    # 하위 호환성을 위한 속성
    @property
    def DEFAULT_CONFIGS(self) -> Dict[str, Dict[str, Any]]:
        return self.get_default_configs()
    
    @staticmethod
    def create_llm(**kwargs) -> Any:
        """
        create_llm 호환성 메서드 - create_llm_client의 별칭
        기존 코드와의 호환성을 위한 메서드
        """
        return LLMFactory.create_llm_client(**kwargs)
    
    @staticmethod
    def create_llm_client(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        설정 기반 LLM 클라이언트 생성
        
        Args:
            provider: LLM 제공자 ("openai", "ollama", "anthropic")
            model: 모델명
            config: 추가 설정
            **kwargs: 기타 설정
            
        Returns:
            LLM 클라이언트 인스턴스
            
        Raises:
            ValueError: 지원하지 않는 제공자
            ImportError: 필요한 패키지가 설치되지 않음
        """
        logger.info(f"Creating LLM client: provider={provider}, model={model}")
        
        try:
            # 기본값 설정 - 환경 변수 우선 (LLM First 원칙)
            env_provider = os.getenv("LLM_PROVIDER", "openai").lower()
            provider = provider or env_provider
            
            logger.info(f"Using provider: {provider} (from env: {env_provider})")
            
            if provider not in LLMFactory.SUPPORTED_PROVIDERS:
                raise ValueError(f"Unsupported provider: {provider}. Supported: {LLMFactory.SUPPORTED_PROVIDERS}")
            
            # 제공자별 기본 설정 적용 - 환경 변수 우선 (LLM First 원칙)
            default_configs = LLMFactory.get_default_configs()
            default_config = default_configs.get(provider, {}).copy()
            if config:
                default_config.update(config)
            default_config.update(kwargs)
            
            # 모델 설정
            if model:
                default_config["model"] = model
            elif not default_config.get("model"):
                default_config["model"] = LLMFactory._get_default_model(provider)
            
            # 제공자별 클라이언트 생성
            if provider == "openai":
                return LLMFactory._create_openai_client(default_config)
            elif provider == "ollama":
                return LLMFactory._create_ollama_client(default_config)
            elif provider == "anthropic":
                return LLMFactory._create_anthropic_client(default_config)
            
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            raise
    
    @staticmethod
    def _create_openai_client(config: Dict[str, Any]) -> Any:
        """OpenAI 클라이언트 생성"""
        try:
            from langchain_openai import ChatOpenAI
            
            return ChatOpenAI(
                model=config.get("model", "gpt-4o-mini"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4000),
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                base_url=config.get("api_base") or os.getenv("OPENAI_API_BASE"),
                callbacks=config.get("callbacks")
            )
        except ImportError:
            raise ImportError("langchain-openai package required. Install: pip install langchain-openai")
    
    @staticmethod
    def _create_ollama_client(config: Dict[str, Any]) -> Any:
        """Ollama 클라이언트 생성"""
        try:
            # langchain_ollama 우선 사용 (도구 호출 지원)
            from langchain_ollama import ChatOllama
            
            # 환경 변수에서 OLLAMA_MODEL 확인 (LLM First 원칙)
            env_model = os.getenv("OLLAMA_MODEL", "qwen3-4b-fast:latest")
            model = config.get("model", env_model)
            
            # 모델의 도구 호출 능력 확인
            if not LLMFactory._is_ollama_model_tool_capable(model):
                logger.warning(f"Model '{model}' may have limited tool calling support")
                recommended = LLMFactory._get_model_recommendation()
                logger.info(f"Consider using recommended model: {recommended['name']}")
            
            return ChatOllama(
                model=model,
                temperature=config.get("temperature", 0.7),
                base_url=config.get("base_url") or os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
                callbacks=config.get("callbacks")
            )
            
        except ImportError:
            try:
                # 폴백: langchain_community 사용
                from langchain_community.chat_models.ollama import ChatOllama
                logger.warning("Using deprecated langchain_community.ChatOllama - Tool calling NOT supported")
                
                # 환경 변수에서 OLLAMA_MODEL 확인 (LLM First 원칙)
                env_model = os.getenv("OLLAMA_MODEL", "qwen3-4b-fast:latest")
                return ChatOllama(
                    model=config.get("model", env_model),
                    temperature=config.get("temperature", 0.7),
                    base_url=config.get("base_url") or os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
                    callbacks=config.get("callbacks")
                )
            except ImportError:
                raise ImportError("Ollama package required. Install: pip install langchain-ollama")
    
    @staticmethod
    def _create_anthropic_client(config: Dict[str, Any]) -> Any:
        """Anthropic 클라이언트 생성"""
        try:
            from langchain_anthropic import ChatAnthropic
            
            return ChatAnthropic(
                model=config.get("model", "claude-3-haiku-20240307"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 4000),
                api_key=config.get("api_key") or os.getenv("ANTHROPIC_API_KEY"),
                callbacks=config.get("callbacks")
            )
        except ImportError:
            raise ImportError("langchain-anthropic package required. Install: pip install langchain-anthropic")
    
    @staticmethod
    def get_available_models(provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        사용 가능한 모델 목록 반환
        
        Args:
            provider: 특정 제공자의 모델만 조회 (선택사항)
            
        Returns:
            모델 정보 리스트
        """
        logger.info(f"Getting available models for provider: {provider}")
        
        models = []
        providers_to_check = [provider] if provider else LLMFactory.SUPPORTED_PROVIDERS
        
        for prov in providers_to_check:
            try:
                if prov == "openai":
                    models.extend(LLMFactory._get_openai_models())
                elif prov == "ollama":
                    models.extend(LLMFactory._get_ollama_models())
                elif prov == "anthropic":
                    models.extend(LLMFactory._get_anthropic_models())
            except Exception as e:
                logger.warning(f"Failed to get models for {prov}: {e}")
        
        return models
    
    @staticmethod
    def _get_openai_models() -> List[Dict[str, Any]]:
        """OpenAI 모델 목록"""
        return [
            {
                "provider": "openai",
                "name": "gpt-4o",
                "description": "Most capable GPT-4 model",
                "context_length": 128000,
                "tool_calling": True,
                "cost_tier": "high"
            },
            {
                "provider": "openai", 
                "name": "gpt-4o-mini",
                "description": "Affordable and intelligent small model",
                "context_length": 128000,
                "tool_calling": True,
                "cost_tier": "low"
            },
            {
                "provider": "openai",
                "name": "gpt-4-turbo",
                "description": "Previous generation GPT-4 Turbo",
                "context_length": 128000,
                "tool_calling": True,
                "cost_tier": "medium"
            },
            {
                "provider": "openai",
                "name": "gpt-3.5-turbo",
                "description": "Fast and affordable model",
                "context_length": 16385,
                "tool_calling": True,
                "cost_tier": "low"
            }
        ]
    
    @staticmethod
    def _get_ollama_models() -> List[Dict[str, Any]]:
        """Ollama 모델 목록"""
        models = []
        
        # 로컬에서 사용 가능한 모델 조회
        available_models = LLMFactory._get_available_ollama_models()
        
        if available_models:
            for model in available_models:
                models.append({
                    "provider": "ollama",
                    "name": model["name"],
                    "description": f"Local Ollama model ({model.get('family', 'unknown')} family)",
                    "size": model.get("size", 0),
                    "tool_calling": model.get("tool_calling_capable", False),
                    "cost_tier": "free",
                    "local": True
                })
        else:
            # 사용 가능한 모델이 없으면 추천 모델 목록 반환
            recommended_models = LLMFactory._get_recommended_ollama_models()
            for category, info in recommended_models.items():
                models.append({
                    "provider": "ollama",
                    "name": info["name"],
                    "description": info["description"],
                    "ram_requirement": info["ram_requirement"],
                    "use_case": info["use_case"],
                    "tool_calling": LLMFactory._is_ollama_model_tool_capable(info["name"]),
                    "cost_tier": "free",
                    "local": False,
                    "category": category
                })
        
        return models
    
    @staticmethod
    def _get_anthropic_models() -> List[Dict[str, Any]]:
        """Anthropic 모델 목록"""
        return [
            {
                "provider": "anthropic",
                "name": "claude-3-5-sonnet-20241022",
                "description": "Most intelligent Claude model",
                "context_length": 200000,
                "tool_calling": True,
                "cost_tier": "high"
            },
            {
                "provider": "anthropic",
                "name": "claude-3-haiku-20240307",
                "description": "Fastest and most compact Claude model",
                "context_length": 200000,
                "tool_calling": True,
                "cost_tier": "low"
            },
            {
                "provider": "anthropic",
                "name": "claude-3-opus-20240229",
                "description": "Most powerful Claude model",
                "context_length": 200000,
                "tool_calling": True,
                "cost_tier": "premium"
            }
        ]
    
    @staticmethod
    def validate_model_config(
        provider: str,
        model: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        모델 설정 유효성 검증
        
        Args:
            provider: LLM 제공자
            model: 모델명
            config: 추가 설정
            
        Returns:
            검증 결과
        """
        logger.info(f"Validating model config: {provider}/{model}")
        
        validation_result = {
            "valid": False,
            "provider": provider,
            "model": model,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "capabilities": {},
            "estimated_cost": "unknown"
        }
        
        try:
            # 제공자 검증
            if provider not in LLMFactory.SUPPORTED_PROVIDERS:
                validation_result["errors"].append(f"Unsupported provider: {provider}")
                return validation_result
            
            # 제공자별 검증
            if provider == "openai":
                validation_result.update(LLMFactory._validate_openai_config(model, config))
            elif provider == "ollama":
                validation_result.update(LLMFactory._validate_ollama_config(model, config))
            elif provider == "anthropic":
                validation_result.update(LLMFactory._validate_anthropic_config(model, config))
            
            # 전체 검증 상태 결정
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
            logger.info(f"Validation result: {validation_result['valid']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            validation_result["errors"].append(f"Validation failed: {str(e)}")
            return validation_result
    
    @staticmethod
    def _validate_openai_config(model: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """OpenAI 설정 검증"""
        result = {
            "capabilities": {"tool_calling": True, "streaming": True},
            "estimated_cost": "medium"
        }
        
        # API 키 확인
        api_key = (config or {}).get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            result.setdefault("errors", []).append("OPENAI_API_KEY not configured")
        
        # 모델 검증
        openai_models = [m["name"] for m in LLMFactory._get_openai_models()]
        if model not in openai_models:
            result.setdefault("warnings", []).append(f"Model '{model}' not in known OpenAI models")
        
        # 비용 추정
        if "gpt-4o" in model:
            result["estimated_cost"] = "high"
        elif "gpt-4o-mini" in model or "gpt-3.5" in model:
            result["estimated_cost"] = "low"
        
        return result
    
    @staticmethod
    def _validate_ollama_config(model: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ollama 설정 검증"""
        result = {
            "capabilities": {
                "tool_calling": LLMFactory._is_ollama_model_tool_capable(model),
                "streaming": True,
                "local": True
            },
            "estimated_cost": "free"
        }
        
        # Ollama 서버 연결 확인
        ollama_status = LLMFactory._get_ollama_status()
        if not ollama_status["connection"]["connected"]:
            result.setdefault("errors", []).append("Ollama server not accessible")
            result.setdefault("recommendations", []).append("Start Ollama server: ollama serve")
        
        # 모델 설치 확인
        available_models = [m["name"] for m in ollama_status["available_models"]]
        if model not in available_models:
            result.setdefault("warnings", []).append(f"Model '{model}' not installed locally")
            result.setdefault("recommendations", []).append(f"Install model: ollama pull {model}")
        
        # 도구 호출 능력 확인
        if not result["capabilities"]["tool_calling"]:
            result.setdefault("warnings", []).append(f"Model '{model}' has limited tool calling support")
            recommended = LLMFactory._get_model_recommendation()
            result.setdefault("recommendations", []).append(f"Consider using: {recommended['name']}")
        
        return result
    
    @staticmethod
    def _validate_anthropic_config(model: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Anthropic 설정 검증"""
        result = {
            "capabilities": {"tool_calling": True, "streaming": True},
            "estimated_cost": "medium"
        }
        
        # API 키 확인
        api_key = (config or {}).get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            result.setdefault("errors", []).append("ANTHROPIC_API_KEY not configured")
        
        # 모델 검증
        anthropic_models = [m["name"] for m in LLMFactory._get_anthropic_models()]
        if model not in anthropic_models:
            result.setdefault("warnings", []).append(f"Model '{model}' not in known Anthropic models")
        
        # 비용 추정
        if "opus" in model:
            result["estimated_cost"] = "premium"
        elif "haiku" in model:
            result["estimated_cost"] = "low"
        elif "sonnet" in model:
            result["estimated_cost"] = "high"
        
        return result
    
    @staticmethod
    def _get_default_model(provider: str) -> str:
        """제공자별 기본 모델 반환 - 환경 변수 우선 (LLM First 원칙)"""
        defaults = {
            "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "ollama": os.getenv("OLLAMA_MODEL", "qwen3-4b-fast:latest"),
            "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        }
        return defaults.get(provider, os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    @staticmethod
    def get_system_recommendations() -> Dict[str, Any]:
        """
        시스템 환경에 맞는 LLM 추천
        
        Returns:
            추천 정보
        """
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "primary_recommendation": {},
            "alternatives": [],
            "setup_instructions": []
        }
        
        try:
            # 시스템 리소스 확인
            import psutil
            ram_gb = psutil.virtual_memory().total // (1024**3)
            
            # 환경 변수 확인 (LLM First 원칙)
            env_provider = os.getenv("LLM_PROVIDER", "").upper()
            env_model = os.getenv("OLLAMA_MODEL", "qwen3-4b-fast:latest")
            
            # Ollama 상태 확인
            ollama_status = LLMFactory._get_ollama_status()
            
            # LLM_PROVIDER=OLLAMA인 경우 처음부터 OLLAMA 우선 추천
            if env_provider == "OLLAMA":
                if ollama_status["connection"]["connected"]:
                    # Ollama 서버가 동작하는 경우
                    recommendations["primary_recommendation"] = {
                        "provider": "ollama",
                        "model": env_model,
                        "reason": "LLM_PROVIDER=OLLAMA is set - using configured Ollama model",
                        "setup_required": False,
                        "estimated_cost": "free"
                    }
                    
                    # 대안으로 OpenAI 제공
                    recommendations["alternatives"].append({
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "reason": "Cloud-based fallback option",
                        "setup_required": True,
                        "estimated_cost": "low"
                    })
                else:
                    # Ollama 서버가 동작하지 않는 경우 - 폴백으로 OpenAI 추천하지만 안내 메시지 추가
                    recommendations["primary_recommendation"] = {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "reason": "LLM_PROVIDER=OLLAMA is set but Ollama server not accessible - using OpenAI fallback",
                        "setup_required": True,
                        "estimated_cost": "low"
                    }
                    
                    recommendations["alternatives"].append({
                        "provider": "ollama",
                        "model": env_model,
                        "reason": "Preferred option (start Ollama server)",
                        "setup_required": True,
                        "estimated_cost": "free"
                    })
                    
                    recommendations["setup_instructions"] = [
                        f"LLM_PROVIDER=OLLAMA is configured but Ollama server is not accessible",
                        f"To use Ollama: Start server with 'ollama serve' and pull model '{env_model}'",
                        "For OpenAI fallback: Set OPENAI_API_KEY environment variable"
                    ]
            
            elif ollama_status["connection"]["connected"] and ollama_status["available_models"]:
                # LLM_PROVIDER가 OLLAMA가 아니지만 Ollama가 사용 가능한 경우
                recommended_model = LLMFactory._get_model_recommendation(ram_gb)
                recommendations["primary_recommendation"] = {
                    "provider": "ollama",
                    "model": recommended_model["name"],
                    "reason": "Local, free, and privacy-friendly",
                    "setup_required": False,
                    "estimated_cost": "free"
                }
                
                # 대안 제공
                recommendations["alternatives"].append({
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "reason": "Cloud-based, highly capable",
                    "setup_required": True,
                    "estimated_cost": "low"
                })
                
            else:
                # Ollama가 사용 불가능한 경우 OpenAI 추천
                recommendations["primary_recommendation"] = {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "reason": "Reliable cloud service, good performance",
                    "setup_required": True,
                    "estimated_cost": "low"
                }
                
                # Ollama 설정 안내 - 환경 변수 모델 사용
                recommendations["alternatives"].append({
                    "provider": "ollama",
                    "model": env_model,
                    "reason": "Free local alternative",
                    "setup_required": True,
                    "estimated_cost": "free"
                })
                
                recommendations["setup_instructions"] = [
                    "For Ollama: Install Ollama server and pull a model",
                    "For OpenAI: Set OPENAI_API_KEY environment variable"
                ]
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            # 폴백 추천
            recommendations["primary_recommendation"] = {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "reason": "Default reliable option",
                "setup_required": True,
                "estimated_cost": "low"
            }
        
        return recommendations
    
    # === 내부 헬퍼 함수들 (의존성 제거를 위해 직접 구현) ===
    
    @staticmethod
    def _is_ollama_model_tool_capable(model: str) -> bool:
        """
        Ollama 모델의 도구 호출 능력 확인
        """
        # 도구 호출을 지원하는 것으로 알려진 모델들 (LLM First 원칙)
        tool_capable_models = [
            "okamototk/gemma3-tools:4b",
            "llama3.1:8b",
            "llama3.1:70b",
            "llama3.2:3b",
            "mistral:7b",
            "qwen2.5:7b",
            "qwen3-4b-fast:latest",  # .env에서 설정된 최적화 모델
            "gemma2:9b"
        ]
        
        # 모델명이 도구 호출 지원 목록에 있는지 확인
        for capable_model in tool_capable_models:
            if model.lower() == capable_model.lower():
                return True
        
        # 모델명에 "tools"가 포함되어 있으면 도구 호출 지원으로 간주
        if "tools" in model.lower():
            return True
        
        # 환경 변수에서 도구 호출 지원 여부 확인 (LLM First 원칙)
        env_tool_calling = os.getenv("OLLAMA_TOOL_CALLING_SUPPORTED", "false").lower()
        if env_tool_calling in ["true", "1", "yes"]:
            return True
            
        return False
    
    @staticmethod
    def _get_model_recommendation(ram_gb: int = 8) -> Dict[str, Any]:
        """
        시스템 리소스에 맞는 모델 추천
        """
        env_model = os.getenv("OLLAMA_MODEL", "okamototk/gemma3-tools:4b")
        
        # 환경 변수에 모델이 설정되어 있으면 우선 사용
        if env_model:
            return {
                "name": env_model,
                "reason": "Configured in OLLAMA_MODEL environment variable",
                "ram_requirement": "Unknown",
                "tool_calling": LLMFactory._is_ollama_model_tool_capable(env_model)
            }
        
        # RAM 기반 추천
        if ram_gb >= 32:
            return {
                "name": "llama3.1:70b",
                "reason": "High RAM available - using large model",
                "ram_requirement": "32GB+",
                "tool_calling": True
            }
        elif ram_gb >= 16:
            return {
                "name": "llama3.1:8b",
                "reason": "Medium RAM available - using medium model",
                "ram_requirement": "16GB+",
                "tool_calling": True
            }
        else:
            return {
                "name": "okamototk/gemma3-tools:4b",
                "reason": "Limited RAM - using efficient model with tool support",
                "ram_requirement": "8GB+",
                "tool_calling": True
            }
    
    @staticmethod
    def _get_available_ollama_models() -> List[Dict[str, Any]]:
        """
        로컬에서 사용 가능한 Ollama 모델 목록 조회
        """
        try:
            import subprocess
            import json
            
            # ollama list 명령어 실행
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                models = []
                lines = result.stdout.strip().split('\n')[1:]  # 헤더 제외
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            model_name = parts[0]
                            size = parts[2] if len(parts) > 2 else "Unknown"
                            
                            models.append({
                                "name": model_name,
                                "size": size,
                                "family": "unknown",
                                "tool_calling_capable": LLMFactory._is_ollama_model_tool_capable(model_name)
                            })
                
                return models
            else:
                logger.warning(f"Failed to get Ollama models: {result.stderr}")
                return []
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Could not query Ollama models: {e}")
            return []
    
    @staticmethod
    def _get_ollama_status() -> Dict[str, Any]:
        """
        Ollama 서버 상태 확인
        """
        status = {
            "connection": {"connected": False, "error": None},
            "available_models": [],
            "server_info": {}
        }
        
        try:
            import requests
            
            # Ollama 서버 연결 확인
            base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                status["connection"]["connected"] = True
                
                # 사용 가능한 모델 목록 가져오기
                data = response.json()
                models = []
                
                for model_info in data.get("models", []):
                    models.append({
                        "name": model_info.get("name", "unknown"),
                        "size": model_info.get("size", 0),
                        "modified_at": model_info.get("modified_at", ""),
                        "tool_calling_capable": LLMFactory._is_ollama_model_tool_capable(
                            model_info.get("name", "")
                        )
                    })
                
                status["available_models"] = models
            else:
                status["connection"]["error"] = f"HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            status["connection"]["error"] = str(e)
        except Exception as e:
            status["connection"]["error"] = f"Unexpected error: {e}"
        
        return status
    
    @staticmethod
    def _get_recommended_ollama_models() -> Dict[str, Dict[str, Any]]:
        """
        추천 Ollama 모델 목록
        """
        return {
            "efficient": {
                "name": "okamototk/gemma3-tools:4b",
                "description": "Efficient model with tool calling support",
                "ram_requirement": "8GB",
                "use_case": "General purpose with limited resources"
            },
            "balanced": {
                "name": "llama3.1:8b",
                "description": "Balanced performance and resource usage",
                "ram_requirement": "16GB",
                "use_case": "Good balance of capability and efficiency"
            },
            "powerful": {
                "name": "llama3.1:70b",
                "description": "High-performance model for complex tasks",
                "ram_requirement": "32GB+",
                "use_case": "Complex reasoning and analysis"
            },
            "coding": {
                "name": "qwen2.5:7b",
                "description": "Optimized for coding tasks",
                "ram_requirement": "12GB",
                "use_case": "Code generation and analysis"
            }
        }