#!/usr/bin/env python3
"""
독립적인 LLMFactory 구현 - 의존성 없음
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class IndependentLLMFactory:
    """
    독립적인 LLM Factory - 외부 의존성 없음
    - 설정 기반 LLM 클라이언트 생성
    - 모델 검증 및 추천
    - 다중 제공자 지원 (OpenAI, Ollama, Anthropic)
    """
    
    # 지원되는 LLM 제공자
    SUPPORTED_PROVIDERS = ["openai", "ollama", "anthropic"]
    
    # 기본 설정
    DEFAULT_CONFIGS = {
        "openai": {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 4000,
            "api_base": "https://api.openai.com/v1"
        },
        "ollama": {
            "model": "okamototk/gemma3-tools:4b",
            "temperature": 0.7,
            "base_url": "http://localhost:11434"
        },
        "anthropic": {
            "model": "claude-3-haiku-20240307",
            "temperature": 0.7,
            "max_tokens": 4000
        }
    }
    
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
            # 기본값 설정 - LLM_PROVIDER=OLLAMA인 경우 처음부터 OLLAMA 사용
            env_provider = os.getenv("LLM_PROVIDER", "").upper()
            if env_provider == "OLLAMA":
                provider = provider or "ollama"
            else:
                provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()
            
            if provider not in IndependentLLMFactory.SUPPORTED_PROVIDERS:
                raise ValueError(f"Unsupported provider: {provider}. Supported: {IndependentLLMFactory.SUPPORTED_PROVIDERS}")
            
            # 제공자별 기본 설정 적용
            default_config = IndependentLLMFactory.DEFAULT_CONFIGS.get(provider, {}).copy()
            if config:
                default_config.update(config)
            default_config.update(kwargs)
            
            # 모델 설정
            if model:
                default_config["model"] = model
            elif not default_config.get("model"):
                default_config["model"] = IndependentLLMFactory._get_default_model(provider)
            
            # 제공자별 클라이언트 생성
            if provider == "openai":
                return IndependentLLMFactory._create_openai_client(default_config)
            elif provider == "ollama":
                return IndependentLLMFactory._create_ollama_client(default_config)
            elif provider == "anthropic":
                return IndependentLLMFactory._create_anthropic_client(default_config)
            
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
            
            # 환경 변수에서 OLLAMA_MODEL 확인 (기본값: okamototk/gemma3-tools:4b)
            env_model = os.getenv("OLLAMA_MODEL", "okamototk/gemma3-tools:4b")
            model = config.get("model", env_model)
            
            # 모델의 도구 호출 능력 확인
            if not IndependentLLMFactory._is_ollama_model_tool_capable(model):
                logger.warning(f"Model '{model}' may have limited tool calling support")
                recommended = IndependentLLMFactory._get_model_recommendation()
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
                
                # 환경 변수에서 OLLAMA_MODEL 확인 (기본값: okamototk/gemma3-tools:4b)
                env_model = os.getenv("OLLAMA_MODEL", "okamototk/gemma3-tools:4b")
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
    def _get_default_model(provider: str) -> str:
        """제공자별 기본 모델 반환"""
        defaults = {
            "openai": "gpt-4o-mini",
            "ollama": os.getenv("OLLAMA_MODEL", "okamototk/gemma3-tools:4b"),
            "anthropic": "claude-3-haiku-20240307"
        }
        return defaults.get(provider, "gpt-4o-mini")
    
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
            
            # 환경 변수 확인
            env_provider = os.getenv("LLM_PROVIDER", "").upper()
            env_model = os.getenv("OLLAMA_MODEL", "okamototk/gemma3-tools:4b")
            
            # Ollama 상태 확인
            ollama_status = IndependentLLMFactory._get_ollama_status()
            
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
                recommended_model = IndependentLLMFactory._get_model_recommendation(ram_gb)
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
        # 도구 호출을 지원하는 것으로 알려진 모델들
        tool_capable_models = [
            "okamototk/gemma3-tools:4b",
            "llama3.1:8b",
            "llama3.1:70b",
            "llama3.2:3b",
            "mistral:7b",
            "qwen2.5:7b",
            "gemma2:9b"
        ]
        
        # 모델명이 도구 호출 지원 목록에 있는지 확인
        for capable_model in tool_capable_models:
            if model.lower() == capable_model.lower():
                return True
        
        # 모델명에 "tools"가 포함되어 있으면 도구 호출 지원으로 간주
        if "tools" in model.lower():
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
                "tool_calling": IndependentLLMFactory._is_ollama_model_tool_capable(env_model)
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
                                "tool_calling_capable": IndependentLLMFactory._is_ollama_model_tool_capable(model_name)
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
                        "tool_calling_capable": IndependentLLMFactory._is_ollama_model_tool_capable(
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


# 테스트 함수
def test_independent_llm_factory():
    """독립적인 LLMFactory 테스트"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== 독립적인 LLMFactory 테스트 ===\n")
    
    print("1. 환경 변수:")
    print(f"   LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"   OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL')}")
    print()
    
    print("2. 기본 설정:")
    print(f"   DEFAULT_CONFIGS['ollama']['model']: {IndependentLLMFactory.DEFAULT_CONFIGS['ollama']['model']}")
    print()
    
    print("3. 기본 모델 반환:")
    default_model = IndependentLLMFactory._get_default_model("ollama")
    print(f"   _get_default_model('ollama'): {default_model}")
    print()
    
    print("4. 시스템 추천:")
    try:
        recommendations = IndependentLLMFactory.get_system_recommendations()
        primary = recommendations['primary_recommendation']
        print(f"   Provider: {primary['provider']}")
        print(f"   Model: {primary['model']}")
        print(f"   Reason: {primary['reason']}")
        print()
    except Exception as e:
        print(f"   추천 시스템 오류: {e}")
        print()
    
    print("5. 클라이언트 생성 테스트:")
    try:
        # provider 지정 없이 생성 (환경 변수 기반)
        client = IndependentLLMFactory.create_llm_client()
        print(f"   ✅ 성공: {type(client).__name__}")
        
        # 모델 정보 확인
        model_info = "Unknown"
        if hasattr(client, 'model'):
            model_info = client.model
        elif hasattr(client, 'model_name'):
            model_info = client.model_name
        print(f"   모델: {model_info}")
        
    except Exception as e:
        print(f"   ❌ 실패: {e}")
    
    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    test_independent_llm_factory()